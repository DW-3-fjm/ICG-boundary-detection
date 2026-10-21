"""Regression checks for incomplete contours, HD95 and export-independent states.

Fixtures are synthetic software checks, not clinical validation.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from icg_pipeline.eval import Evaluator, evaluate_tracking
from icg_pipeline.inference import ICGPredictor
from icg_pipeline.tracking import GatedContourTracker, OpticalFlowTracker
from icg_pipeline.utils import write_json


def textured_frame():
    return np.random.default_rng(421).integers(20, 230, (128, 128, 3), dtype=np.uint8)


def circle():
    angles = np.arange(100) * (2 * np.pi / 100)
    return np.column_stack((64 + 45 * np.cos(angles), 64 + 45 * np.sin(angles))).astype(np.float32)


@pytest.mark.parametrize('policy', ['legacy', 'gated'])
@pytest.mark.parametrize('lost_points', [1, 19])
def test_partial_point_loss_never_becomes_a_smaller_closed_contour(monkeypatch, policy, lost_points):
    frame = textured_frame()
    if policy == 'gated':
        tracker = GatedContourTracker(working_size=(128, 128), num_points=100)
        tracker.update(frame, circle(), .99)
        flow = tracker.flow
    else:
        tracker = flow = OpticalFlowTracker(num_points=100, min_valid_fraction=.8)
        flow.initialize(frame, circle())
    locations = np.rint(flow.points[:lost_points]).astype(np.int32)
    calls = 0

    def inconsistent_flow(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = np.zeros((128, 128, 2), np.float32)
        if calls == 2:  # Only the chosen backward-flow samples fail consistency.
            result[locations[:, 1], locations[:, 0], 0] = 10
        return result

    monkeypatch.setattr(cv2, 'calcOpticalFlowFarneback', inconsistent_flow)
    result = tracker.update(frame) if policy == 'gated' else tracker.track(frame)
    points = result['tracked_contour'] if policy == 'gated' else result.points
    status = result['tracking_status'] if policy == 'gated' else result.status
    fraction = result['valid_point_fraction'] if policy == 'gated' else result.valid_point_fraction
    assert fraction == pytest.approx((100 - lost_points) / 100)
    assert points is None
    assert 'incomplete_contour' in status
    assert flow.points is None
    if policy == 'gated':
        # Lost identity remains explicit until an authorized new initialization.
        assert tracker.update(frame, circle(), .99)['tracked_contour'] is None
        tracker.reset()
        assert len(tracker.update(frame, circle(), .99)['tracked_contour']) == 100


def test_complete_flow_preserves_point_count_and_geometry_over_many_frames(monkeypatch):
    frame = textured_frame()
    tracker = GatedContourTracker(working_size=(128, 128), num_points=100)
    initial = tracker.update(frame, circle(), .99)['tracked_contour'].copy()
    monkeypatch.setattr(cv2, 'calcOpticalFlowFarneback',
                        lambda *args, **kwargs: np.zeros((128, 128, 2), np.float32))
    for _ in range(10):
        result = tracker.update(frame)
        assert result['tracking_status'] == 'propagated'
        assert result['valid_point_fraction'] == 1.
        assert np.array_equal(result['tracked_contour'], initial)


def square():
    return np.array([[20, 20], [80, 20], [80, 80], [20, 80]], np.float32)


def test_tracking_reports_mean_of_per_frame_hd95_and_keeps_failures_in_success_rate():
    contour = square()
    shifted = contour + np.array([3, 0], np.float32)
    direct = evaluate_tracking([contour, shifted], [contour, contour])
    result = Evaluator({}).evaluate_tracking_sequence(
        [{'tracked_contour': contour, 'tracking_status': 'segmentation_initialized'},
         {'tracked_contour': shifted, 'tracking_status': 'propagated'},
         {'tracked_contour': None, 'tracking_status': 'tracking_failed:quality'}],
        [contour, contour, contour])
    for metrics in (direct, result):
        assert metrics['mean_hausdorff_95_contour_distance'] == pytest.approx(1.5)
    assert result['tracking_success_rate'] == pytest.approx(2 / 3)
    assert result['tracking_failure_frequency'] == pytest.approx(1 / 3)
    assert result['distance_metrics_condition'] == 'successful_annotated_frames_only'


@pytest.mark.parametrize('scenario', ['all_failed', 'unannotated', 'empty'])
def test_tracking_metric_schema_is_stable_without_successful_references(tmp_path, scenario):
    contour = square()
    good = Evaluator({}).evaluate_tracking_sequence(
        [{'tracked_contour': contour, 'tracking_status': 'propagated'}], [contour])
    rows = [{'tracked_contour': None, 'tracking_status': 'tracking_failed:quality'}]
    references = [contour] if scenario == 'all_failed' else [None]
    if scenario == 'empty':
        rows, references = [], []
    result = Evaluator({}).evaluate_tracking_sequence(rows, references)
    assert set(result) == set(good)
    assert np.isnan(result['mean_hausdorff_95_contour_distance'])
    assert result['evaluated_frame_count'] == 0
    if scenario == 'all_failed':
        assert result['tracking_success_rate'] == 0
    else:
        assert np.isnan(result['tracking_success_rate'])
    path = tmp_path / 'metrics.json'
    write_json(path, result)
    saved = json.loads(path.read_text(encoding='utf8'))
    assert saved['mean_hausdorff_95_contour_distance'] is None
    assert 'NaN' not in path.read_text(encoding='utf8')


def predictor_config(tmp_path):
    return {'device': {'use_cuda': False},
            'model': {'in_channels': 3, 'out_channels': 2},
            'data': {'image_size': [32, 32], 'in_channels': 3},
            'inference': {'output_dir': str(tmp_path / 'predictions'), 'save_overlay': False},
            'tracking': {'enabled': True, 'correction_policy': 'gated',
                         'working_size': [128, 128], 'min_probability': .75}}


@pytest.mark.parametrize('scenario', ['uninitialized', 'failure_and_latch'])
def test_exporting_video_preserves_algorithm_states_and_metadata(tmp_path, monkeypatch, scenario):
    frame = textured_frame()
    frames = [frame, frame] if scenario == 'uninitialized' else [frame, np.zeros_like(frame), frame]
    source = tmp_path / 'source.mp4'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (128, 128))
    if not writer.isOpened():
        pytest.skip('OpenCV video codec unavailable')
    for item in frames:
        writer.write(item)
    writer.release()
    predictor = ICGPredictor(predictor_config(tmp_path), model=torch.nn.Identity())

    def observation(_frame):
        return {'mask': np.zeros((128, 128), np.uint8),
                'probability_map': np.zeros((128, 128), np.float32),
                'contours': [], 'contour_points': [], 'selected_contour': square(),
                'segmentation_probability': .60 if scenario == 'uninitialized' else .99,
                'original_size': (128, 128)}

    monkeypatch.setattr(predictor, 'predict_image', observation)
    raw = predictor.predict_video(str(source))
    exported = predictor.predict_video(str(source), str(tmp_path / 'export.mp4'))
    raw_states = [r['tracking_status'] for r in raw['frames']]
    assert [r['tracking_status'] for r in exported['frames']] == raw_states
    metadata = json.loads(Path(exported['metadata_path']).read_text(encoding='utf8'))
    assert [r['tracking_status'] for r in metadata['frames']] == raw_states
    assert len(raw_states) == len(frames)
    if scenario == 'uninitialized':
        assert raw_states == ['uninitialized_no_reliable_segmentation'] * 2
    else:
        assert raw_states[1] == 'tracking_failed:failed_quality_check'
        assert raw_states[2] == 'tracking_failed:latched_until_reset'


def test_uninitialized_overlay_shows_waiting_without_drawing_a_supported_boundary(tmp_path, monkeypatch):
    predictor = ICGPredictor(predictor_config(tmp_path), model=torch.nn.Identity())
    result = {'selected_contour': square(), 'tracked_contour': None,
              'tracking_status': 'uninitialized_no_reliable_segmentation'}
    labels, lines = [], []
    monkeypatch.setattr(cv2, 'putText', lambda img, label, *args: labels.append(label) or img)
    monkeypatch.setattr(cv2, 'drawContours', lambda *args: lines.append('contour'))
    monkeypatch.setattr(cv2, 'polylines', lambda *args: lines.append('polyline'))
    predictor.create_overlay(textured_frame(), result)
    assert labels == ['WAITING FOR RELIABLE INITIALIZATION']
    assert not lines
    assert result['tracking_status'] == 'uninitialized_no_reliable_segmentation'
