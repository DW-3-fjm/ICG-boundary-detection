"""Synthetic checks are software evidence, not fluorescence-fading validation."""
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def test_gated_tracker_rejects_conflicting_confident_correction_and_latches_failure():
    import icg_pipeline.tracking as tracking
    assert hasattr(tracking, 'GatedContourTracker'), 'gated policy is missing'
    rng = np.random.default_rng(42)
    frame = rng.integers(20, 230, (128, 128, 3), dtype=np.uint8)
    contour = np.array([[25, 25], [75, 25], [75, 75], [25, 75]], np.float32)
    tracker = tracking.GatedContourTracker(working_size=(128, 128), num_points=40)
    first = tracker.update(frame, contour, .99)
    assert first['tracking_status'] == 'segmentation_initialized'
    second = cv2.warpAffine(frame, np.float32([[1, 0, 3], [0, 1, 2]]), (128, 128))
    conflicting = contour + [35, 35]
    result = tracker.update(second, conflicting, .99)
    assert result['tracking_status'] == 'propagated'
    assert result['correction_status'] == 'rejected'
    assert result['temporal_iou'] < .5
    assert np.allclose(np.median(result['tracked_contour'], axis=0), [53, 52], atol=2)
    corrected = tracker.update(second, contour + [3, 2], .99)
    assert corrected['tracking_status'] == 'segmentation_corrected'
    failed = tracker.update(np.zeros_like(frame), contour, .99)
    assert failed['tracked_contour'] is None
    assert failed['tracking_status'].startswith('tracking_failed')
    # A later confident segmentation must not silently recover a lost identity.
    assert tracker.update(frame, contour, .99)['tracked_contour'] is None
    tracker.reset()
    assert tracker.update(frame, contour, .99)['tracked_contour'] is not None


def test_gated_tracker_requires_confidence_and_maps_back_to_source_coordinates():
    from icg_pipeline.tracking import GatedContourTracker
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 255, (192, 256, 3), dtype=np.uint8)
    contour = np.array([[40, 40], [160, 40], [160, 140], [40, 140]], np.float32)
    tracker = GatedContourTracker(working_size=(96, 128))
    assert tracker.update(frame, contour, .6)['tracked_contour'] is None
    initial = tracker.update(frame, contour, .9)
    assert np.allclose(initial['tracked_contour'].min(0), [40, 40])
    moved = cv2.warpAffine(frame, np.float32([[1, 0, 4], [0, 1, 2]]), (256, 192))
    result = tracker.update(moved, None, None)
    assert result['tracking_status'] == 'propagated'
    assert np.allclose(result['tracked_contour'].min(0), [44, 42], atol=2)


def test_synthetic_sequences_are_repeatable_with_source_split_and_warped_masks():
    import icg_pipeline.synthetic as synthetic
    image = np.random.default_rng(4).integers(0, 255, (96, 128, 3), dtype=np.uint8)
    mask = np.zeros((96, 128), np.uint8)
    mask[20:60, 30:70] = 255
    for scenario in ('translation', 'rotation', 'deformation', 'brightness', 'occlusion'):
        a = synthetic.generate_sequence(image, mask, scenario=scenario, split='val', seed=42, frames=4)
        b = synthetic.generate_sequence(image, mask, scenario=scenario, split='val', seed=42, frames=4)
        assert len(a) == 4
        assert all(np.array_equal(x['image'], y['image']) for x, y in zip(a, b))
        assert all(x['source_split'] == 'val' for x in a)
        assert all(set(np.unique(x['mask'])).issubset({0, 255}) for x in a)
        if scenario == 'translation':
            ys, xs = np.where(a[-1]['mask'] > 0)
            assert (xs.min(), ys.min()) == (36, 23)
        if scenario == 'occlusion':
            assert np.array_equal(a[0]['mask'], a[1]['mask'])


def test_tracking_success_counts_annotated_failures_in_denominator():
    from icg_pipeline.eval import Evaluator
    contour = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], np.float32)
    result = Evaluator({}).evaluate_tracking_sequence(
        [{'tracked_contour': contour, 'tracking_status': 'segmentation_initialized'},
         {'tracked_contour': None, 'tracking_status': 'tracking_failed:quality'},
         {'tracked_contour': None, 'tracking_status': 'tracking_failed:latched_until_reset'}],
        [contour, contour, None])
    assert result['tracking_success_rate'] == .5
    assert result['annotated_frame_count'] == 2
    assert result['evaluated_frame_count'] == 1
    assert result['distance_metrics_condition'] == 'successful_annotated_frames_only'
