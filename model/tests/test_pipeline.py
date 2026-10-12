"""Regression tests for ICG segmentation preprocessing and tracking."""

import sys
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from icg_pipeline import ICGDataset, OpticalFlowTracker, UNet, contour_to_points, extract_contours
from icg_pipeline.data import write_image, postprocess_mask, preprocess_image
from icg_pipeline.eval import Evaluator
from icg_pipeline.inference import ICGPredictor


class ConstantSegmentation(torch.nn.Module):
    def forward(self, image):
        logits = torch.zeros((image.shape[0], 2, image.shape[2], image.shape[3]), device=image.device)
        logits[:, 0] = 2.0
        logits[:, 1, 8:24, 10:30] = 3.0
        return logits


@pytest.mark.parametrize("original_size", [(32, 32), (64, 64), (71, 113), (20, 24)])
@pytest.mark.parametrize("threshold, accepted", [(0.7, True), (0.75, False)])
def test_contour_gate_scores_native_mask_pixels_across_output_sizes(tmp_path, original_size, threshold, accepted):
    config = {
        "device": {"use_cuda": False},
        "model": {"in_channels": 1, "out_channels": 2},
        "data": {"image_size": [32, 32], "in_channels": 1},
        "inference": {"min_contour_area": 10, "min_mean_probability": threshold,
                      "output_dir": str(tmp_path / "out")},
    }
    predictor = ICGPredictor(config, model=ConstantSegmentation())
    result = predictor.predict_image(np.zeros(original_size, dtype=np.uint8))
    assert (result["selected_contour"] is not None) == accepted
    assert result["mask"].shape == original_size
    if accepted:
        # Every native foreground pixel has logits [2, 3], regardless of output size.
        assert result["segmentation_probability"] == pytest.approx(1 / (1 + np.exp(-1)), abs=1e-6)


def test_unet_forward_preserves_spatial_size_for_odd_inputs():
    model = UNet(in_channels=1, out_channels=2, base_filters=8, depth=3)
    assert model(torch.randn(1, 1, 65, 71)).shape == (1, 2, 65, 71)


def test_preprocess_preserves_rgb_channels_and_coordinate_size():
    bgr = np.zeros((20, 30, 3), dtype=np.uint8)
    bgr[..., 2] = 255
    tensor = preprocess_image(bgr, (10, 15), channels=3)
    assert tensor.shape == (1, 3, 10, 15)
    assert tensor[0, 0].mean() > tensor[0, 2].mean()


def test_class_logits_are_decoded_with_argmax_not_sigmoid():
    logits = torch.empty((1, 2, 4, 5))
    logits[:, 0] = 1.0
    logits[:, 1] = 0.9
    assert not postprocess_mask(logits, (8, 10)).any()
    logits[:, 1, 1:3, 2:4] = 2
    decoded = postprocess_mask(logits, (8, 10))
    assert decoded.shape == (8, 10)
    assert set(np.unique(decoded)) == {0, 255}


def test_dataset_pairs_by_stem_and_binarizes_annotation(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    image = np.full((12, 14), 128, np.uint8)
    mask = np.zeros((12, 14), np.uint8)
    mask[3:8, 4:10] = 127
    write_image(str(tmp_path / "images" / "case.png"), image)
    write_image(str(tmp_path / "masks" / "case.png"), mask)
    item = ICGDataset(str(tmp_path), image_size=(16, 18), is_train=False)[0]
    assert item["image"].shape == (1, 16, 18)
    assert item["mask"].dtype == torch.long
    assert set(torch.unique(item["mask"]).tolist()) == {0, 1}
    assert item["image_path"].endswith("case.png")


def test_dataset_rejects_missing_masks_instead_of_making_empty_targets(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    write_image(str(tmp_path / "images" / "case.png"), np.zeros((8, 8), np.uint8))
    with pytest.raises(ValueError, match="Missing masks"):
        ICGDataset(str(tmp_path))


def test_contour_extraction_filters_and_orders_components():
    mask = np.zeros((120, 120), dtype=np.uint8)
    cv2.rectangle(mask, (5, 5), (20, 20), 255, -1)
    cv2.rectangle(mask, (40, 40), (100, 100), 255, -1)
    contours = extract_contours(mask, min_area=100)
    assert len(contours) == 2
    assert cv2.contourArea(contours[0]) > cv2.contourArea(contours[1])


def test_contour_resampling_is_finite_and_preserves_xy_convention():
    contour = np.array([[[10, 20]], [[30, 20]], [[30, 40]], [[10, 40]]], np.int32)
    points = contour_to_points(contour, 80)
    assert points.shape == (80, 2)
    assert points.dtype == np.float32
    assert np.isfinite(points).all()
    assert 10 <= points[:, 0].min() <= points[:, 0].max() <= 30
    assert 20 <= points[:, 1].min() <= points[:, 1].max() <= 40


def test_optical_flow_initializes_and_propagates_textured_boundary():
    rng = np.random.default_rng(4)
    first = rng.integers(0, 256, (128, 128), dtype=np.uint8)
    second = cv2.warpAffine(first, np.float32([[1, 0, 3], [0, 1, 2]]), (128, 128))
    contour = np.array([[25, 25], [90, 25], [90, 90], [25, 90]], np.float32)
    tracker = OpticalFlowTracker(num_points=40)
    tracker.initialize(first, contour)
    tracked = tracker.track(second)
    assert tracked.status == "propagated"
    assert tracked.points is not None
    assert tracked.valid_point_fraction >= 0.6
    assert np.allclose(np.median(tracked.points - contour_to_points(contour, 40), axis=0), [3, 2], atol=1.0)


def test_optical_flow_reports_uninitialized_and_invalid_contours():
    tracker = OpticalFlowTracker(num_points=20)
    result = tracker.track(np.zeros((32, 32), np.uint8))
    assert result.status == "uninitialized"
    assert result.points is None
    with pytest.raises(ValueError, match="inside the frame"):
        tracker.initialize(np.zeros((32, 32), np.uint8), np.array([[-5, 3], [10, 3], [10, 10], [-5, 10]]))


def test_optical_flow_flags_unobservable_flat_frame():
    rng = np.random.default_rng(8)
    first = rng.integers(20, 240, (96, 96), dtype=np.uint8)
    contour = np.array([[20, 20], [75, 20], [75, 75], [20, 75]], np.float32)
    tracker = OpticalFlowTracker(num_points=32)
    tracker.initialize(first, contour)
    result = tracker.track(np.zeros_like(first))
    assert result.status == "failed_quality_check"
    assert result.points is None


def test_video_reinitializes_on_segmentation_and_logs_propagation_failure(tmp_path, monkeypatch):
    video_path = tmp_path / "sequence.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 96))
    if not writer.isOpened():
        pytest.skip("OpenCV video writer codec unavailable")
    rng = np.random.default_rng(12)
    first = rng.integers(20, 240, (96, 96), dtype=np.uint8)
    first = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
    second = cv2.warpAffine(first, np.float32([[1, 0, 2], [0, 1, 1]]), (96, 96))
    writer.write(first)
    writer.write(second)
    writer.write(np.zeros_like(first))
    writer.release()

    config = {
        "device": {"use_cuda": False},
        "model": {"in_channels": 1, "out_channels": 2},
        "data": {"image_size": [32, 32], "in_channels": 1},
        "inference": {"min_contour_area": 10, "min_mean_probability": 0.5, "output_dir": str(tmp_path / "out")},
        "tracking": {"enabled": True, "num_points": 32},
    }
    predictor = ICGPredictor(config, model=ConstantSegmentation())
    contour = np.array([[[20, 20]], [[75, 20]], [[75, 75]], [[20, 75]]], np.int32)
    call_count = 0

    def mock_prediction(_frame):
        nonlocal call_count
        call_count += 1
        selected = contour if call_count == 1 else None
        return {
            "mask": np.zeros((96, 96), np.uint8),
            "probability_map": np.zeros((96, 96), np.float32),
            "contours": [],
            "contour_points": [],
            "selected_contour": selected,
            "original_size": (96, 96),
        }

    monkeypatch.setattr(predictor, "predict_image", mock_prediction)
    output = tmp_path / "result.mp4"
    result = predictor.predict_video(str(video_path), str(output))
    statuses = [row["tracking_status"] for row in result["frames"]]
    assert statuses[0] == "segmentation_supported"
    assert statuses[1] == "propagated"
    assert statuses[2].startswith("tracking_failed:")
    assert result["frames"][2]["tracked_contour"] is None
    assert 'mask' not in result['frames'][0]
    assert 'probability_map' not in result['frames'][0]
    assert result['frames'][0]['end_to_end_time_ms'] >= result['frames'][0]['processing_time_ms']
    assert output.is_file()
    saved = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert len(saved["frames"]) == 3
    assert saved["frames"][1]["processing_time_ms"] >= 0


def test_tracking_evaluation_uses_external_reference_and_elapsed_time():
    contour = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], np.float32)
    shifted = contour + np.array([2, 0], np.float32)
    evaluator = Evaluator({})
    result = evaluator.evaluate_tracking_sequence(
        [
            {"tracked_contour": contour, "tracking_status": "segmentation_supported", "processing_time_ms": 10},
            {"tracked_contour": shifted, "tracking_status": "propagated", "processing_time_ms": 20},
            {"tracked_contour": None, "tracking_status": "tracking_failed:quality"},
        ],
        [contour, contour, None],
        [0.0, 1.0, 2.0],
    )
    assert result["mean_contour_distance"] == pytest.approx(1.0, abs=0.5)
    assert result["tracking_failure_frequency"] == pytest.approx(1 / 3)
    assert result["drift_by_elapsed_seconds"]["1.0"] > 0
    assert result["frames_per_second"] == pytest.approx(1000 / 15)


def test_predict_image_returns_full_resolution_mask_and_contours():
    config = {
        "device": {"use_cuda": False},
        "model": {"in_channels": 1, "out_channels": 2},
        "data": {"image_size": [32, 32], "in_channels": 1},
        "inference": {"min_contour_area": 10, "min_mean_probability": 0.5, "output_dir": "."},
        "tracking": {"enabled": True},
    }
    predictor = ICGPredictor(config, model=ConstantSegmentation())
    prediction = predictor.predict_image(np.zeros((64, 80), np.uint8))
    assert prediction["mask"].shape == (64, 80)
    assert prediction["selected_contour"] is not None
    assert prediction["selected_contour"].shape == (100, 2)
    assert prediction["mask"].max() == 255


def test_one_epoch_training_checkpoint_and_single_image_inference(tmp_path):
    from icg_pipeline import Trainer, run_inference

    for split, count in (("train", 2), ("val", 1), ("test", 1)):
        for subdir in ("images", "masks"):
            (tmp_path / "data" / split / subdir).mkdir(parents=True)
        for index in range(count):
            image = np.zeros((32, 32), np.uint8)
            image[8:24, 8:24] = 200
            mask = np.zeros_like(image)
            mask[10:22, 10:22] = 255
            name = f"sample_{index}.png"
            write_image(str(tmp_path / "data" / split / "images" / name), image)
            write_image(str(tmp_path / "data" / split / "masks" / name), mask)

    config = {
        "data": {
            "train_data_dir": str(tmp_path / "data" / "train"),
            "val_data_dir": str(tmp_path / "data" / "val"),
            "test_data_dir": str(tmp_path / "data" / "test"),
            "image_size": [32, 32],
            "in_channels": 1,
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
        },
        "model": {
            "in_channels": 1, "out_channels": 2, "base_filters": 4, "depth": 2,
            "use_batchnorm": True, "dropout": 0.0, "bilinear": False,
        },
        "training": {
            "epochs": 1, "learning_rate": 1e-3, "weight_decay": 0,
            "optimizer": "adamw", "scheduler": "cosine_annealing",
            "scheduler_params": {"T_max": 1, "eta_min": 1e-6},
            "loss_function": "dice_ce", "loss_params": {"dice_weight": 0.5, "ce_weight": 0.5},
            "gradient_clip": 1.0, "mixed_precision": False,
            "early_stopping": {"patience": 2, "min_delta": 1e-4, "mode": "max"},
            "checkpoint_dir": str(tmp_path / "checkpoints"), "save_top_k": 1,
        },
        "inference": {
            "model_path": str(tmp_path / "checkpoints" / "best_model.pt"),
            "output_dir": str(tmp_path / "predictions"),
            "save_overlay": True, "overlay_alpha": 0.5,
            "min_contour_area": 4, "min_mean_probability": 0.0,
        },
        "tracking": {"enabled": True, "num_points": 32},
        "device": {"use_cuda": False},
        "logging": {"log_dir": str(tmp_path / "logs"), "level": "ERROR", "use_tensorboard": False},
        "evaluation": {"metrics": ["dice", "iou"]},
    }
    result = Trainer(config).train()
    checkpoint = tmp_path / "checkpoints" / "best_model.pt"
    assert checkpoint.is_file()
    assert result["best_val_dice"] >= 0
    inference = run_inference(
        "", str(tmp_path / "data" / "test" / "images" / "sample_0.png"),
        config_override=config,
    )
    assert Path(inference["mask_path"]).is_file()
    assert Path(inference["overlay_path"]).is_file()
    assert inference["mask"].shape == (32, 32)
