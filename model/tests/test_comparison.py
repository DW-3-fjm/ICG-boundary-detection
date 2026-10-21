"""Behavioral tests for the controlled ICG experiments (no clinical data)."""
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from icg_pipeline import training
from icg_pipeline.data import ICGDataset, preprocess_image


def paired_data(root, count=2, size=(40, 56)):
    for part in ('images', 'masks'):
        (root / part).mkdir(parents=True)
    for i in range(count):
        image = np.zeros((*size, 3), np.uint8)
        image[8:25, 5:19] = (30, 100, 220)
        mask = np.zeros(size, np.uint8)
        mask[8:25, 5:19] = 255
        cv2.imwrite(str(root / 'images' / f'{i}.png'), image)
        cv2.imwrite(str(root / 'masks' / f'{i}.png'), mask)


def smoke_config(tmp_path, architecture='unet'):
    return {
        'data': {'train_data_dir': str(tmp_path / 'train'), 'val_data_dir': str(tmp_path / 'val'),
                 'test_data_dir': str(tmp_path / 'never_open_test'), 'image_size': [64, 64],
                 'in_channels': 3, 'batch_size': 2, 'num_workers': 0,
                 'normalization': 'imagenet', 'augmentations': {}},
        'model': {'name': architecture, 'in_channels': 3, 'out_channels': 2,
                  'base_filters': 4, 'depth': 2, 'dropout': 0,
                  'encoder_weights': None, 'freeze_encoder_batchnorm': True},
        'training': {'epochs': 1, 'seed': 42, 'learning_rate': 3e-4, 'encoder_learning_rate': 3e-5,
                     'weight_decay': 1e-5, 'optimizer': 'adamw', 'scheduler': 'cosine_annealing',
                     'scheduler_params': {'T_max': 1, 'eta_min': 1e-6},
                     'loss_function': 'tversky_ce_boundary', 'loss_params': {},
                     'mixed_precision': False, 'early_stopping': {'enabled': False},
                     'checkpoint_dir': str(tmp_path / 'weights'), 'save_top_k': 0},
        'inference': {'model_path': str(tmp_path / 'weights' / 'best_model.pt'),
                      'output_dir': str(tmp_path / 'predictions'), 'min_contour_area': 4},
        'device': {'use_cuda': False},
        'logging': {'log_dir': str(tmp_path / 'logs'), 'level': 'ERROR', 'use_tensorboard': False},
        'evaluation': {'metrics': ['dice', 'iou', 'precision', 'recall', 'hausdorff_95',
                                   'surface_distance', 'area_ratio']},
    }


def test_tversky_weights_false_positives_more_than_false_negatives():
    assert hasattr(training, 'TverskyLoss'), 'Tversky loss is missing'
    criterion = training.TverskyLoss(fp_weight=0.7, fn_weight=0.3)
    target = torch.tensor([[[1, 1, 0, 0]]])
    def logits(prob):
        p = torch.tensor(prob).reshape(1, 1, 4)
        return torch.stack(((1-p).log(), p.log()), dim=1)
    # Equal TP and one matching FP/FN against separate targets:
    # approximately TP=1, FP=1 versus TP=1, FN=1.
    fp = criterion(logits([.99, .99, .01, .01]), torch.tensor([[[1, 0, 0, 0]]]))
    fn = criterion(logits([.99, .01, .01, .01]), torch.tensor([[[1, 1, 0, 0]]]))
    assert fp > fn


@pytest.mark.parametrize('name', ['dice_ce', 'tversky_ce', 'tversky_ce_boundary'])
@pytest.mark.parametrize('value', [0, 1])
def test_losses_are_finite_on_large_half_precision_empty_and_full_targets(name, value):
    criterion = training.get_loss_function({'training': {'loss_function': name, 'loss_params': {}}})
    logits = torch.zeros(1, 2, 288, 512, dtype=torch.float16, requires_grad=True)
    targets = torch.full((1, 288, 512), value, dtype=torch.long)
    loss = criterion(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_boundary_ce_detects_displacement_and_ignores_degenerate_targets():
    assert hasattr(training, 'BoundaryCELoss'), 'boundary CE is missing'
    criterion = training.BoundaryCELoss()
    target = torch.zeros(1, 32, 32, dtype=torch.long)
    target[:, 8:24, 8:24] = 1
    correct = F.one_hot(target, 2).permute(0, 3, 1, 2).float() * 8
    shifted = torch.roll(correct, 4, dims=3)
    assert criterion(shifted, target) > criterion(correct, target)
    for value in (0, 1):
        logits = torch.randn(1, 2, 32, 32, requires_grad=True)
        loss = criterion(logits, torch.full_like(target, value))
        assert loss.item() == 0
        loss.backward()
        assert logits.grad is not None and not logits.grad.any()


def test_combined_loss_penalizes_distant_false_positive_island():
    loss = training.get_loss_function({'training': {'loss_function': 'tversky_ce_boundary'}})
    target = torch.zeros(1, 40, 40, dtype=torch.long)
    target[:, 5:15, 5:15] = 1
    correct = F.one_hot(target, 2).permute(0, 3, 1, 2).float() * 8
    island = correct.clone()
    island[:, 0, 30:36, 30:36] = 0
    island[:, 1, 30:36, 30:36] = 8
    assert loss(island, target) > loss(correct, target)


def test_imagenet_preprocessing_matches_dataset_and_configured_flip(tmp_path):
    paired_data(tmp_path, count=1)
    bgr = cv2.imread(str(tmp_path / 'images' / '0.png'))
    ds = ICGDataset(str(tmp_path), image_size=(40, 56), channels=3, is_train=False,
                    normalization='imagenet', augmentations={'horizontal_flip': 1})
    assert torch.equal(ds[0]['image'], ds[0]['image'])
    assert torch.equal(ds[0]['image'], preprocess_image(bgr, (40, 56), 3, 'imagenet')[0])
    train = ICGDataset(str(tmp_path), image_size=(40, 56), channels=3, is_train=True,
                       normalization='imagenet', augmentations={'horizontal_flip': 1})
    assert torch.equal(train[0]['mask'], ds[0]['mask'].flip(1))
    assert torch.equal(train[0]['image'], ds[0]['image'].flip(2))


@pytest.mark.parametrize('architecture', ['unet', 'unet_resnet18'])
def test_cpu_train_checkpoint_restore_original_resolution_eval(tmp_path, architecture):
    from icg_pipeline.models import create_model
    from icg_pipeline.inference import ICGPredictor
    from icg_pipeline.eval import evaluate_source_split
    paired_data(tmp_path / 'train')
    paired_data(tmp_path / 'val', count=1)
    cfg = smoke_config(tmp_path, architecture)
    trainer = training.Trainer(cfg)
    if architecture == 'unet_resnet18':
        bn = trainer.model.encoder.bn1
        before = bn.running_mean.clone()
        trainer.model.train()
        assert not bn.training
        assert trainer.optimizer.param_groups[0]['lr'] == 3e-5
    trainer.train()
    if architecture == 'unet_resnet18':
        assert torch.equal(before, bn.running_mean)
    saved = torch.load(cfg['inference']['model_path'], weights_only=True)
    assert saved['optimizer_updates'] == 1
    assert saved['metadata']['preprocessing']['normalization'] == 'imagenet'
    # Missing model/data metadata in the caller is restored from the checkpoint.
    inference_cfg = copy.deepcopy(cfg)
    inference_cfg.pop('model')
    inference_cfg.pop('data')
    predictor = ICGPredictor(inference_cfg)
    assert predictor.model(torch.randn(1, 3, 65, 71)).shape == (1, 2, 65, 71)
    result = evaluate_source_split(cfg, 'val', predictor=predictor)
    assert result['sample_count'] == 1
    assert result['per_sample'][0]['image_size_hw'] == [40, 56]
    assert result['distance_units'] == 'original_image_pixels'
    incompatible = copy.deepcopy(cfg)
    incompatible['data']['normalization'] = 'legacy'
    with pytest.raises(ValueError, match='normalization'):
        ICGPredictor(incompatible)


def test_evaluation_does_not_hide_nonfinite_failures():
    from icg_pipeline.eval import Evaluator
    empty = np.zeros((32, 32), np.uint8)
    target = empty.copy()
    target[8:24, 8:24] = 255
    result = Evaluator({'evaluation': {'metrics': ['dice', 'hausdorff_95']}}).evaluate_dataset(
        [{'mask': target}, {'mask': empty}], [target, target])
    assert np.isinf(result['aggregated']['mean_hausdorff_95'])
    assert result['aggregated']['nonfinite_count_hausdorff_95'] == 1


def test_pretrained_encoder_is_not_reinitialized_and_inference_never_downloads(monkeypatch):
    from torchvision import models
    from icg_pipeline.models import create_model
    real_builder = models.resnet18
    calls = []
    def cached_encoder(*, weights):
        calls.append(weights)
        encoder = real_builder(weights=None)
        with torch.no_grad():
            encoder.conv1.weight.fill_(.123)
        return encoder
    monkeypatch.setattr(models, 'resnet18', cached_encoder)
    cfg = {'model': {'name': 'unet_resnet18', 'in_channels': 3, 'out_channels': 2,
                     'encoder_weights': 'IMAGENET1K_V1'}}
    model = create_model(cfg)
    assert calls == [models.ResNet18_Weights.IMAGENET1K_V1]
    assert torch.all(model.encoder.conv1.weight == .123)
    calls.clear()
    create_model(cfg, load_pretrained=False)
    assert calls == [None]
    def unavailable(*, weights):
        raise OSError('offline fixture')
    monkeypatch.setattr(models, 'resnet18', unavailable)
    with pytest.raises(RuntimeError, match='random fallback is disabled'):
        create_model(cfg)


def test_six_configs_resolve_and_runner_preserves_test_isolation(tmp_path):
    from icg_pipeline.comparison import run_validation, run_frozen_test
    from icg_pipeline.utils import load_config
    config_dir = Path(__file__).resolve().parents[1] / 'configs'
    configs = [config_dir / f'compare_{arch}_{loss}.yaml'
               for arch in ('unet', 'unet_resnet18') for loss in ('control', 'tversky', 'combined')]
    for path in configs:
        cfg = load_config(str(path))
        criterion = training.get_loss_function(cfg)
        assert torch.isfinite(criterion(torch.zeros(1, 2, 32, 32), torch.zeros(1, 32, 32, dtype=torch.long)))
        assert cfg['data']['normalization'] == 'imagenet'
        assert cfg['training']['early_stopping']['enabled'] is False
        assert cfg['tracking']['correction_policy'] == 'gated'
    root = tmp_path / 'data'
    paired_data(root / 'train')
    paired_data(root / 'val', count=1)
    for part in ('images', 'masks'):
        (root / 'val' / part / '0.png').rename(root / 'val' / part / 'val.png')
    manifest = {'records': [
        {'filename': '0.png', 'split': 'train', 'source_batch': 'a'},
        {'filename': '1.png', 'split': 'train', 'source_batch': 'a'},
        {'filename': 'val.png', 'split': 'val', 'source_batch': 'b'},
        {'filename': 'test.png', 'split': 'test', 'source_batch': 'c'}]}
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf8')
    # No test directory exists while training/validation run.
    config_path = tmp_path / 'smoke.yaml'
    import yaml
    config_path.write_text(yaml.safe_dump(smoke_config(tmp_path)), encoding='utf8')
    output = tmp_path / 'runs'
    summary = run_validation([config_path], [42], root, manifest_path, output)
    assert len(summary['runs']) == 1
    assert (output / 'frozen_experiments.json').is_file()
    run_dir = Path(summary['runs'][0]['run_dir'])
    assert (run_dir / 'val' / 'overlays' / 'val.png').is_file()
    original_mask = (root / 'val/masks/val.png').read_bytes()
    (root / 'val/masks/val.png').write_bytes(original_mask + b'changed annotation')
    with pytest.raises(ValueError, match='Development data changed'):
        run_frozen_test(output)
    assert not (output / 'test_evaluation_started.json').exists()
    (root / 'val/masks/val.png').write_bytes(original_mask)
    paired_data(root / 'test', count=1)
    for part in ('images', 'masks'):
        (root / 'test' / part / '0.png').rename(root / 'test' / part / 'test.png')
    tested = run_frozen_test(output)
    assert tested['runs'][0]['metrics']['mean_dice'] >= 0
    with pytest.raises(FileExistsError):
        run_frozen_test(output)


def test_promising_rule_checks_recall_and_dice_instead_of_rewarding_shrinkage():
    from icg_pipeline.comparison import summarize_runs
    def row(loss, dice, recall, hd95):
        return {'architecture': 'unet', 'variant': loss, 'seed': 42,
                'metrics': {'mean_dice': dice, 'mean_recall': recall,
                            'mean_hausdorff_95': hd95, 'mean_precision': .8, 'mean_area_ratio': .9}}
    result = summarize_runs([row('control', .8, .8, 50), row('combined', .79, .7, 30)])
    assert result['unet/combined']['promising'] is False


def test_comparison_rechecks_development_data_after_each_training_run(tmp_path, monkeypatch):
    from icg_pipeline import comparison
    import yaml
    root = tmp_path / 'data'
    paired_data(root / 'train')
    paired_data(root / 'val', count=1)
    for part in ('images', 'masks'):
        (root / 'val' / part / '0.png').rename(root / 'val' / part / 'val.png')
    records = [{'filename': name, 'split': split, 'source_batch': split}
               for split, names in [('train', ['0.png', '1.png']), ('val', ['val.png']), ('test', ['test.png'])]
               for name in names]
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'records': records}), encoding='utf8')
    config_path = tmp_path / 'smoke.yaml'
    config_path.write_text(yaml.safe_dump(smoke_config(tmp_path)), encoding='utf8')
    original_train = comparison.Trainer.train
    def mutate_after_training(trainer):
        result = original_train(trainer)
        mask = root / 'val/masks/val.png'
        mask.write_bytes(mask.read_bytes() + b'mutated while training')
        return result
    monkeypatch.setattr(comparison.Trainer, 'train', mutate_after_training)
    with pytest.raises(ValueError, match='Development data changed'):
        comparison.run_validation([config_path], [42], root, manifest, tmp_path / 'out')
    assert not (tmp_path / 'out/validation_summary.json').exists()
