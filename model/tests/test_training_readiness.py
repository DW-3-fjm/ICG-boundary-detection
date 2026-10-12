"""Regression cases for review findings that can invalidate training or evaluation."""
import sys
import csv
import importlib.util
import random
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from icg_pipeline.data import write_image, ICGDataset, get_dataloader
from icg_pipeline.models import create_model
from icg_pipeline.training import get_optimizer
from icg_pipeline.utils import load_config, get_device
from icg_pipeline.eval import Evaluator, compute_hausdorff_distance_95, _get_contour_points


def make_split(path, count=3):
    for part in ('images', 'masks'):
        (path / part).mkdir(parents=True)
    image = np.zeros((20, 22), np.uint8)
    image[3:10, 2:8] = 255
    for i in range(count):
        write_image(str(path / 'images' / f'{i}.png'), image)
        write_image(str(path / 'masks' / f'{i}.png'), image)
    return image


def test_repository_yaml_hyperparameters_are_numeric_and_optimizer_can_start():
    cfg = load_config(str(Path(__file__).resolve().parents[1] / 'configs/config.yaml'))
    assert isinstance(cfg['training']['learning_rate'], float)
    assert isinstance(cfg['training']['early_stopping']['min_delta'], float)
    optimizer = get_optimizer(torch.nn.Linear(1, 1), cfg)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-4)


@pytest.mark.parametrize('attention,bilinear', [(False,False),(False,True),(True,False),(True,True)])
def test_configured_unet_variants_forward_and_backward(attention, bilinear):
    model = create_model({'model': {'in_channels':3, 'out_channels':2, 'base_filters':4,
        'depth':2, 'use_attention':attention, 'bilinear':bilinear}})
    output = model(torch.randn(2, 3, 33, 37))
    assert output.shape == (2, 2, 33, 37)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_loader_only_needs_selected_split_and_keeps_final_batch(tmp_path):
    make_split(tmp_path / 'test')
    cfg = {'data': {'test_data_dir':str(tmp_path / 'test'), 'image_size':[20,22], 'batch_size':2}}
    loader = get_dataloader(cfg, 'test')
    assert [len(batch['mask']) for batch in loader] == [2,1]


def test_training_augmentations_can_be_disabled_and_enabled_explicitly(tmp_path):
    expected = make_split(tmp_path)
    dataset = ICGDataset(str(tmp_path), image_size=(20,22), is_train=True, augmentations={})
    for _ in range(8):
        np.testing.assert_array_equal(dataset[0]['mask'].numpy(), expected > 0)
    dataset.augmentations = {'horizontal_flip':1.0}
    np.testing.assert_array_equal(dataset[0]['mask'].numpy(), np.fliplr(expected > 0))
    dataset.is_train = False
    np.testing.assert_array_equal(dataset[0]['mask'].numpy(), expected > 0)


def test_loader_rejects_channel_mismatch_before_training(tmp_path):
    with pytest.raises(ValueError, match='must match'):
        get_dataloader({'data': {'in_channels':1}, 'model':{'in_channels':3}}, 'train')


def test_failed_boundaries_are_not_silently_removed_from_aggregates():
    mask = np.zeros((20,20), np.uint8)
    mask[4:14,4:14] = 1
    result = Evaluator({'evaluation':{'metrics':['dice','hausdorff_95']}}).evaluate_pairs(iter([
        ({'mask':mask},mask), ({'mask':np.zeros_like(mask)},mask)]))
    aggregate = result['aggregated']
    assert aggregate['mean_dice'] == pytest.approx(.5)
    assert np.isinf(aggregate['mean_hausdorff_95'])
    assert aggregate['nonfinite_count_hausdorff_95'] == 1
    assert aggregate['finite_mean_hausdorff_95'] == 0
    failed = Evaluator({'evaluation':{'metrics':['hausdorff_95']}}).evaluate_pairs([
        ({'mask':np.zeros_like(mask)},mask)])['aggregated']
    assert np.isinf(failed['mean_hausdorff_95'])
    assert np.isnan(failed['std_hausdorff_95'])


def test_tree_boundary_distance_matches_pairwise_reference():
    a = np.zeros((80,90), np.uint8)
    b = a.copy()
    a[10:45,20:60] = 1
    b[12:47,24:64] = 1
    distances = cdist(_get_contour_points(a), _get_contour_points(b))
    expected = max(np.percentile(distances.min(axis=0),95), np.percentile(distances.min(axis=1),95))
    assert compute_hausdorff_distance_95(a,b) == pytest.approx(expected)


def test_explicit_cuda_does_not_silently_fall_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA requested'):
        get_device({'use_cuda':True})
    assert get_device({'use_cuda':False}).type == 'cpu'


def test_train_without_test_split_and_resume_preserves_state(tmp_path):
    from icg_pipeline.training import Trainer
    for split in ('train','val'):
        make_split(tmp_path / split, count=2)
    cfg = load_config(str(Path(__file__).resolve().parents[1] / 'configs/config.yaml'))
    cfg['data'].update(train_data_dir=str(tmp_path/'train'), val_data_dir=str(tmp_path/'val'),
        test_data_dir=str(tmp_path/'absent_test'), image_size=[20,22], batch_size=2, num_workers=0, augmentations={})
    cfg['model'].update(base_filters=4, depth=2)
    cfg['training'].update(epochs=1, checkpoint_dir=str(tmp_path/'checkpoints'), save_top_k=1)
    cfg['logging'].update(log_dir=str(tmp_path/'logs'), use_tensorboard=False)
    cfg['device']['use_cuda'] = False
    trainer = Trainer(cfg)
    trainer.train()
    expected = random.random(), np.random.random(), torch.rand(1).item()
    resumed = Trainer(cfg)
    resumed.load_checkpoint(str(tmp_path/'checkpoints/last_model.pt'))
    actual = random.random(), np.random.random(), torch.rand(1).item()
    assert actual == expected
    assert resumed.current_epoch == 1
    assert resumed.best_val_dice == trainer.best_val_dice
    assert vars(resumed.early_stopping) == vars(trainer.early_stopping)
    assert resumed.best_metrics == trainer.best_metrics
    assert (tmp_path/'checkpoints/best_model.pt').is_file()
    best = Trainer(cfg)
    best.load_checkpoint(str(tmp_path/'checkpoints/best_model.pt'))
    assert best.best_metrics == trainer.best_metrics


def test_data_preflight_requires_groups_to_claim_split_verification(tmp_path):
    spec = importlib.util.spec_from_file_location('check_data', Path(__file__).resolve().parents[1]/'scripts/check_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for index, split in enumerate(('train','val','test')):
        make_split(tmp_path/split, count=1)
        image = np.zeros((20,22), np.uint8)
        image[1:3+index,1:5] = 100
        write_image(str(tmp_path/split/'images/0.png'), image)
    cfg = {'data':{f'{s}_data_dir':str(tmp_path/s) for s in ('train','val','test')}}
    assert module.check_data(cfg)['ok']
    assert not module.check_data(cfg)['patient_split_verified']
    manifest = tmp_path/'groups.csv'
    with manifest.open('w',newline='',encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['split','filename','group_id'])
        writer.writerows([(s,'0.png',s) for s in ('train','val','test')])
    assert module.check_data(cfg,manifest)['patient_split_verified']
    manifest.write_text('split,filename,group_id\ntrain,0.png,same\nval,0.png,same\ntest,0.png,third\n',encoding='utf-8')
    report = module.check_data(cfg,manifest)
    assert not report['ok']
    assert not report['patient_split_verified']


def test_data_preflight_detects_duplicate_images_across_splits(tmp_path):
    spec = importlib.util.spec_from_file_location('check_data', Path(__file__).resolve().parents[1]/'scripts/check_data.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for split in ('train','val','test'):
        make_split(tmp_path/split,count=1)
    cfg = {'data':{f'{s}_data_dir':str(tmp_path/s) for s in ('train','val','test')}}
    report = module.check_data(cfg)
    assert not report['ok']
    assert any('Exact duplicate' in error for error in report['errors'])


@pytest.mark.parametrize('kind', ['color', 'uint16', 'multiclass'])
def test_training_rejects_unsupported_masks_instead_of_silently_merging(tmp_path, kind):
    make_split(tmp_path, count=1)
    if kind == 'color':
        invalid = np.full((20, 22, 3), 255, np.uint8)
    elif kind == 'uint16':
        invalid = np.full((20, 22), 65535, np.uint16)
    else:
        invalid = np.zeros((20, 22), np.uint8)
        invalid[0] = 1
        invalid[1] = 2
    write_image(str(tmp_path/'masks/0.png'), invalid)
    with pytest.raises(ValueError, match='mask|Mask'):
        ICGDataset(tmp_path)[0]


def test_evaluation_rejects_broadcastable_shape_mismatch_and_misspelled_metric():
    from icg_pipeline.eval import evaluate_segmentation
    with pytest.raises(ValueError, match='equal shape'):
        evaluate_segmentation(np.ones((4, 1)), np.ones((4, 4)))
    with pytest.raises(ValueError, match='Unknown'):
        evaluate_segmentation(np.ones((4, 4)), np.ones((4, 4)), ['dcie'])


def test_dataset_rejects_ambiguous_mask_files(tmp_path):
    make_split(tmp_path, count=1)
    write_image(str(tmp_path/'masks/0.bmp'), np.zeros((20,22), np.uint8))
    with pytest.raises(ValueError, match='Ambiguous masks'):
        ICGDataset(tmp_path)


def test_overfit_mode_needs_only_training_data_and_does_not_augment_fit_loader(tmp_path):
    from icg_pipeline.training import Trainer
    make_split(tmp_path/'train', count=2)
    cfg = load_config(str(Path(__file__).resolve().parents[1]/'configs/config.yaml'))
    cfg['data'].update(train_data_dir=str(tmp_path/'train'), val_data_dir=str(tmp_path/'absent'),
                       image_size=[20,22], batch_size=2, num_workers=0, augmentations={'horizontal_flip':1})
    cfg['model'].update(base_filters=4, depth=2)
    cfg['training'].update(epochs=1, overfit_only=True, checkpoint_dir=str(tmp_path/'checkpoints'))
    cfg['logging'].update(log_dir=str(tmp_path/'logs'), use_tensorboard=False)
    cfg['device']['use_cuda'] = False
    trainer = Trainer(cfg)
    assert trainer.train_loader.dataset.is_train
    assert not trainer.val_loader.dataset.is_train
    assert trainer.val_loader.dataset.augmentations is None
    trainer.train()
    checkpoint = torch.load(tmp_path/'checkpoints/last_model.pt', weights_only=True)
    assert checkpoint['training_mode'] == 'overfit_diagnostic'
    cfg['training']['overfit_only'] = False
    cfg['data']['val_data_dir'] = str(tmp_path/'train')
    regular = Trainer(cfg)
    with pytest.raises(ValueError, match='training mode differs'):
        regular.load_checkpoint(str(tmp_path/'checkpoints/last_model.pt'))


def test_dice_loss_large_half_precision_masks_do_not_overflow_pixel_sums():
    from icg_pipeline.training import DiceLoss
    targets = torch.zeros((1,288,512), dtype=torch.long)
    targets[:,20:80,20:80] = 1
    logits = torch.full((1,2,288,512), -8.0, dtype=torch.float16)
    logits[:,0] = torch.where(targets == 0, 8., -8.)
    logits[:,1] = -logits[:,0]
    logits.requires_grad_()
    loss = DiceLoss()(logits, targets)
    assert torch.isfinite(loss) and loss < 1e-3
    loss.backward()
    assert torch.isfinite(logits.grad).all()
