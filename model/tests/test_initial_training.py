"""Synthetic-only checks for split identity, fresh initialization and source-resolution evaluation."""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from icg_pipeline.data import write_image
from icg_pipeline.experiment import read_split_manifest, convert_initial_labelme, prepare_initial_dataset, train_initial, evaluate_initial
from icg_pipeline.utils import load_config


def fixture(tmp_path, duplicate=False):
    source = tmp_path / 'converted'
    mapping, rows = [], []
    for i, split in enumerate(['train', 'val', 'test']):
        name = f'sample_{i+1:05d}.png'
        image = np.full((64, 96, 3), 40 + (0 if duplicate else i) * 40, dtype=np.uint8)
        mask = np.zeros((64, 96), dtype=np.uint8)
        cv2.rectangle(mask, (18, 15), (60, 50), 255, -1)
        image[mask > 0] = [90, 170, 110]
        write_image(str(source / 'train/images' / name), image)
        write_image(str(source / 'train/masks' / name), mask)
        mapping.append(dict(filename=name, labels=['Z'], split='train', annotation=f'folder_{i}/1.json'))
        rows.append(dict(filename=name, phase='Z', source_batch=f'batch_{i+1:02d}', split=split))
    (source / 'conversion.private.json').write_text(json.dumps({'source_mapping': mapping}), encoding='utf-8')
    manifest = tmp_path / 'split.csv'
    write_manifest(manifest, rows)
    return source, manifest, rows


def write_manifest(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['filename', 'phase', 'source_batch', 'split'])
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize('field,value', [('phase', 'H'), ('filename', '../escape.png'), ('source_batch', 'batch_01')])
def test_manifest_rejects_wrong_phase_path_and_cross_split_group(tmp_path, field, value):
    _, manifest, rows = fixture(tmp_path)
    rows[1][field] = value
    write_manifest(manifest, rows)
    with pytest.raises(ValueError):
        read_split_manifest(manifest)


def test_duplicate_images_fail_before_output_is_created(tmp_path):
    source, manifest, _ = fixture(tmp_path, duplicate=True)
    output = tmp_path / 'prepared'
    with pytest.raises(ValueError, match='duplicate'):
        prepare_initial_dataset(source, output, manifest, (32, 48))
    assert not output.exists()


def test_private_source_batch_mapping_must_match(tmp_path):
    source, manifest, rows = fixture(tmp_path)
    rows[0]['source_batch'] = 'batch_77'
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match='mapping differs'):
        prepare_initial_dataset(source, tmp_path / 'prepared', manifest, (32, 48))


def test_prepare_preserves_fullres_and_explicit_patient_limit(tmp_path):
    source, manifest, _ = fixture(tmp_path)
    output = tmp_path / 'prepared'
    result = prepare_initial_dataset(source, output, manifest, (32, 48))
    assert result['counts'] == {'train': 1, 'val': 1, 'test': 1}
    assert result['patient_split_verified'] is False
    for row in result['records']:
        original = source / 'train/images' / row['filename']
        full = output / 'fullres' / row['split'] / 'images' / row['filename']
        assert full.read_bytes() == original.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_initial_dataset(source, output, manifest, (32, 48))


def test_raw_labelme_ingest_keeps_numbering_but_only_trains_z(tmp_path):
    import base64
    source = tmp_path / 'labelme'
    for i, phase in enumerate(['Q', 'Z', 'Z', 'Z', 'H', 'L']):
        folder = source / f'folder_{i}'
        folder.mkdir(parents=True)
        image = np.full((20, 28, 3), 30 + i * 20, np.uint8)
        ok, encoded = cv2.imencode('.png', image)
        assert ok
        annotation = {'imageHeight':20, 'imageWidth':28, 'imageData':base64.b64encode(encoded).decode(),
                      'shapes':[{'label':phase, 'shape_type':'polygon', 'points':[[3,3],[20,3],[20,15],[3,15]]}]}
        (folder / '1.json').write_text(json.dumps(annotation), encoding='utf-8')
    converted = tmp_path / 'converted'
    conversion = convert_initial_labelme(source, converted)
    assert conversion['samples'] == 5  # L excluded; Q/H retained only for stable source numbering.
    rows = [dict(filename=f'sample_{i+2:05d}.png', phase='Z', source_batch=f'batch_{i+2:02d}', split=split)
            for i, split in enumerate(['train','val','test'])]
    manifest = tmp_path / 'split.csv'
    write_manifest(manifest, rows)
    prepared = prepare_initial_dataset(converted, tmp_path / 'prepared', manifest, (16, 24))
    assert prepared['counts'] == {'train':1,'val':1,'test':1}
    assert {r['filename'] for r in prepared['records']} == {'sample_00002.png','sample_00003.png','sample_00004.png'}


def test_fresh_train_original_resolution_evaluation_and_one_shot_test(tmp_path):
    torch.set_num_threads(2)
    source, manifest, _ = fixture(tmp_path)
    prepared = tmp_path / 'prepared'
    prepare_initial_dataset(source, prepared, manifest, (32, 48))
    model_root = Path(__file__).resolve().parents[1]
    config = load_config(str(model_root / 'configs/initial_z_holdout.yaml'))
    config['data'].update({f'{s}_data_dir': str(prepared / 'resized' / s) for s in ['train', 'val', 'test']})
    config['data'].update(image_size=[32, 48], batch_size=1, pin_memory=False, augmentations={})
    config['model'].update(base_filters=2, depth=2)
    config['training'].update(epochs=1, checkpoint_dir=str(tmp_path / 'checkpoints'), mixed_precision=False)
    config['logging'].update(log_dir=str(tmp_path / 'logs'))
    config['inference'].update(model_path=str(tmp_path / 'checkpoints/best_model.pt'), output_dir=str(tmp_path / 'predictions'))
    config['device']['use_cuda'] = False
    config['experiment'].update(split_manifest=str(manifest), prepared_data_dir=str(prepared))
    config_path = tmp_path / 'test.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    summary = train_initial(config, config_path)
    assert summary['best_epoch'] == 1
    with pytest.raises(FileExistsError):
        train_initial(config, config_path)
    result = evaluate_initial(config, 'test')
    assert result['sample_count'] == 1
    assert result['source_shapes'] == [[64, 96]] or result['source_shapes'] == [(64, 96)]
    assert result['patient_split_verified'] is False
    assert result['distance_units'] == 'original_image_pixels'
    with pytest.raises(FileExistsError):
        evaluate_initial(config, 'test')
    lock = json.loads((tmp_path / 'logs/test_evaluation.lock.json').read_text(encoding='utf-8'))
    assert lock['evaluations_completed'] == 1
