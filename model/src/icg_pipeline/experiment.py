"""Portable, explicitly within-patient initial-training experiment workflow.

Clinical images, masks, private source mappings, predictions and weights stay local.
Source-folder groups never imply patient independence.
"""
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

from .data import read_binary_mask, read_source_image, write_image

MODEL_ROOT = Path(__file__).resolve().parents[2]
SCOPE = 'within_patient_source_batch_holdout_NOT_independent_patient'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _safe(value):
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _path(value):
    path = Path(value)
    return path if path.is_absolute() else (MODEL_ROOT / path).resolve()


def read_split_manifest(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'filename', 'phase', 'source_batch', 'split'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Manifest needs filename,phase,source_batch,split')
        rows = [{k: row[k].strip() for k in required} for row in reader]
    names, groups = set(), {}
    for row in rows:
        name, group, split = row['filename'], row['source_batch'], row['split']
        if (not name or '/' in name or '\\' in name or Path(name).name != name
                or Path(name).suffix.lower() != '.png' or name in names):
            raise ValueError('Manifest filenames must be unique, plain PNG basenames')
        if row['phase'] != 'Z' or split not in {'train', 'val', 'test'} or not group:
            raise ValueError('This pilot requires Z labels and explicit nonempty source groups')
        if group in groups and groups[group] != split:
            raise ValueError('A source batch occurs in multiple splits')
        names.add(name)
        groups[group] = split
    if {r['split'] for r in rows} != {'train', 'val', 'test'}:
        raise ValueError('Nonempty train, val and test sets are required')
    return sorted(rows, key=lambda r: r['filename'])


def convert_initial_labelme(source, destination):
    """Rebuild anonymous Q/Z/H numbering, keeping the identifying map private."""
    from .data.labelme import _annotations, convert_labelme_dataset
    source, sources = _annotations(source, exclude_labels=['L'])
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Choose a new private conversion directory')
    manifest = destination.with_suffix('.source_manifest.private.csv')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['annotation', 'split', 'group_id'])
        writer.writerows((p.relative_to(source).as_posix(), 'train', 'single_group_user_reported') for p in sources)
    return convert_labelme_dataset(source, destination, manifest, ['Q', 'Z', 'H'],
                                   train_only=True, exclude_labels=['L'])


def prepare_initial_dataset(converted_root, output, split_manifest, image_size=(288, 512)):
    """Apply the reviewed anonymous split to private converted full-resolution data."""
    source, output = Path(converted_root).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Source and output directories must be separate')
    if output.exists():
        raise FileExistsError('Refuse to overwrite an existing prepared dataset')
    height, width = map(int, image_size)
    if min(height, width) <= 0:
        raise ValueError('image_size must contain positive H,W values')
    rows = read_split_manifest(split_manifest)
    conversion = json.loads((source / 'conversion.private.json').read_text(encoding='utf-8'))
    mapping = conversion['source_mapping']
    by_name = {r['filename']: r for r in mapping}
    if len(by_name) != len(mapping):
        raise ValueError('Ambiguous converted filenames')
    parents = sorted({str(Path(r['annotation']).parent) for r in mapping})
    groups = {parent: f'batch_{i+1:02d}' for i, parent in enumerate(parents)}
    if {r['filename'] for r in mapping if r['labels'] == ['Z']} != {r['filename'] for r in rows}:
        raise ValueError('Manifest must cover all and only converted Z images; check the private mapping')
    fingerprints, records = set(), []
    for row in rows:
        item = by_name[row['filename']]
        if (item['labels'] != ['Z'] or item['split'] != 'train'
                or groups[str(Path(item['annotation']).parent)] != row['source_batch']):
            raise ValueError('Private source mapping differs from the reviewed source-batch split')
        image = read_source_image(source / 'train/images' / row['filename'])
        mask = read_binary_mask(source / 'train/masks' / row['filename'])
        if image.shape[:2] != mask.shape or not mask.any() or (mask > 0).all():
            raise ValueError('Inspect mismatched, empty or full target masks')
        digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        if digest in fingerprints:
            raise ValueError('Exact duplicate image in initial-training data')
        fingerprints.add(digest)
        records.append(dict(row, source_image_sha256=digest, source_size=list(mask.shape)))
    # Validate every source before creating the new prepared directory.
    output.mkdir(parents=True)
    for row in rows:
        name, split = row['filename'], row['split']
        image = read_source_image(source / 'train/images' / name)
        mask = read_binary_mask(source / 'train/masks' / name)
        for kind in ['images', 'masks']:
            destination = output / 'fullres' / split / kind / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / 'train' / kind / name, destination)
        write_image(str(output / 'resized' / split / 'images' / name),
                    cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))
        write_image(str(output / 'resized' / split / 'masks' / name),
                    cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST))
    report = {'scope': SCOPE, 'patient_split_verified': False,
              'patient_group': 'single_group_user_reported', 'manifest_sha256': _sha(split_manifest),
              'input_shape': [height, width], 'counts': {s: sum(r['split'] == s for r in rows) for s in ['train', 'val', 'test']},
              'records': records}
    _json(output / 'preparation.private.json', report)
    return report


def _prepared(config):
    experiment = config['experiment']
    manifest_path = _path(experiment['split_manifest'])
    rows = read_split_manifest(manifest_path)
    root = _path(experiment['prepared_data_dir'])
    report = json.loads((root / 'preparation.private.json').read_text(encoding='utf-8'))
    if report['manifest_sha256'] != _sha(manifest_path) or report['scope'] != SCOPE:
        raise ValueError('Prepared data do not match the frozen split')
    if list(config['data']['image_size']) != report['input_shape']:
        raise ValueError('Prepare images at the configured input size')
    for split in ['train', 'val', 'test']:
        expected = {r['filename'] for r in rows if r['split'] == split}
        configured = Path(config['data'][f'{split}_data_dir']).resolve()
        if configured != (root / 'resized' / split).resolve():
            raise ValueError('Training data paths must point to the prepared resized split')
        for version in ['resized', 'fullres']:
            for kind in ['images', 'masks']:
                actual = {p.name for p in (root / version / split / kind).iterdir() if p.is_file()}
                if actual != expected:
                    raise ValueError('Prepared split coverage differs from the manifest')
    return root, rows, manifest_path


def train_initial(config, config_path):
    """Use the same Trainer as train.py; reject resuming/overwriting this pilot."""
    import torch
    from .training import Trainer
    _root, rows, manifest_path = _prepared(config)
    if config['training'].get('overfit_only', False):
        raise ValueError('Initial holdout training cannot use overfit_only')
    log = Path(config['logging']['log_dir'])
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    if (log.exists() and any(log.iterdir())) or (checkpoint_dir.exists() and any(checkpoint_dir.iterdir())):
        raise FileExistsError('Choose fresh log/checkpoint directories; do not overwrite a completed run')
    trainer = Trainer(config)
    digest = hashlib.sha256()
    for name, value in trainer.model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    assert trainer.current_epoch == 0 and not trainer.optimizer.state
    _json(log / 'initialization_provenance.json', {
        'fresh_initialization': True, 'resume': None, 'initial_epoch': 0,
        'optimizer_state_entries': 0, 'initial_model_state_sha256': digest.hexdigest(),
        'config_file_sha256': _sha(config_path), 'manifest_sha256': _sha(manifest_path),
        'torch_version': str(torch.__version__), 'scope': SCOPE, 'patient_split_verified': False,
        'counts': {s: sum(r['split'] == s for r in rows) for s in ['train', 'val', 'test']},
    })
    result = trainer.train()
    checkpoint = torch.load(checkpoint_dir / 'best_model.pt', map_location='cpu', weights_only=True)
    summary = {'scope': SCOPE, 'completed_epochs': trainer.current_epoch + 1,
               'best_epoch': checkpoint['epoch'] + 1, 'best_validation_dice_input_resolution': result['best_val_dice'],
               'config_file_sha256': _sha(config_path), 'manifest_sha256': _sha(manifest_path)}
    _json(log / 'training_summary.json', summary)
    return summary


def evaluate_initial(config, split='test'):
    """Evaluate source-resolution masks, with a one-shot test ledger and checkpoint hash."""
    import torch
    from .eval import Evaluator
    from .inference import ICGPredictor
    if split not in {'train', 'val', 'test'}:
        raise ValueError('Unknown split')
    root, rows, manifest_path = _prepared(config)
    log = Path(config['logging']['log_dir'])
    provenance = json.loads((log / 'initialization_provenance.json').read_text(encoding='utf-8'))
    if provenance['manifest_sha256'] != _sha(manifest_path):
        raise ValueError('Split changed after training')
    checkpoint_path = Path(config['inference']['model_path'])
    if checkpoint_path.resolve() != (Path(config['training']['checkpoint_dir']) / 'best_model.pt').resolve():
        raise ValueError('Evaluate the validation-selected best checkpoint')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint['training_mode'] != 'train_val':
        raise ValueError('Do not evaluate a train-fit checkpoint as held-out training')
    # Changing data-independent inference options after looking at test is still tuning.
    for key in ['model', 'data', 'training', 'inference']:
        if checkpoint['config'][key] != config[key]:
            raise ValueError('Evaluation config differs from the frozen training/checkpoint config')
    epoch = checkpoint['epoch'] + 1
    del checkpoint
    result_path = log / f'{split}_evaluation_original_resolution.json'
    if result_path.exists():
        raise FileExistsError('Evaluation already exists; retain the original result')
    predictor = ICGPredictor(config)
    ledger = log / 'test_evaluation.lock.json'
    if split == 'test':
        with ledger.open('x', encoding='utf-8') as handle:
            json.dump({'state': 'started', 'checkpoint_sha256': _sha(checkpoint_path),
                       'manifest_sha256': _sha(manifest_path)}, handle)
    selected = [r for r in rows if r['split'] == split]
    dimensions = set()
    def pairs():
        for row in selected:
            name = row['filename']
            image = read_source_image(root / 'fullres' / split / 'images' / name)
            gt = read_binary_mask(root / 'fullres' / split / 'masks' / name)
            dimensions.add(tuple(gt.shape))
            prediction = predictor.predict_image(image)
            write_image(str(Path(config['inference']['output_dir']) / split / 'masks_original_resolution' / name), prediction['mask'])
            yield {'mask': prediction['mask'], 'image_path': name}, gt
    result = Evaluator(config).evaluate_pairs(pairs())
    result.update(split=split, sample_count=len(selected), checkpoint_epoch=epoch,
                  evaluation_scope='training_fit_only' if split == 'train' else SCOPE,
                  patient_split_verified=False, distance_units='original_image_pixels',
                  source_shapes=sorted(dimensions), manifest_sha256=_sha(manifest_path))
    _json(result_path, result)
    if split == 'test':
        _json(ledger, {'state': 'complete', 'evaluations_completed': 1,
                       'checkpoint_sha256': _sha(checkpoint_path), 'manifest_sha256': _sha(manifest_path)})
    return _safe(result)
