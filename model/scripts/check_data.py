#!/usr/bin/env python
"""Read-only preflight for paired binary masks, duplicates and optional group splits."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from icg_pipeline.data import ICGDataset, read_binary_mask, read_source_image
from icg_pipeline.utils import load_config


def check_data(config, manifest=None, splits=('train', 'val', 'test')):
    if not splits or len(set(splits)) != len(splits) or set(splits) - {'train', 'val', 'test'}:
        raise ValueError('Select unique train/val/test splits')
    errors, warnings, stats, fingerprints = [], [], {}, {}
    expected = set()
    for split in splits:
        try:
            dataset = ICGDataset(config['data'][f'{split}_data_dir'], is_train=False)
        except (ValueError, OSError) as exc:
            errors.append(f'{split}: {exc}')
            continue
        seen_names = set()
        empty, full, dimensions = 0, 0, set()
        for image_path, mask_path in dataset.pairs:
            expected.add((split, image_path.name))
            if image_path.stem in seen_names:
                errors.append(f'{split}: ambiguous duplicate image stem: {image_path.stem}')
            seen_names.add(image_path.stem)
            try:
                image = read_source_image(image_path)
                mask = read_binary_mask(mask_path)
            except (ValueError, OSError) as exc:
                errors.append(f'{split}: {image_path.name}: {exc}')
                continue
            if image.shape[:2] != mask.shape:
                errors.append(f'{split}: image/mask shape mismatch: {image_path.name}')
            if image.dtype != np.uint8:
                errors.append(f'{split}: unsupported source dtype {image.dtype}: {image_path.name}; current model normalizes 8-bit data')
            values = np.unique(mask)
            if len(values) > 2 or (len(values) == 2 and values[0] != 0):
                errors.append(f'{split}: mask must encode background=0 and one foreground value: {mask_path.name}')
            empty += int(not mask.any())
            full += int((mask > 0).all())
            dimensions.add(tuple(image.shape[:2]))
            # Decoded-pixel hash also catches identical images in different lossless formats.
            digest = hashlib.sha256(str((image.shape, image.dtype)).encode() + image.tobytes()).hexdigest()
            if digest in fingerprints and fingerprints[digest][0] != split:
                errors.append(f'Exact duplicate image across splits: {fingerprints[digest]} and {(split,image_path.name)}')
            fingerprints[digest] = (split, image_path.name)
        stats[split] = {'pairs':len(dataset), 'empty_masks':empty, 'full_masks':full, 'image_sizes':sorted(dimensions)}
        if empty or full:
            warnings.append(f'{split}: inspect {empty} empty and {full} full masks; these may be intentional')
    verified = False
    if manifest:
        with Path(manifest).open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if not {'split','filename','group_id'} <= set(reader.fieldnames or []):
                raise ValueError('Manifest columns must include split,filename,group_id')
            rows = list(reader)
        present, groups = set(), {}
        for row in rows:
            key = row['split'].strip(), row['filename'].strip()
            group = row['group_id'].strip()
            if key in present or not group:
                errors.append(f'Duplicate manifest row or empty group_id: {key}')
            present.add(key)
            if group in groups and groups[group] != key[0]:
                errors.append(f'A patient/procedure group occurs in multiple splits: {key}')
            groups[group] = key[0]
        if present != expected:
            errors.append(f'Manifest coverage mismatch: missing={len(expected-present)}, extra={len(present-expected)}')
        verified = not errors
    else:
        warnings.append('Patient/procedure split NOT verified: supply a private split,filename,group_id CSV. Image hashes cannot detect adjacent-frame leakage.')
    if set(splits) == {'train'}:
        warnings.append('Train-only data: no independent validation or test set is available.')
    return {'ok':not errors, 'patient_split_verified':verified, 'checked_splits':list(splits), 'splits':stats, 'errors':errors, 'warnings':warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs/config.yaml'))
    parser.add_argument('--manifest', help='Private CSV; use patient ID as group_id if multiple operations belong to one patient')
    parser.add_argument('--output', default='logs/data_check.json')
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val', 'test'])
    args = parser.parse_args()
    report = check_data(load_config(args.config), args.manifest, args.splits)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()
