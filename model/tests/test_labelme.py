"""Synthetic LabelMe fixtures test conversion semantics without using patient data."""
import base64
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from icg_pipeline.data import ICGDataset, read_binary_mask, write_image
from icg_pipeline.data.labelme import convert_labelme_dataset, make_manifest_template


def sample(path, level=100):
    image = np.full((16, 20, 3), level, np.uint8)
    ok, encoded = cv2.imencode('.png', image)
    assert ok
    annotation = {
        'version': '5.0.0', 'imageHeight': 16, 'imageWidth': 20,
        'imagePath': 'missing.png', 'imageData': base64.b64encode(encoded).decode(),
        'shapes': [
            {'label': 'target', 'shape_type': 'polygon', 'points': [[1, 1], [7, 1], [7, 7], [1, 7]]},
            {'label': 'target', 'shape_type': 'polygon', 'points': [[4, 4], [10, 4], [10, 10], [4, 10]]},
            {'label': 'other', 'shape_type': 'polygon', 'points': [[12, 2], [18, 2], [18, 8], [12, 8]]},
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(annotation), encoding='utf-8')
    return annotation


def fixture_data(tmp_path):
    root = tmp_path / '标注'
    rows = []
    for index, split in enumerate(('train', 'val', 'test')):
        path = root / f'folder_{index}' / '1.json'  # duplicate basenames are intentional
        sample(path, level=50 + index)
        rows.append((path.relative_to(root).as_posix(), split, f'patient_{index}'))
    manifest = tmp_path / 'split.csv'
    write_manifest(manifest, rows)
    return root, manifest, rows


def write_manifest(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['annotation', 'split', 'group_id'])
        writer.writerows(rows)


def convert(root, output, manifest, **kwargs):
    return convert_labelme_dataset(root, output, manifest,
                                   kwargs.get('foreground', ['target']), kwargs.get('background', ['other']))


def test_conversion_unions_polygons_preserves_geometry_and_originals(tmp_path):
    root, manifest, _ = fixture_data(tmp_path)
    originals = {p: p.read_bytes() for p in root.rglob('*.json')}
    output = tmp_path / 'prepared'
    report = convert(root, output, manifest)
    assert report['samples'] == 3
    assert report['splits'] == {'test': 1, 'train': 1, 'val': 1}
    dataset = ICGDataset(output / 'train', image_size=(16, 20), channels=3, is_train=False)
    mask = dataset[0]['mask'].numpy()
    assert mask[5, 5] == 1  # overlap is a union, not a hole
    assert mask[2, 2] == mask[9, 9] == 1
    assert mask[4, 14] == 0  # explicit background label
    assert mask.shape == (16, 20)
    assert all(p.read_bytes() == contents for p, contents in originals.items())
    assert len({p.name for p in output.glob('*/images/*.png')}) == 3
    provenance = json.loads((output/'conversion.private.json').read_text(encoding='utf-8'))
    assert all(row['labels'] == ['other','target'] for row in provenance['source_mapping'])
    with (output / 'groups.csv').open(encoding='utf-8-sig', newline='') as handle:
        assert len(list(csv.DictReader(handle))) == 3


def test_template_leaves_patient_groups_and_splits_unassigned(tmp_path):
    root, _, _ = fixture_data(tmp_path)
    destination = tmp_path / 'template.csv'
    assert make_manifest_template(root, destination) == 3
    with destination.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert all(not row['split'] and not row['group_id'] for row in rows)
    with pytest.raises(FileExistsError):
        make_manifest_template(root, destination)


@pytest.mark.parametrize('kind', ['unknown_label', 'group_leak', 'missing_row', 'duplicate_image', 'bad_dimensions', 'line', 'outside', 'typo'])
def test_conversion_rejects_invalid_input_before_writing_output(tmp_path, kind):
    root, manifest, rows = fixture_data(tmp_path)
    first = root / rows[0][0]
    annotation = json.loads(first.read_text(encoding='utf-8'))
    kwargs = {}
    if kind == 'unknown_label':
        annotation['shapes'][0]['label'] = 'unassigned'
    elif kind == 'group_leak':
        rows[1] = (rows[1][0], rows[1][1], rows[0][2])
    elif kind == 'missing_row':
        rows.pop()
    elif kind == 'duplicate_image':
        second = root / rows[1][0]
        other = json.loads(second.read_text(encoding='utf-8'))
        other['imageData'] = annotation['imageData']
        second.write_text(json.dumps(other), encoding='utf-8')
    elif kind == 'bad_dimensions':
        annotation['imageWidth'] = 19
    elif kind == 'line':
        annotation['shapes'][0]['shape_type'] = 'linestrip'
    elif kind == 'outside':
        annotation['shapes'][0]['points'][0] = [-10, 2]
    elif kind == 'typo':
        kwargs['foreground'] = ['target', 'typo']
    first.write_text(json.dumps(annotation), encoding='utf-8')
    write_manifest(manifest, rows)
    output = tmp_path / 'prepared'
    with pytest.raises(ValueError):
        convert(root, output, manifest, **kwargs)
    assert not output.exists()


def test_image_edge_vertices_tolerate_floating_roundoff(tmp_path):
    root, manifest, rows = fixture_data(tmp_path)
    first = root / rows[0][0]
    annotation = json.loads(first.read_text(encoding='utf-8'))
    annotation['shapes'][0]['points'] = [[-1e-14, 0], [20, 0], [20, 16], [0, 16]]
    first.write_text(json.dumps(annotation), encoding='utf-8')
    output = tmp_path / 'prepared'
    convert(root, output, manifest)
    assert read_binary_mask(next((output / 'train/masks').glob('*.png'))).min() == 255


def test_converter_preserves_existing_output_and_rejects_source_overlap(tmp_path):
    root, manifest, _ = fixture_data(tmp_path)
    output = tmp_path / 'prepared'
    output.mkdir()
    marker = output / 'keep.txt'
    marker.write_text('keep')
    with pytest.raises(FileExistsError):
        convert(root, output, manifest)
    assert marker.read_text() == 'keep'
    with pytest.raises(ValueError, match='separate'):
        convert(root, root / 'prepared', manifest)


def test_embedded_external_disagreement_is_rejected(tmp_path):
    root, manifest, rows = fixture_data(tmp_path)
    write_image(str((root / rows[0][0]).parent / 'missing.png'), np.zeros((16, 20, 3), np.uint8))
    with pytest.raises(ValueError, match='disagree'):
        convert(root, tmp_path / 'prepared', manifest)


def test_external_image_is_supported_when_embedded_image_is_absent(tmp_path):
    root, manifest, rows = fixture_data(tmp_path)
    first = root / rows[0][0]
    annotation = json.loads(first.read_text(encoding='utf-8'))
    image = cv2.imdecode(np.frombuffer(base64.b64decode(annotation['imageData']), np.uint8), cv2.IMREAD_UNCHANGED)
    write_image(str(first.parent / 'missing.png'), image)
    annotation['imageData'] = None
    first.write_text(json.dumps(annotation), encoding='utf-8')
    assert convert(root, tmp_path / 'prepared', manifest)['samples'] == 3


def test_single_patient_train_only_conversion_excludes_whole_unknown_phase_samples(tmp_path):
    root, manifest, rows = fixture_data(tmp_path)
    first = root / rows[0][0]
    annotation = json.loads(first.read_text(encoding='utf-8'))
    annotation['shapes'][0]['label'] = 'unknown_phase'
    first.write_text(json.dumps(annotation), encoding='utf-8')
    template = tmp_path/'selected.csv'
    assert make_manifest_template(root, template, exclude_labels=['unknown_phase']) == 2
    rows = [(row[0], 'train', 'single_patient') for row in rows[1:]]
    write_manifest(manifest, rows)
    output = tmp_path/'prepared'
    report = convert_labelme_dataset(root, output, manifest, ['target'], ['other'],
                                    train_only=True, exclude_labels=['unknown_phase'])
    assert report['samples'] == 2
    assert report['train_only']
    assert not (output/'val').exists() and not (output/'test').exists()
    assert report['excluded_labels'] == ['unknown_phase']
    assert all(row['group_id'] == 'single_patient' for row in json.loads((output/'conversion.private.json').read_text())['source_mapping'])
