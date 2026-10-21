"""Exercise the teammate-facing benchmark without videos or model downloads."""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def test_reference_initialized_synthetic_benchmark_reports_split_units_and_failures(tmp_path):
    from icg_pipeline.synthetic import benchmark_synthetic
    root = tmp_path / 'val'
    (root / 'images').mkdir(parents=True)
    (root / 'masks').mkdir()
    image = np.random.default_rng(99).integers(20, 235, (96, 128, 3), dtype=np.uint8)
    mask = np.zeros((96, 128), np.uint8)
    mask[20:70, 20:90] = 255
    cv2.imwrite(str(root / 'images/case.png'), image)
    cv2.imwrite(str(root / 'masks/case.png'), mask)
    config = {'data': {'val_data_dir': str(root), 'image_size': [96, 128]},
              'tracking': {'working_size': [96, 128], 'num_points': 40}}
    report = benchmark_synthetic(config, 'val', tmp_path / 'benchmark', frames=4)
    assert report['synthetic_only'] is True
    assert report['distance_units'] == 'working_image_pixels'
    assert len(report['sequences']) == 5
    translation = next(s for s in report['sequences'] if s['scenario'] == 'translation')
    assert translation['source_split'] == 'val'
    assert translation['metrics']['mean_contour_distance'] < 2
    assert (tmp_path / 'benchmark/summary.json').is_file()
