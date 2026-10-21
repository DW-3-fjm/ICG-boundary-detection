#!/usr/bin/env python
"""Evaluate only the requested split at source resolution."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from icg_pipeline.utils import load_config, write_json
from icg_pipeline.eval import evaluate_source_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs/config.yaml'))
    parser.add_argument('--model', required=True)
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output', default='eval_results.json')
    parser.add_argument('--overlays', help='Optional directory for original-resolution masks and overlays')
    args = parser.parse_args()
    config = load_config(args.config)
    config['inference']['model_path'] = str(Path(args.model).resolve())
    results = evaluate_source_split(config, args.split, output_dir=args.overlays)
    write_json(args.output, results)
    for key, value in results['aggregated'].items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()
