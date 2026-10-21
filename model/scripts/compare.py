#!/usr/bin/env python
"""Run the six segmentation configurations sequentially; test is a separate stage."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icg_pipeline.comparison import run_validation, run_frozen_test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', help='Original-resolution root containing train/, val/, test/')
    parser.add_argument('--manifest', help='Frozen split_manifest.json from the training review')
    parser.add_argument('--output-root', required=True, help='New local artifact directory')
    parser.add_argument('--architectures', nargs='+', default=['unet', 'unet_resnet18'],
                        choices=['unet', 'unet_resnet18'])
    parser.add_argument('--losses', nargs='+', default=['control', 'tversky', 'combined'],
                        choices=['control', 'tversky', 'combined'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--epochs', type=int, help='Override for software smoke runs only')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--evaluate-test', action='store_true', help='Evaluate the frozen checkpoints once; never retrain')
    args = parser.parse_args()
    if args.evaluate_test:
        if args.data_root or args.manifest or args.epochs or args.device:
            parser.error('Test stage accepts only the frozen --output-root and --evaluate-test')
        run_frozen_test(args.output_root)
    else:
        if not args.data_root or not args.manifest:
            parser.error('Validation stage requires --data-root and --manifest')
        if args.epochs is not None and args.epochs < 1:
            parser.error('--epochs must be positive')
        configs = [ROOT / 'configs' / f'compare_{architecture}_{loss}.yaml'
                   for architecture in args.architectures for loss in args.losses]
        run_validation(configs, args.seeds, args.data_root, args.manifest, args.output_root, args.epochs, args.device)


if __name__ == '__main__':
    main()
