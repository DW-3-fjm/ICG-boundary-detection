#!/usr/bin/env python
"""Prepare private data, train a fresh baseline, or evaluate original-resolution masks."""
import argparse
import json
import sys
from pathlib import Path

MODEL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL / 'src'))

from icg_pipeline.experiment import convert_initial_labelme, prepare_initial_dataset, train_initial, evaluate_initial
from icg_pipeline.utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Use approved local LabelMe or already-converted data')
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument('--source', help='Private original LabelMe directory; original tree must be preserved')
    source.add_argument('--converted-root', help='Private fullres directory with conversion.private.json')
    prepare.add_argument('--output', required=True, help='New private prepared directory')
    prepare.add_argument('--manifest', default=str(MODEL / 'experiments/initial_training_20261019/split_manifest.csv'))
    prepare.add_argument('--image-size', nargs=2, type=int, default=[288, 512], metavar=('HEIGHT', 'WIDTH'))
    for command in ['train', 'evaluate']:
        entry = sub.add_parser(command)
        entry.add_argument('--config', default=str(MODEL / 'configs/initial_z_holdout.yaml'))
        entry.add_argument('--device', choices=['cpu', 'cuda'])
        if command == 'evaluate':
            entry.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    args = parser.parse_args()
    if args.command == 'prepare':
        converted = args.converted_root
        if args.source:
            converted = str(Path(args.output).resolve().with_name(Path(args.output).name + '_conversion_private'))
            convert_initial_labelme(args.source, converted)
        result = prepare_initial_dataset(converted, args.output, args.manifest, args.image_size)
        result = {k: v for k, v in result.items() if k != 'records'}
    else:
        config = load_config(args.config)
        if args.device:
            config['device']['use_cuda'] = args.device == 'cuda'
        result = (train_initial(config, args.config) if args.command == 'train'
                  else evaluate_initial(config, args.split))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
