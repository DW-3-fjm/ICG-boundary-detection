#!/usr/bin/env python
"""Test known synthetic motion from stills; no claim about real fluorescence fading."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from icg_pipeline.utils import load_config
from icg_pipeline.synthetic import benchmark_synthetic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', help='Required only for model initialization')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--initialization', choices=['reference', 'model'], default='reference')
    parser.add_argument('--frames', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.initialization == 'model' and not args.model:
        parser.error('--model is required for model initialization')
    config = load_config(args.config)
    if args.model:
        config['inference']['model_path'] = str(Path(args.model).resolve())
    result = benchmark_synthetic(config, args.split, args.output, args.frames, args.seed, args.initialization)
    print(f"Evaluated {len(result['sequences'])} synthetic sequences; not a real fading-video result.")


if __name__ == '__main__':
    main()
