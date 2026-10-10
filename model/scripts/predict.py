#!/usr/bin/env python
"""Inference entry point for ICG U-Net pipeline."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from icg_pipeline import load_config, run_inference


def main():
    parser = argparse.ArgumentParser(description='Run ICG U-Net inference')
    parser.add_argument('--config', type=str, default=str(Path(__file__).parent.parent / 'configs' / 'config.yaml'),
                        help='Path to configuration file')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or video')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output file/directory')
    parser.add_argument('--no-track', action='store_true',
                        help='Disable contour tracking')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Disable tracking if requested
    if args.no_track:
        config['tracking']['enabled'] = False
    
    # Run inference
    results = run_inference(args.config, args.input, args.output, config_override=config)
    if 'frames' in results:
        print(f"Processed {results['frame_count']} frames; mean FPS: {results['mean_fps']:.2f}")
        print(f"Tracking metadata: {results['metadata_path']}")
    else:
        print(f"Image processed; mask saved to {results['mask_path']}")


if __name__ == '__main__':
    main()