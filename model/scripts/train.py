#!/usr/bin/env python
"""Training entry point for ICG U-Net pipeline."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from icg_pipeline import load_config, Trainer


def main():
    parser = argparse.ArgumentParser(description='Train ICG U-Net segmentation model')
    parser.add_argument('--config', type=str, default=str(Path(__file__).parent.parent / 'configs' / 'config.yaml'),
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'],
                        help='Override device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        config['device']['use_cuda'] = args.device == 'cuda'
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Run training
    results = trainer.train()
    
    print(f"Training completed. Best validation Dice: {results['best_val_dice']:.4f}")


if __name__ == '__main__':
    main()