#!/usr/bin/env python
"""Evaluation entry point for ICG U-Net pipeline."""

import argparse
import sys
import json
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from icg_pipeline import load_config, Evaluator
from icg_pipeline.data import get_dataloaders
from icg_pipeline.inference import ICGPredictor


def main():
    parser = argparse.ArgumentParser(description='Evaluate ICG U-Net model')
    parser.add_argument('--config', type=str, default=str(Path(__file__).parent.parent / 'configs' / 'config.yaml'),
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='eval_results.json',
                        help='Path to save evaluation results')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    config['inference']['model_path'] = str(Path(args.model).resolve())
    predictor = ICGPredictor(config)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config)
    loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}[args.split]
    
    # Create evaluator
    evaluator = Evaluator(config)
    
    # Run evaluation
    predictions = []
    ground_truths = []
    
    print(f"Evaluating on {args.split} set...")
    for batch in loader:
        images = batch['image']
        masks = batch['mask']
        
        for i in range(images.size(0)):
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            if img.shape[2] == 1:
                img = img[:, :, 0]
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            gt = masks[i].numpy()
            
            # Denormalize image
            img = (img * 0.5 + 0.5) * 255
            img = img.astype(np.uint8)
            
            pred = predictor.predict_image(img)
            predictions.append(pred)
            ground_truths.append(gt)
    
    # Compute metrics
    results = evaluator.evaluate_dataset(predictions, ground_truths)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nEvaluation Results:")
    print("=" * 50)
    for key, value in results['aggregated'].items():
        print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    main()