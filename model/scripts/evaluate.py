#!/usr/bin/env python
"""Evaluation entry point for ICG U-Net pipeline."""

import argparse
import sys
import json
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from icg_pipeline import load_config, Evaluator
from icg_pipeline.data import get_dataloader, read_source_image, read_binary_mask
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
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='Override evaluation device')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    if args.device:
        config['device']['use_cuda'] = args.device == 'cuda'
    
    config['inference']['model_path'] = str(Path(args.model).resolve())
    predictor = ICGPredictor(config)
    
    # Create dataloaders
    dataset = get_dataloader(config, args.split).dataset
    
    # Create evaluator
    evaluator = Evaluator(config)
    
    # Run evaluation
    print(f"Evaluating on {args.split} set...")
    def pairs():
        for image_path, mask_path in dataset.pairs:
            image = read_source_image(image_path)
            gt = read_binary_mask(mask_path)
            if image.shape[:2] != gt.shape:
                raise ValueError(f"Image/mask size mismatch: {image_path}")
            pred = predictor.predict_image(image)
            yield {"mask": pred["mask"], "image_path": str(image_path)}, gt
    
    # Compute metrics
    results = evaluator.evaluate_pairs(pairs())
    results['split'] = args.split
    results['distance_units'] = 'original_image_pixels'
    results['sample_count'] = len(dataset)
    results['checkpoint_training_mode'] = predictor.checkpoint_training_mode
    results['evaluation_scope'] = 'training_fit_only' if args.split == 'train' else 'held_out_split_patient_independence_unverified'
    if args.split == 'train':
        print('TRAINING-FIT DIAGNOSTIC: these are not independent validation/test scores.')

    def json_safe(value):
        if isinstance(value, dict):
            return {k: json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_safe(v) for v in value]
        return None if isinstance(value, float) and not np.isfinite(value) else value
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(json_safe(results), f, indent=2, allow_nan=False, ensure_ascii=False)
    
    # Print summary
    print("\nEvaluation Results:")
    print("=" * 50)
    for key, value in results['aggregated'].items():
        print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    main()
