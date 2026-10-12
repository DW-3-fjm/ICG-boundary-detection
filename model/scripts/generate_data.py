#!/usr/bin/env python
"""Generate synthetic ICG data for testing and development."""

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from icg_pipeline.data import write_image, _read_image


def generate_synthetic_icg_data(
    output_dir: str,
    num_samples: int = 100,
    image_size: tuple = (512, 512),
    split_ratios: tuple = (0.7, 0.15, 0.15)
):
    """Generate synthetic ICG fluorescence images with ground truth masks.
    
    Creates realistic-looking ICG data with:
    - Vascular-like structures
    - Fluorescence intensity variations
    - Noise and artifacts
    """
    output_dir = Path(output_dir)
    splits = ['train', 'val', 'test']
    split_names = ['train', 'val', 'test']
    
    # Create directories
    for split in split_names:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    # Calculate split sizes
    train_size = int(num_samples * split_ratios[0])
    val_size = int(num_samples * split_ratios[1])
    test_size = num_samples - train_size - val_size
    split_sizes = [train_size, val_size, test_size]
    
    sample_idx = 0
    
    for split_idx, (split, size) in enumerate(zip(split_names, split_sizes)):
        print(f"Generating {size} samples for {split}...")
        
        for i in range(size):
            # Generate synthetic image
            image, mask = generate_sample(image_size)
            
            # Save
            img_path = output_dir / split / 'images' / f'sample_{sample_idx:05d}.png'
            mask_path = output_dir / split / 'masks' / f'sample_{sample_idx:05d}.png'
            
            write_image(str(img_path), image)
            write_image(str(mask_path), mask)
            
            sample_idx += 1
    
    print(f"Generated {num_samples} samples in {output_dir}")


def generate_sample(image_size: tuple) -> tuple:
    """Generate a single synthetic ICG sample."""
    h, w = image_size
    
    # Create base image (dark background)
    image = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Generate vascular-like structure (ground truth)
    num_vessels = np.random.randint(1, 4)
    
    for _ in range(num_vessels):
        # Random vessel path
        start_x = np.random.randint(w // 4, 3 * w // 4)
        start_y = np.random.randint(h // 4, 3 * h // 4)
        length = np.random.randint(50, 200)
        thickness = np.random.randint(3, 12)
        
        # Create curved path
        points = []
        x, y = start_x, start_y
        angle = np.random.uniform(0, 2 * np.pi)
        
        for _ in range(length):
            points.append([x, y])
            angle += np.random.uniform(-0.3, 0.3)
            x += int(np.cos(angle) * 2)
            y += int(np.sin(angle) * 2)
            x = np.clip(x, thickness, w - thickness - 1)
            y = np.clip(y, thickness, h - thickness - 1)
        
        points = np.array(points, dtype=np.int32)
        
        # Draw vessel on mask
        cv2.polylines(mask, [points], False, 255, thickness)
    
    # Fill mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 255, -1)
    
    # Generate fluorescence image
    # Base fluorescence in vessel regions
    vessel_mask = mask > 0
    image[vessel_mask] = np.random.uniform(150, 255, vessel_mask.sum())
    
    # Add background autofluorescence
    bg_noise = np.random.normal(10, 5, (h, w))
    image[~vessel_mask] = np.clip(bg_noise[~vessel_mask], 0, 50)
    
    # Add Gaussian blur for realistic fluorescence spread
    image = cv2.GaussianBlur(image, (5, 5), 2)
    
    # Add noise
    noise = np.random.normal(0, 8, (h, w))
    image = np.clip(image + noise, 0, 255)
    
    # Add motion blur occasionally
    if np.random.random() < 0.3:
        kernel_size = np.random.randint(3, 7)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
        image = cv2.filter2D(image, -1, kernel)
    
    # Convert to uint8
    image = image.astype(np.uint8)
    
    return image, mask


def create_video_from_samples(data_dir: str, output_path: str, fps: int = 30, duration: float = 10):
    """Create a video from generated samples for testing video inference."""
    data_dir = Path(data_dir)
    image_files = sorted(list((data_dir / 'test' / 'images').glob('*.png')))
    
    if not image_files:
        print("No images found. Generate data first.")
        return
    
    # Read first frame to get size
    first_frame = _read_image(image_files[0], 1)
    h, w = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h), isColor=False)
    
    total_frames = int(fps * duration)
    for i in range(total_frames):
        idx = i % len(image_files)
        frame = _read_image(image_files[idx], 1)
        out.write(frame)
    
    out.release()
    print(f"Created video: {output_path} ({total_frames} frames)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic ICG data')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=200, help='Number of samples')
    parser.add_argument('--image-size', type=int, nargs=2, default=[512, 512], help='Image size (H W)')
    parser.add_argument('--create-video', action='store_true', help='Create test video')
    parser.add_argument('--video-output', type=str, default='test_video.mp4', help='Video output path')
    
    args = parser.parse_args()
    
    generate_synthetic_icg_data(args.output, args.num_samples, tuple(args.image_size))
    
    if args.create_video:
        create_video_from_samples(args.output, args.video_output)
