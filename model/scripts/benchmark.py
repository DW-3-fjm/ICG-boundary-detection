#!/usr/bin/env python
"""Measure synthetic training throughput; this is not a clinical experiment."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from icg_pipeline.models import create_model
from icg_pipeline.training import get_loss_function, get_optimizer
from icg_pipeline.utils import load_config, get_device, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs/config.yaml'))
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--image-size', type=int)
    parser.add_argument('--base-filters', type=int)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--output', default='logs/benchmark.json')
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1:
        parser.error('steps and warmup must be positive')
    config = load_config(args.config)
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.image_size:
        config['data']['image_size'] = [args.image_size] * 2
    if args.base_filters:
        config['model']['base_filters'] = args.base_filters
    device = get_device(config['device'])
    set_seed(int(config['training'].get('seed', 42)))
    model = create_model(config).to(device).train()
    criterion = get_loss_function(config).to(device)
    optimizer = get_optimizer(model, config)
    batch = config['data']['batch_size']
    height, width = config['data']['image_size']
    channels = config['model']['in_channels']
    inputs = torch.randn(batch, channels, height, width, device=device)
    targets = torch.randint(0, 2, (batch, height, width), device=device)
    amp = bool(config['training'].get('mixed_precision', False)) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            loss = criterion(model(inputs), targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip = float(config['training'].get('gradient_clip', 1.0))
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        return loss.detach()

    for _ in range(args.warmup):
        step()
    synchronize()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(args.steps):
        synchronize()
        start = time.perf_counter()
        loss = step()
        synchronize()
        times.append(time.perf_counter() - start)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite loss during benchmark')
    result = {
        'kind': 'synthetic_training_benchmark_not_clinical_results',
        'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
        'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'parameters': model.count_parameters(), 'model': config['model'],
        'image_size': [height, width], 'batch_size': batch, 'mixed_precision': amp,
        'steps': args.steps, 'warmup': args.warmup,
        'mean_step_seconds': statistics.mean(times),
        'median_step_seconds': statistics.median(times),
        'training_images_per_second': batch / statistics.mean(times),
        'peak_allocated_GiB': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None,
        'peak_reserved_GiB': torch.cuda.max_memory_reserved(device) / 2**30 if device.type == 'cuda' else None,
        'limitations': 'Random tensors already on device. Includes forward, loss, backward, gradient clipping and optimizer. Excludes decoding, augmentation, CPU-to-GPU transfer, validation, checkpoint I/O and video processing. Not end-to-end FPS.',
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
