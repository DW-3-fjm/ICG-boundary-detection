"""Training pipeline for ICG U-Net segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import time
import random
import numpy as np

from ..utils import AverageMeter, EarlyStopping, save_checkpoint, get_device, setup_logging
from ..models import UNet
from ..data import get_dataloader

logger = logging.getLogger(__name__)


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W), targets: (B, H, W)
        # FP16 pixel sums overflow above 65504, common for 288x512/512x512.
        # Keep Dice reductions in FP32 even inside CUDA autocast.
        probs = F.softmax(logits.float(), dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """Combined Dice and Cross Entropy loss."""
    
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(smooth)
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        return self.dice_weight * dice + self.ce_weight * ce


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def get_loss_function(config: Dict[str, Any]) -> nn.Module:
    """Get loss function based on config."""
    loss_config = config['training']
    loss_name = loss_config['loss_function']
    params = loss_config.get('loss_params', {})
    
    if loss_name == 'dice':
        return DiceLoss(**params)
    elif loss_name == 'dice_ce':
        return DiceCELoss(**params)
    elif loss_name == 'focal':
        return FocalLoss(**params)
    elif loss_name == 'ce':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Get optimizer based on config."""
    train_config = config['training']
    opt_name = train_config['optimizer']
    lr = train_config['learning_rate']
    weight_decay = train_config.get('weight_decay', 1e-5)
    
    if opt_name == 'adamw':
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler based on config."""
    train_config = config['training']
    sched_name = train_config['scheduler']
    params = train_config.get('scheduler_params', {})
    
    if sched_name == 'cosine_annealing':
        return CosineAnnealingLR(optimizer, **params)
    elif sched_name == 'reduce_on_plateau':
        return ReduceLROnPlateau(optimizer, mode='max', **params)
    elif sched_name == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    gradient_clip: float = 1.0,
    mixed_precision: bool = False
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if mixed_precision and scaler is not None:
            with autocast(device_type=device.type):
                logits = model(images)
                loss = criterion(logits, masks)
            
            scaler.scale(loss).backward()
            
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            
            loss.backward()
            
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
        
        # Compute dice score
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            dice = compute_dice_score(preds, masks)
        
        loss_meter.update(loss.item(), images.size(0))
        dice_meter.update(dice, images.size(0))
    
    return {
        'loss': loss_meter.avg,
        'dice': dice_meter.avg
    }


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mixed_precision: bool = False
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            
            if mixed_precision:
                with autocast(device_type=device.type):
                    logits = model(images)
                    loss = criterion(logits, masks)
            else:
                logits = model(images)
                loss = criterion(logits, masks)
            
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            
            dice = compute_dice_score(preds, masks)
            iou = compute_iou_score(preds, masks)
            
            loss_meter.update(loss.item(), images.size(0))
            dice_meter.update(dice, images.size(0))
            iou_meter.update(iou, images.size(0))
    
    return {
        'loss': loss_meter.avg,
        'dice': dice_meter.avg,
        'iou': iou_meter.avg
    }


def compute_dice_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
    """Compute foreground-only Dice for class-index masks."""
    preds, targets = (preds == 1).float(), (targets == 1).float()
    intersection = (preds * targets).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def compute_iou_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
    """Compute foreground-only IoU for class-index masks."""
    preds, targets = (preds == 1).float(), (targets == 1).float()
    intersection = (preds * targets).sum(dim=(1, 2))
    union = (preds + targets - preds * targets).sum(dim=(1, 2))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


class Trainer:
    """Main trainer class for ICG U-Net."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        if (config['training'].get('monitor_metric', 'val_dice') != 'val_dice'
                or config['training'].get('monitor_mode', 'max') != 'max'
                or config['training'].get('early_stopping', {}).get('mode', 'max') != 'max'):
            raise ValueError('Only validation Dice maximization is supported for checkpointing/early stopping')
        self.device = get_device(config['device'])
        self.logger = setup_logging(config['logging']['log_dir'], config['logging']['level'])
        self.overfit_only = bool(config['training'].get('overfit_only', False))
        if self.overfit_only:
            self.logger.warning('OVERFIT DIAGNOSTIC: scores are measured on training images, NOT independent validation/test')
        
        # Set seed for reproducibility
        from ..utils import set_seed
        set_seed(int(config['training'].get('seed', 42)))
        
        # Create model
        self.model = create_model(config).to(self.device)
        self.logger.info(f"Model created on {self.device}: {self.model.count_parameters():,} parameters")
        
        # Create dataloaders
        self.train_loader = get_dataloader(config, 'train', training=True)
        self.val_loader = get_dataloader(config, 'train' if self.overfit_only else 'val')
        
        # Create loss, optimizer, scheduler
        self.criterion = get_loss_function(config).to(self.device)
        self.optimizer = get_optimizer(self.model, config)
        self.scheduler = get_scheduler(self.optimizer, config)
        
        # Mixed precision
        self.mixed_precision = bool(config['training'].get('mixed_precision', False)) and self.device.type == 'cuda'
        self.scaler = GradScaler('cuda') if self.mixed_precision else None
        
        # Early stopping
        early_stop_config = config['training'].get('early_stopping', {})
        self.early_stopping = EarlyStopping(
            patience=early_stop_config.get('patience', 15),
            min_delta=early_stop_config.get('min_delta', 1e-4),
            mode=early_stop_config.get('mode', 'max')
        )
        
        # Checkpointing
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = config['training'].get('save_top_k', 3)
        if not isinstance(self.save_top_k, int) or self.save_top_k < 0:
            raise ValueError('save_top_k must be a nonnegative integer')
        self.best_metrics = []
        
        # Training state
        self.current_epoch = 0
        self.best_val_dice = float('-inf')
        
        # TensorBoard
        self.writer = None
        if config['logging'].get('use_tensorboard', False):
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(config['logging']['log_dir'])
    
    def train(self) -> Dict[str, float]:
        """Run full training loop."""
        self.logger.info("Starting training...")
        start_time = time.time()
        
        train_config = self.config['training']
        num_epochs = train_config['epochs']
        validation_label = 'Train-fit' if self.overfit_only else 'Val'
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = train_one_epoch(
                self.model, self.train_loader, self.optimizer, self.criterion,
                self.device, self.scaler,
                train_config.get('gradient_clip', 1.0),
                self.mixed_precision
            )
            
            # Validation
            val_metrics = validate(
                self.model, self.val_loader, self.criterion,
                self.device, self.mixed_precision
            )
            
            # Learning rate scheduling
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics['dice'])
            else:
                self.scheduler.step()
            
            # Logging
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Dice: {train_metrics['dice']:.4f} | "
                f"{validation_label} Loss: {val_metrics['loss']:.4f} | "
                f"{validation_label} Dice: {val_metrics['dice']:.4f} | "
                f"{validation_label} IoU: {val_metrics['iou']:.4f} | "
                f"LR: {lr:.2e}"
            )
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('train/loss', train_metrics['loss'], epoch)
                self.writer.add_scalar('train/dice', train_metrics['dice'], epoch)
                score_prefix = 'train_fit' if self.overfit_only else 'val'
                self.writer.add_scalar(f'{score_prefix}/loss', val_metrics['loss'], epoch)
                self.writer.add_scalar(f'{score_prefix}/dice', val_metrics['dice'], epoch)
                self.writer.add_scalar(f'{score_prefix}/iou', val_metrics['iou'], epoch)
                self.writer.add_scalar('lr', lr, epoch)
            
            should_stop = self.early_stopping(val_metrics['dice'])
            # Checkpointing
            val_dice = val_metrics['dice']
            is_best = val_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_dice
            
            # Save top-k checkpoints
            self.save_topk_checkpoint(epoch, val_metrics)
            if is_best:
                self.save_checkpoint('best_model.pt', val_metrics)
            self.save_checkpoint('last_model.pt', val_metrics)
            
            # Early stopping
            if should_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        
        if self.writer:
            self.writer.close()
        
        return {'best_val_dice': self.best_val_dice}
    
    def save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save model checkpoint."""
        filepath = self.checkpoint_dir / filename
        numpy_state = np.random.get_state()
        save_checkpoint({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'training_mode': 'overfit_diagnostic' if self.overfit_only else 'train_val',
            'best_val_dice': self.best_val_dice,
            'best_metrics': self.best_metrics,
            'early_stopping_state': vars(self.early_stopping).copy(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'rng_state': {
                'python': random.getstate(),
                'numpy': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else [],
            },
        }, str(filepath))
        self.logger.info(f"Saved checkpoint: {filepath}")
    
    def save_topk_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save checkpoint if in top-k."""
        val_dice = metrics['dice']
        self.best_metrics.append((val_dice, epoch))
        self.best_metrics.sort(key=lambda x: x[0], reverse=True)
        
        if len(self.best_metrics) > self.save_top_k:
            # Remove worst
            worst_dice, worst_epoch = self.best_metrics.pop()
            worst_path = self.checkpoint_dir / f'epoch_{worst_epoch}_dice_{worst_dice:.4f}.pt'
            if worst_path.exists():
                worst_path.unlink()
        
        # Save current if in top-k
        if (val_dice, epoch) in self.best_metrics:
            filename = f'epoch_{epoch}_dice_{val_dice:.4f}.pt'
            self.save_checkpoint(filename, metrics)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        from ..utils import load_checkpoint
        # Check the mode before applying any state: train-fit and independent
        # validation histories are not interchangeable for model selection.
        expected_mode = 'overfit_diagnostic' if self.overfit_only else 'train_val'
        checkpoint = load_checkpoint(filepath, self.model, self.optimizer, expected_training_mode=expected_mode)
        self.current_epoch = checkpoint.get('epoch', -1) + 1
        self.best_val_dice = checkpoint.get('best_val_dice', checkpoint.get('metrics', {}).get('dice', 0.0))
        self.best_metrics = checkpoint.get('best_metrics', [])
        if checkpoint.get('early_stopping_state'):
            self.early_stopping.__dict__.update(checkpoint['early_stopping_state'])
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if checkpoint.get('rng_state'):
            rng = checkpoint['rng_state']
            random.setstate(rng['python'])
            state = rng['numpy']
            np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
            torch.set_rng_state(rng['torch'])
            if self.device.type == 'cuda' and rng['cuda']:
                torch.cuda.set_rng_state_all(rng['cuda'])
        self.logger.info(f"Loaded checkpoint from {filepath}")


def create_model(config: Dict[str, Any]) -> UNet:
    """Factory function to create model (imported from models)."""
    from ..models import create_model as create_model_fn
    return create_model_fn(config)
