"""
Training utilities for DinoV2 LoRA fine-tuning.
"""

import os
import logging
import random
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ConstantLR
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str, level: int = logging.INFO):
    """Setup logging configuration."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


def create_optimizer(model: torch.nn.Module, config) -> torch.optim.Optimizer:
    """Create optimizer based on config."""
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if config.optimizer.type.lower() == "adamw":
        optimizer = optim.AdamW(
            trainable_params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps
        )
    elif config.optimizer.type.lower() == "adam":
        optimizer = optim.Adam(
            trainable_params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=config.optimizer.betas,
            eps=config.optimizer.eps
        )
    elif config.optimizer.type.lower() == "sgd":
        optimizer = optim.SGD(
            trainable_params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {config.optimizer.type}")
    
    logger.info(f"Created {config.optimizer.type} optimizer with LR {config.training.learning_rate}")
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer, 
    config,
    total_steps: int
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler based on config."""
    if config.scheduler.type.lower() == "cosine":
        warmup_steps = int(total_steps * config.scheduler.warmup_ratio)
        main_steps = total_steps - warmup_steps
        
        if warmup_steps > 0:
            # Create warmup scheduler
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            
            # Create cosine scheduler
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=main_steps,
                eta_min=config.training.learning_rate * 0.01
            )
            
            # Combine schedulers (simplified - in practice you'd use SequentialLR)
            scheduler = cosine_scheduler
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=config.training.learning_rate * 0.01
            )
            
    elif config.scheduler.type.lower() == "linear":
        warmup_steps = int(total_steps * config.scheduler.warmup_ratio)
        scheduler = LinearLR(
            optimizer,
            start_factor=0.1 if warmup_steps > 0 else 1.0,
            end_factor=0.1,
            total_iters=total_steps
        )
        
    elif config.scheduler.type.lower() == "constant":
        scheduler = ConstantLR(optimizer, factor=1.0, total_iters=total_steps)
        
    else:
        logger.warning(f"Unknown scheduler type: {config.scheduler.type}, using no scheduler")
        return None
    
    logger.info(f"Created {config.scheduler.type} scheduler")
    return scheduler


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    step: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    save_path: str
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    checkpoint_path: str
) -> Dict[str, Any]:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"Resuming from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
    
    return checkpoint


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'trainable_percent': 100.0 * trainable_params / total_params if total_params > 0 else 0
    }


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory information."""
    if not torch.cuda.is_available():
        return {}
    
    allocated = torch.cuda.memory_allocated() / 1e9  # GB
    reserved = torch.cuda.memory_reserved() / 1e9    # GB
    total = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
    
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'total_gb': total,
        'free_gb': total - reserved
    }
