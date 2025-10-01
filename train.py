"""
Training script for DinoV2 LoRA fine-tuning on ImageNet.
"""

import os
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast
import wandb
from tqdm import tqdm

# Local imports
from data.imagenet_dataset import create_imagenet_dataloaders
from models.dinov2_lora import create_dinov2_lora_model, get_model_size_info
from configs.config_manager import ConfigManager, ExperimentConfig
from utils.metrics import accuracy, top_k_accuracy
from utils.training_utils import (
    setup_logging, 
    save_checkpoint, 
    load_checkpoint,
    create_optimizer,
    create_scheduler,
    set_seed
)

logger = logging.getLogger(__name__)
import os
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

class DinoV2LoRATrainer:
    """Trainer class for DinoV2 LoRA fine-tuning."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = self._setup_device()
        
        # Setup directories
        self.checkpoint_dir = Path(config.checkpointing.save_dir)
        self.log_dir = Path(config.logging.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        
        # Setup logging
        self._setup_logging()
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        logger.info(f"Using device: {device}")
        if device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
        
        return device
    
    def _setup_logging(self):
        """Setup experiment logging."""
        if self.config.logging.use_wandb:
            run_name = self.config.logging.run_name
            if run_name is None:
                run_name = f"dinov2-lora-{int(time.time())}"
            
            wandb.init(
                project=self.config.logging.project_name,
                name=run_name,
                config=self.config.to_dict()
            )
            logger.info(f"Initialized wandb run: {run_name}")
    
    def setup_data(self):
        """Setup data loaders."""
        logger.info("Setting up data loaders...")
        
        self.train_loader, self.val_loader = create_imagenet_dataloaders(
            data_root=self.config.data.dataset_path,
            batch_size=self.config.data.batch_size,
            image_size=self.config.data.image_size,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
    
    def setup_model(self):
        """Setup model, optimizer, and scheduler."""
        logger.info("Setting up model...")
        
        # Create model
        # Convert target_modules to regular list if it's a ListConfig
        target_modules = self.config.lora.target_modules
        if hasattr(target_modules, 'to_container'):
            target_modules = target_modules.to_container()
        elif not isinstance(target_modules, list):
            target_modules = list(target_modules)
            
        self.model = create_dinov2_lora_model(
            model_name=self.config.model.name,
            num_classes=self.config.model.num_classes,
            lora_r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=target_modules
        )
        
        self.model = self.model.to(self.device)
        
        # Print model info
        model_info = get_model_size_info(self.model)
        logger.info(f"Model parameters: {model_info}")
        
        # Create optimizer
        self.optimizer = create_optimizer(self.model, self.config)
        
        # Create scheduler
        total_steps = len(self.train_loader) * self.config.training.epochs
        self.scheduler = create_scheduler(self.optimizer, self.config, total_steps)
        
        logger.info("Model setup complete")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {self.current_epoch+1}/{self.config.training.epochs}"
        )
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass
            if self.config.mixed_precision.enabled:
                with autocast(dtype=torch.bfloat16, device_type="cuda"):
                    outputs = self.model(images)
                    loss = nn.CrossEntropyLoss()(outputs, targets)
            else:
                outputs = self.model(images)
                loss = nn.CrossEntropyLoss()(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(targets).sum().item()
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct_predictions / total_samples:.2f}%',
                'lr': f'{current_lr:.2e}'
            })
            
            # Logging
            if self.global_step % self.config.training.logging_steps == 0:
                self._log_metrics({
                    'train/loss': loss.item(),
                    'train/learning_rate': current_lr,
                    'train/step': self.global_step
                })
            
            self.global_step += 1
        
        # Epoch metrics
        avg_loss = total_loss / total_samples
        accuracy = 100.0 * correct_predictions / total_samples
        
        return {
            'train_loss': avg_loss,
            'train_accuracy': accuracy
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        top5_correct = 0
        
        progress_bar = tqdm(self.val_loader, desc="Evaluating")
        
        for images, targets in progress_bar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            if self.config.mixed_precision.enabled:
                with autocast(dtype=torch.bfloat16, device_type="cuda"):
                    outputs = self.model(images)
                    loss = nn.CrossEntropyLoss()(outputs, targets)
            else:
                outputs = self.model(images)
                loss = nn.CrossEntropyLoss()(outputs, targets)
            
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Top-1 accuracy
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(targets).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_correct += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct_predictions / total_samples:.2f}%'
            })
        
        avg_loss = total_loss / total_samples
        top1_accuracy = 100.0 * correct_predictions / total_samples
        top5_accuracy = 100.0 * top5_correct / total_samples
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': top1_accuracy,
            'val_top5_accuracy': top5_accuracy
        }
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to wandb."""
        if self.config.logging.use_wandb:
            wandb.log(metrics, step=self.global_step)
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config.to_dict()
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint with {self.config.checkpointing.metric_for_best}: {metrics[self.config.checkpointing.metric_for_best]:.4f}")
        
        # Save LoRA weights separately
        lora_path = self.checkpoint_dir / f"lora_weights_epoch_{self.current_epoch}"
        self.model.save_lora_weights(lora_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Setup
        self.setup_data()
        self.setup_model()
        
        # Initial evaluation
        if self.config.evaluation.eval_on_start:
            val_metrics = self.evaluate()
            self._log_metrics(val_metrics)
            logger.info(f"Initial evaluation: {val_metrics}")
        
        # Training loop
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Evaluate
            if (epoch + 1) % (self.config.training.eval_steps // len(self.train_loader) + 1) == 0:
                val_metrics = self.evaluate()
                
                # Combine metrics
                all_metrics = {**train_metrics, **val_metrics}
                all_metrics['epoch'] = epoch
                
                # Log metrics
                self._log_metrics(all_metrics)
                
                # Check if best model
                current_metric = val_metrics[self.config.checkpointing.metric_for_best]
                is_best = current_metric > self.best_metric
                if is_best:
                    self.best_metric = current_metric
                
                # Save checkpoint
                self.save_checkpoint(all_metrics, is_best)
                
                logger.info(f"Epoch {epoch+1} - Train: {train_metrics}, Val: {val_metrics}")
        
        logger.info("Training completed!")
        
        if self.config.logging.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train DinoV2 with LoRA on ImageNet")
    parser.add_argument("--config", type=str, default="default", help="Config name")
    parser.add_argument("--experiment-name", type=str, help="Experiment name")
    parser.add_argument("--override", nargs="+", help="Config overrides (key=value)")
    
    args = parser.parse_args()
    
    # Parse overrides
    overrides = {}
    if args.override:
        for override in args.override:
            key, value = override.split("=", 1)
            # Try to parse as number or boolean
            try:
                if value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                elif "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string
            overrides[key] = value
    
    # Load config
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config, overrides)
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Setup logging
    setup_logging(config.logging.log_dir)
    
    # Create trainer and start training
    trainer = DinoV2LoRATrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
