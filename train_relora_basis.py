"""Training script for DinoV2 ReLoRA fine-tuning on ImageNet."""

import argparse
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from configs.config_manager import ConfigManager, ExperimentConfig
from data.imagenet_dataset import create_imagenet_dataloaders
from models.dinov2_relora_basis import (
    CycleBasisManager,
    DinoV2ReLoRAClassifier,
    ReLoRAConfig,
    assign_relora_cycle_bases,
    merge_relora_layers,
    prune_relora_optimizer_states,
)
from utils.training_utils import (
    create_optimizer,
    create_scheduler,
    save_checkpoint,
    setup_logging,
    set_seed,
)

logger = logging.getLogger(__name__)

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")


class DinoV2ReLoRATrainer:
    """Trainer implementing Algorithm 1 (ReLoRA) for DinoV2."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.device = self._setup_device()

        self.checkpoint_dir = Path(config.checkpointing.save_dir)
        self.log_dir = Path(config.logging.log_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.model: Optional[DinoV2ReLoRAClassifier] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

        self.current_epoch = 0
        self.global_step = 0
        self.relora_step = 0
        self.best_metric = 0.0
        self.relora_active = False
        self.relora_cycle_index = 0

        self._setup_logging()

    def _setup_device(self) -> torch.device:
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)

        logger.info(f"Using device: {device}")
        if device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        return device

    def _setup_logging(self) -> None:
        if self.config.logging.use_wandb:
            run_name = self.config.logging.run_name
            if run_name is None:
                run_name = f"dinov2-relora-{int(time.time())}"

            wandb.init(
                project=self.config.logging.project_name,
                name=run_name,
                config=self.config.to_dict(),
            )
            logger.info(f"Initialized wandb run: {run_name}")

    def setup_data(self) -> None:
        logger.info("Setting up data loaders...")
        self.train_loader, self.val_loader = create_imagenet_dataloaders(
            data_root=self.config.data.dataset_path,
            batch_size=self.config.data.batch_size,
            image_size=self.config.data.image_size,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
        )
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")

    def setup_model(self) -> None:
        logger.info("Initializing model...")
        self.model = DinoV2ReLoRAClassifier(
            model_name=self.config.model.name,
            num_classes=self.config.model.num_classes,
            dropout=self.config.model.dropout,
        ).to(self.device)

        self.relora_config = self._build_relora_config()
        replaced = self.model.apply_relora(self.relora_config)
        logger.info(f"Inserted ReLoRA adapters into {len(replaced)} linear layers")

        self.optimizer = create_optimizer(self.model, self.config)
        adapter_lr = self.config.relora.adapter_learning_rate
        adapter_weight_decay = self.config.relora.adapter_weight_decay
        for group in self.optimizer.param_groups:
            group["lr"] = adapter_lr
            group["weight_decay"] = adapter_weight_decay

        total_steps = len(self.train_loader) * self.config.training.epochs
        self.scheduler = create_scheduler(self.optimizer, self.config, total_steps)

        self.relora_active = True
        self.relora_step = 0
        self.relora_cycle_index = 0
        self._resample_relora_bases()

        logger.info("Model initialization complete")

    def _build_relora_config(self) -> ReLoRAConfig:
        relora_cfg = self.config.relora
        target_modules = relora_cfg.target_modules
        if hasattr(target_modules, "to_container"):
            target_modules = target_modules.to_container()
        elif target_modules is not None and not isinstance(target_modules, list):
            target_modules = list(target_modules)

        return ReLoRAConfig(
            rank=relora_cfg.rank,
            alpha=relora_cfg.alpha,
            dropout=relora_cfg.dropout,
            merge_scale=relora_cfg.merge_scale,
            target_modules=target_modules,
            prune_b_state=relora_cfg.prune_b_state,
        )

    def _get_autocast_dtype(self) -> Optional[torch.dtype]:
        if not self.config.mixed_precision.enabled:
            return None
        dtype_str = str(self.config.mixed_precision.dtype).lower()
        if dtype_str in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype_str in {"fp16", "float16", "half"}:
            return torch.float16
        logger.warning(
            f"Unknown mixed precision dtype '{self.config.mixed_precision.dtype}', disabling"
        )
        return None

    def _should_use_grad_scaler(self, autocast_dtype: Optional[torch.dtype]) -> bool:
        return autocast_dtype == torch.float16 and self.device.type == "cuda"

    def train_epoch(self) -> Dict[str, float]:
        assert self.model is not None and self.optimizer is not None
        self.model.train()

        autocast_dtype = self._get_autocast_dtype()
        use_scaler = self._should_use_grad_scaler(autocast_dtype)
        scaler = GradScaler(enabled=use_scaler)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.epochs}",
        )

        for images, targets in progress_bar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            autocast_ctx = (
                autocast(device_type=self.device.type, dtype=autocast_dtype)
                if autocast_dtype
                else nullcontext()
            )
            with autocast_ctx:
                outputs = self.model(images)
                loss = nn.CrossEntropyLoss()(outputs, targets)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if self.config.training.gradient_clip_norm > 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_norm,
                    )
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                if self.config.training.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip_norm,
                    )
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()

            self.global_step += 1
            if self.relora_active:
                self.relora_step += 1
                if self.relora_step % self.config.relora.merge_frequency == 0:
                    merged = merge_relora_layers(self.model.backbone)
                    prune_relora_optimizer_states(self.model.backbone, self.optimizer)
                    logger.info(
                        f"Merged and reinitialized {merged} ReLoRA layers at step {self.global_step}"
                    )
                    self.relora_cycle_index += 1
                    self._resample_relora_bases()

            avg_loss = total_loss / max(1, total_samples)
            acc = 100.0 * total_correct / max(1, total_samples)
            lr = self.optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(
                {"loss": f"{avg_loss:.4f}", "acc": f"{acc:.2f}%", "lr": lr}
            )

            if self.global_step % self.config.training.logging_steps == 0:
                self._log_metrics(
                    {
                        "train/loss": loss.item(),
                        "train/learning_rate": lr,
                        "train/step": self.global_step,
                    }
                )

        return {
            "train_loss": total_loss / max(1, total_samples),
            "train_accuracy": 100.0 * total_correct / max(1, total_samples),
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        assert self.model is not None
        self.model.eval()

        autocast_dtype = self._get_autocast_dtype()

        total_loss = 0.0
        total_samples = 0
        total_correct = 0
        top5_correct = 0

        progress_bar = tqdm(self.val_loader, desc="Evaluating")

        for images, targets in progress_bar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            autocast_ctx = (
                autocast(device_type=self.device.type, dtype=autocast_dtype)
                if autocast_dtype
                else nullcontext()
            )
            with autocast_ctx:
                outputs = self.model(images)
                loss = nn.CrossEntropyLoss()(outputs, targets)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()

            _, top5 = outputs.topk(5, dim=1)
            top5_correct += top5.eq(targets.unsqueeze(1)).sum().item()

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100.0 * total_correct / max(1, total_samples):.2f}%",
                }
            )

        return {
            "val_loss": total_loss / max(1, total_samples),
            "val_accuracy": 100.0 * total_correct / max(1, total_samples),
            "val_top5_accuracy": 100.0 * top5_correct / max(1, total_samples),
        }

    def _resample_relora_bases(self) -> None:
        if not self.relora_active:
            return
        assert self.model is not None
        seed = int(getattr(self.config, "seed", 0))
        manager = CycleBasisManager(seed, self.relora_cycle_index, self.device)
        assigned = assign_relora_cycle_bases(self.model.backbone, manager)
        logger.info(
            "Assigned orthonormal bases to %d ReLoRA layers for cycle %d",
            assigned,
            self.relora_cycle_index,
        )

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        if self.config.logging.use_wandb:
            wandb.log(metrics, step=self.global_step)

    def _save_checkpoint(
        self, metrics: Dict[str, float], is_best: bool = False
    ) -> None:
        assert self.model is not None and self.optimizer is not None
        save_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        save_checkpoint(
            self.model,
            self.optimizer,
            self.scheduler,
            self.current_epoch,
            self.global_step,
            metrics,
            self.config.to_dict(),
            str(save_path),
        )
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                self.current_epoch,
                self.global_step,
                metrics,
                self.config.to_dict(),
                str(best_path),
            )

    def train(self) -> None:
        logger.info("Starting ReLoRA training...")
        self.setup_data()
        self.setup_model()

        if self.config.evaluation.eval_on_start:
            val_metrics = self.evaluate()
            logger.info(f"Initial evaluation: {val_metrics}")
            self._log_metrics(val_metrics)

        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch()
            val_metrics = self.evaluate()
            metrics = {**train_metrics, **val_metrics, "epoch": epoch}

            current_metric = val_metrics[self.config.checkpointing.metric_for_best]
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric

            self._log_metrics(metrics)
            self._save_checkpoint(metrics, is_best)

            logger.info(
                f"Epoch {epoch + 1}: train_loss={train_metrics['train_loss']:.4f}, "
                f"train_acc={train_metrics['train_accuracy']:.2f}%, "
                f"val_acc={val_metrics['val_accuracy']:.2f}%"
            )

        logger.info("Training completed")
        if self.config.logging.use_wandb:
            wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DinoV2 with ReLoRA")
    parser.add_argument("--config", type=str, default="relora", help="Config name")
    parser.add_argument(
        "--experiment-name", type=str, default=None, help="Experiment name"
    )
    parser.add_argument("--override", nargs="+", help="Config overrides (key=value)")
    args = parser.parse_args()

    overrides: Dict[str, Any] = {}
    if args.override:
        for override in args.override:
            key, value = override.split("=", 1)
            try:
                if value.lower() in {"true", "false"}:
                    value = value.lower() == "true"
                elif "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
            overrides[key] = value

    config_manager = ConfigManager()
    config = config_manager.load_config(args.config, overrides)

    if args.experiment_name:
        config._config.logging.run_name = args.experiment_name

    set_seed(config.seed)
    setup_logging(config.logging.log_dir)

    trainer = DinoV2ReLoRATrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
