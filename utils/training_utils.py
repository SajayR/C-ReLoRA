"""Lightweight training helpers."""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR

LOGGER = logging.getLogger(__name__)


def _resolve_steps(value: Any, total_steps: int) -> int:
    if value is None:
        return 0
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:  # noqa: PERF203
        raise ValueError(f"Scheduler step value '{value}' is not numeric") from exc
    if numeric < 0:
        raise ValueError(f"Scheduler step value must be non-negative, got {numeric}")
    if numeric == 0:
        return 0
    if numeric < 1.0:
        steps = int(total_steps * numeric)
        return max(1, steps)
    return int(round(numeric))


def setup_logging(output_dir: Path, level: int = logging.INFO) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler()]
    log_file = output_dir / "train.log"
    handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    LOGGER.info("Seed set to %d", seed)


def create_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    name = cfg.get("type", "adamw").lower()
    lr = float(cfg.get("lr", cfg.get("learning_rate", 3e-4)))
    weight_decay = float(cfg.get("weight_decay", 0.0))

    if name == "adamw":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))
        eps = float(cfg.get("eps", 1e-8))
        optimizer = AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    elif name == "adam":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))
        eps = float(cfg.get("eps", 1e-8))
        optimizer = Adam(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    elif name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        optimizer = SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")

    LOGGER.info("Optimizer: %s lr=%.2e weight_decay=%.2e params=%d", name, lr, weight_decay, sum(p.numel() for p in params))
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any], total_steps: int) -> Optional[LambdaLR]:
    total_steps = int(total_steps)
    if total_steps <= 0:
        return None

    cfg = cfg or {}
    name = cfg.get("type", "cosine").lower()
    if name in {"none", "constant"}:
        return None

    min_lr_ratio = float(cfg.get("min_lr_mult", cfg.get("min_lr_ratio", 0.01)))
    min_lr_ratio = min(max(min_lr_ratio, 0.0), 1.0)

    if name == "cosine_restarts":
        first_warmup_steps = _resolve_steps(
            cfg.get("first_warmup_steps", cfg.get("warmup", cfg.get("warmup_epochs", 0))),
            total_steps,
        )
        restart_every = _resolve_steps(cfg.get("restart_every"), total_steps)
        if restart_every <= 0:
            raise ValueError("cosine_restarts scheduler requires 'restart_every' > 0")
        restart_warmup_steps = _resolve_steps(
            cfg.get("restart_warmup_steps", max(1, restart_every // 10)), restart_every
        )
        total_after_warmup = max(1, total_steps - first_warmup_steps)

        def envelope(step: int) -> float:
            if first_warmup_steps and step < first_warmup_steps:
                return step / max(1, first_warmup_steps)
            progress = (step - first_warmup_steps) / float(total_after_warmup)
            progress = min(max(progress, 0.0), 1.0)
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(progress * math.pi))

        def lr_lambda(step: int) -> float:
            clamped_step = min(step, total_steps)
            cycle_index = clamped_step // restart_every
            cycle_step = clamped_step % restart_every

            if cycle_index == 0:
                if first_warmup_steps and clamped_step < first_warmup_steps:
                    return max(envelope(clamped_step), 1e-6)
                return envelope(clamped_step)

            if cycle_step < restart_warmup_steps:
                cycle_start = cycle_index * restart_every
                ramp_end_step = min(cycle_start + restart_warmup_steps, total_steps)
                warmup_target = envelope(ramp_end_step)
                ramp_fraction = cycle_step / max(1, restart_warmup_steps)
                return warmup_target * ramp_fraction

            return envelope(clamped_step)

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        LOGGER.info(
            "Scheduler: %s total=%d first_warmup=%d restart_every=%d restart_warmup=%d min_lr_ratio=%.4f",
            name,
            total_steps,
            first_warmup_steps,
            restart_every,
            restart_warmup_steps,
            min_lr_ratio,
        )
        return scheduler

    warmup_steps = _resolve_steps(cfg.get("warmup", cfg.get("warmup_epochs", 0)), total_steps)

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step / float(warmup_steps), 1e-6)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if name == "linear":
            return (1.0 - progress) * (1.0 - min_lr_ratio) + min_lr_ratio
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(progress * math.pi))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    LOGGER.info(
        "Scheduler: %s warmup=%d steps total=%d min_lr_ratio=%.4f",
        name,
        warmup_steps,
        total_steps,
        min_lr_ratio,
    )
    return scheduler


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    LOGGER.info("Saved checkpoint to %s", path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    LOGGER.info("Loaded checkpoint from %s", path)
    return checkpoint


def grad_norm(parameters, norm_type: float = 2.0) -> float:
    norms = []
    for p in parameters:
        if p.grad is not None:
            norms.append(p.grad.detach().data.norm(norm_type))
    if not norms:
        return 0.0
    total = torch.norm(torch.stack(norms), norm_type)
    return float(total.item())


def param_norm(parameters, norm_type: float = 2.0) -> float:
    values = [p.detach().data.norm(norm_type) for p in parameters if p.requires_grad]
    if not values:
        return 0.0
    total = torch.norm(torch.stack(values), norm_type)
    return float(total.item())


__all__ = [
    "setup_logging",
    "set_seed",
    "create_optimizer",
    "create_scheduler",
    "save_checkpoint",
    "load_checkpoint",
    "grad_norm",
    "param_norm",
]
