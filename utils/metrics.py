"""Minimal metrics helpers."""

from __future__ import annotations

from typing import Iterable, List

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return 100.0 * correct / targets.size(0)


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    if k <= 0:
        return 0.0
    _, topk = logits.topk(min(k, logits.size(1)), dim=1)
    match = topk.eq(targets.view(-1, 1))
    correct = match.any(dim=1).float().sum().item()
    return 100.0 * correct / targets.size(0)


def stack_logits(batches: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([b.detach().cpu() for b in batches], dim=0)


def stack_targets(batches: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([b.detach().cpu() for b in batches], dim=0)


__all__ = ["accuracy", "top_k_accuracy", "stack_logits", "stack_targets"]
