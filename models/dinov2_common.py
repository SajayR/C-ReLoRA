"""Shared Dinov2 classifier utilities."""

from __future__ import annotations

import torch
import torch.nn as nn


class DinoV2Classifier(nn.Module):
    """Thin classification head on top of a Dinov2 backbone."""

    def __init__(self, backbone: nn.Module, hidden_size: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pixel_values=pixel_values)
        cls_token = features.last_hidden_state[:, 0]
        return self.head(cls_token)

    def relora_merge_and_reinit(self) -> None:
        """Delegate ReLoRA merge/reset to the backbone when available."""
        merge_fn = getattr(self.backbone, "merge_and_reinit", None)
        if callable(merge_fn):
            merge_fn()


def set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


__all__ = ["DinoV2Classifier", "set_module_trainable"]
