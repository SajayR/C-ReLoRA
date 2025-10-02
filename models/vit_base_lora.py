"""ViT-Base with LoRA adapter for classification."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import ViTModel

LOGGER = logging.getLogger(__name__)


class ViTClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_classes: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.head(cls_token)


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


def _build_lora_config(params: Dict[str, Any]) -> LoraConfig:
    target_modules: Iterable[str] = params.get(
        "target_modules",
        ["query", "key", "value", "output.dense", "intermediate.dense"],
    )
    if not isinstance(target_modules, (list, tuple)):
        target_modules = list(target_modules)
    return LoraConfig(
        r=int(params.get("r", params.get("rank", 16))),
        lora_alpha=int(params.get("alpha", params.get("lora_alpha", 32))),
        target_modules=list(target_modules),
        lora_dropout=float(params.get("dropout", params.get("lora_dropout", 0.1))),
        bias=params.get("bias", "none"),
        task_type=TaskType.FEATURE_EXTRACTION,
    )


def build_model(model_name: str, num_classes: int, params: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    dropout = float(params.get("dropout", 0.1))
    freeze_backbone = bool(params.get("freeze_backbone", False))
    gradient_checkpointing = bool(params.get("gradient_checkpointing", False))

    LOGGER.info("Loading ViT backbone '%s'", model_name)
    backbone = ViTModel.from_pretrained(model_name)

    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        LOGGER.info("Enabled gradient checkpointing")

    if freeze_backbone:
        _set_trainable(backbone, False)

    lora_params = params.get("lora", {}) or {}
    lora_enabled = bool(lora_params.get("enabled", True))
    lora_config = None

    if lora_enabled:
        lora_config = _build_lora_config(lora_params)
        backbone = get_peft_model(backbone, lora_config)
        backbone.print_trainable_parameters()

    model = ViTClassifier(backbone, backbone.config.hidden_size, num_classes, dropout)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    extras = {
        "backbone": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params else 0.0,
        "lora": {
            "enabled": lora_enabled,
            "config": lora_config.to_dict() if lora_config else None,
        },
    }

    return model, extras


__all__ = ["build_model"]
