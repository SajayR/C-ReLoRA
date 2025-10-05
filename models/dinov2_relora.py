"""Dinov2 backbone wrapped with ReLoRA adapters."""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Iterable

import torch.nn as nn
from transformers import Dinov2Model

from .dinov2_common import DinoV2Classifier, set_module_trainable
from .relora import ReLoRaModel

LOGGER = logging.getLogger(__name__)

_DEFAULT_TARGETS: tuple[str, ...] = ("query", "key", "value", "dense", "fc1", "fc2")


def _normalize_target_modules(raw: Iterable[str] | str | None) -> list[str]:
    if raw is None:
        return list(_DEFAULT_TARGETS)
    if isinstance(raw, str):
        return [raw]
    return list(raw)


def _extract_relora_params(params: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
    relora_cfg = params.get("relora", {}) or {}
    enabled = relora_cfg.get("enabled", True)
    if not enabled:
        return False, {}

    config = {
        "target_modules": _normalize_target_modules(relora_cfg.get("target_modules")),
        "r": int(relora_cfg.get("r", relora_cfg.get("rank", 16))),
        "lora_alpha": int(relora_cfg.get("alpha", relora_cfg.get("lora_alpha", 32))),
        "lora_dropout": float(relora_cfg.get("dropout", relora_cfg.get("lora_dropout", 0.1))),
        "keep_original_weights": bool(relora_cfg.get("keep_original_weights", True)),
        "lora_only": bool(relora_cfg.get("lora_only", False)),
        "trainable_scaling": bool(relora_cfg.get("trainable_scaling", False)),
    }
    return True, config


def build_model(model_name: str, num_classes: int, params: Dict[str, Any]) -> tuple[nn.Module, Dict[str, Any]]:
    dropout = float(params.get("dropout", 0.0))
    freeze_backbone = bool(params.get("freeze_backbone", False))
    gradient_checkpointing = bool(params.get("gradient_checkpointing", False))

    LOGGER.info("Loading Dinov2 backbone '%s'", model_name)
    backbone = Dinov2Model.from_pretrained(model_name)

    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        LOGGER.info("Enabled gradient checkpointing")

    if freeze_backbone:
        set_module_trainable(backbone, False)

    hidden_size = backbone.config.hidden_size

    relora_enabled, relora_config = _extract_relora_params(params)
    relora_summary = None
    if relora_enabled:
        LOGGER.info(
            "Wrapping backbone with ReLoRA r=%d alpha=%d targets=%s",
            relora_config["r"],
            relora_config["lora_alpha"],
            relora_config["target_modules"],
        )
        backbone = ReLoRaModel(backbone, **relora_config)
        relora_summary = copy.deepcopy(relora_config)
    else:
        LOGGER.warning("ReLoRA disabled; returning plain backbone")

    model = DinoV2Classifier(backbone, hidden_size, num_classes, dropout)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    extras = {
        "backbone": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params else 0.0,
        "relora": {
            "enabled": relora_enabled,
            "config": relora_summary,
        },
    }

    return model, extras


__all__ = ["build_model"]
