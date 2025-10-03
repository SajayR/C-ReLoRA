"""DinoV2 with bi-sided BracketAdapter."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Tuple

import torch.nn as nn
from transformers import Dinov2Model

from .bracket_adapter import BracketAdapter

LOGGER = logging.getLogger(__name__)


class DinoV2BracketClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, pixel_values):  # type: ignore[override]
        features = self.backbone(pixel_values=pixel_values)
        cls_token = features.last_hidden_state[:, 0]
        return self.head(cls_token)


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


def _iter_target_linear_modules(model: nn.Module, target_suffixes: Iterable[str]) -> List[Tuple[str, nn.Linear]]:
    suffixes = list(target_suffixes)
    matches: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(name.endswith(suffix) for suffix in suffixes):
            matches.append((name, module))
    return matches


def _set_submodule(root: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(last)] = module
    else:
        setattr(parent, last, module)


def _apply_bracket_adapters(
    backbone: nn.Module,
    rank_out: int,
    rank_in: int,
    alpha: float,
    beta: float,
    lora_alpha: float | None,
    scaling: float | None,
    dropout: float,
    mode: str,
    target_modules: Iterable[str],
) -> List[Tuple[str, BracketAdapter]]:
    if rank_out <= 0 or rank_in <= 0:
        raise ValueError("Both rank_out and rank_in must be positive for BracketAdapter")

    matches = _iter_target_linear_modules(backbone, target_modules)
    wrapped: List[Tuple[str, BracketAdapter]] = []
    if not matches:
        LOGGER.warning("No modules matched for BracketAdapter application")
        return wrapped

    adapter_kwargs: Dict[str, Any] = {
        "alpha": alpha,
        "beta": beta,
        "mode": mode,
        "dropout_p": dropout,
    }
    if scaling is not None:
        adapter_kwargs["scaling"] = float(scaling)
    elif lora_alpha is not None:
        adapter_kwargs["lora_alpha"] = float(lora_alpha)

    for name, linear in matches:
        adapter = BracketAdapter.wrap_linear(
            base_linear=linear,
            rank_out=rank_out,
            rank_in=rank_in,
            **adapter_kwargs,
        )
        _set_submodule(backbone, name, adapter)
        wrapped.append((name, adapter))
    return wrapped


def build_model(model_name: str, num_classes: int, params: Dict[str, Any]):
    dropout = float(params.get("dropout", 0.1))
    freeze_backbone = bool(params.get("freeze_backbone", False))
    gradient_checkpointing = bool(params.get("gradient_checkpointing", False))

    LOGGER.info("Loading DinoV2 backbone '%s'", model_name)
    backbone = Dinov2Model.from_pretrained(model_name)

    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        LOGGER.info("Enabled gradient checkpointing")

    if freeze_backbone:
        _set_trainable(backbone, False)

    bracket_cfg = params.get("bracket", {}) or {}
    bracket_enabled = bool(bracket_cfg.get("enabled", True))
    target_modules = bracket_cfg.get("target_modules", ["query", "key", "value", "dense", "fc1", "fc2"])
    if not isinstance(target_modules, (list, tuple)):
        raise TypeError("bracket.target_modules must be a sequence of suffix strings")

    total_rank_cfg = bracket_cfg.get("r", bracket_cfg.get("rank"))
    rank_out_cfg = bracket_cfg.get("r_out", bracket_cfg.get("rank_out"))
    rank_in_cfg = bracket_cfg.get("r_in", bracket_cfg.get("rank_in"))

    if total_rank_cfg is not None and (rank_out_cfg is not None or rank_in_cfg is not None):
        raise ValueError(
            "Provide either 'bracket.r'/'bracket.rank' or explicit 'r_out'/'r_in', not both"
        )

    if total_rank_cfg is not None:
        total_rank = int(total_rank_cfg)
        if total_rank < 2:
            raise ValueError("BracketAdapter requires total rank >= 2 to keep both sides active")
        rank_out = int(math.ceil(total_rank / 2))
        rank_in = total_rank - rank_out
    else:
        default_rank_out = 8
        rank_out = int(rank_out_cfg) if rank_out_cfg is not None else default_rank_out
        rank_in = int(rank_in_cfg) if rank_in_cfg is not None else int(rank_out)
        total_rank = rank_out + rank_in
    alpha = float(bracket_cfg.get("alpha", 1.0))
    beta = float(bracket_cfg.get("beta", 1.0))
    lora_alpha = bracket_cfg.get("lora_alpha", bracket_cfg.get("lora_scale"))
    lora_alpha = float(lora_alpha) if lora_alpha is not None else None
    scaling = bracket_cfg.get("scaling")
    scaling = float(scaling) if scaling is not None else None
    dropout_adapters = float(bracket_cfg.get("dropout", 0.0))
    mode = str(bracket_cfg.get("mode", "commutator")).lower()

    replaced_pairs: List[Tuple[str, BracketAdapter]] = []
    if bracket_enabled:
        replaced_pairs = _apply_bracket_adapters(
            backbone,
            rank_out=rank_out,
            rank_in=rank_in,
            alpha=alpha,
            beta=beta,
            lora_alpha=lora_alpha,
            scaling=scaling,
            dropout=dropout_adapters,
            mode=mode,
            target_modules=target_modules,
        )
    else:
        LOGGER.info("Bracket adapters disabled; using frozen backbone")

    model = DinoV2BracketClassifier(backbone, backbone.config.hidden_size, num_classes, dropout)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter_param_total = sum(adapter.num_adapter_parameters() for _, adapter in replaced_pairs)

    extras: Dict[str, Any] = {
        "backbone": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params else 0.0,
        "bracket": {
            "enabled": bracket_enabled,
            "rank_out": rank_out,
            "rank_in": rank_in,
            "total_rank": total_rank,
            "alpha": alpha,
            "beta": beta,
            "lora_alpha": lora_alpha,
            "scaling": scaling,
            "dropout": dropout_adapters,
            "mode": mode,
            "target_modules": list(target_modules),
            "replaced_modules": [name for name, _ in replaced_pairs],
            "adapter_parameters": adapter_param_total,
        },
    }

    return model, extras


__all__ = ["build_model"]
