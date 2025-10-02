"""ViT-Base with Commutator-LoRA adapter for classification."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Tuple

import torch.nn as nn
from transformers import ViTModel

from .commutator_lora import CommutatorLoRA

LOGGER = logging.getLogger(__name__)


class ViTClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, num_classes: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, num_classes))
        nn.init.xavier_uniform_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.head(cls_token)


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = trainable


def _iter_target_linear_modules(model: nn.Module, target_keys: Iterable[str]) -> List[Tuple[str, nn.Linear]]:
    target_keys = list(target_keys)
    matches: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(name.endswith(key) for key in target_keys):
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
    attr = parts[-1]
    if attr.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(attr)] = module
    else:
        setattr(parent, attr, module)


def _apply_commutator_adapters(
    backbone: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: Iterable[str],
    skew: bool,
    use_gain: bool,
    gain_mode: str,
    gain_heads: int | None,
    init_std_scale: float,
) -> List[str]:
    if rank <= 0:
        return []
    target_modules = list(target_modules)
    matches = _iter_target_linear_modules(backbone, target_modules)
    replaced: List[str] = []
    if not matches:
        LOGGER.warning("No target modules matched for Commutator-LoRA application")
        return replaced
    for name, linear in matches:
        adapter = CommutatorLoRA.wrap_linear(
            base_linear=linear,
            rank_lora=rank,
            alpha=alpha,
            skew=skew,
            use_gain=use_gain,
            gain_mode=gain_mode,
            n_heads=gain_heads,
            init_std_scale=init_std_scale,
            dropout_p=dropout,
        )
        _set_submodule(backbone, name, adapter)
        replaced.append(name)
    return replaced


def build_model(model_name: str, num_classes: int, params: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    dropout = float(params.get("dropout", 0.1))
    gradient_checkpointing = bool(params.get("gradient_checkpointing", False))
    freeze_backbone = bool(params.get("freeze_backbone", True))

    LOGGER.info("Loading ViT backbone '%s'", model_name)
    backbone = ViTModel.from_pretrained(model_name)

    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        LOGGER.info("Enabled gradient checkpointing")

    if freeze_backbone:
        _set_trainable(backbone, False)

    comm_params = params.get("commutator", {}) or {}
    comm_enabled = bool(comm_params.get("enabled", True))
    target_modules = comm_params.get(
        "target_modules",
        ["query", "key", "value", "output.dense", "intermediate.dense"],
    )
    if not isinstance(target_modules, (list, tuple)):
        target_modules = list(target_modules)

    rank = int(comm_params.get("r", comm_params.get("rank", 16)))
    alpha = float(comm_params.get("alpha", 32.0))
    adapter_dropout = float(comm_params.get("dropout", 0.0))
    skew = bool(comm_params.get("skew", False))
    init_std_scale = float(comm_params.get("init_std_scale", 1.0))

    gain_cfg = comm_params.get("gain", {}) or {}
    use_gain = bool(gain_cfg.get("enabled", gain_cfg.get("use_gain", False)))
    gain_mode = str(gain_cfg.get("mode", "per_out")).lower()
    if gain_mode not in {"per_out", "per_head", "none"}:
        raise ValueError(f"Unsupported gain_mode '{gain_mode}'")
    if gain_mode == "none":
        use_gain = False
    gain_heads = None
    if gain_mode == "per_head":
        gain_heads = gain_cfg.get("n_heads")
        if gain_heads is not None:
            gain_heads = int(gain_heads)
        else:
            gain_heads = getattr(backbone.config, "num_attention_heads", None)
        if gain_heads is None:
            raise ValueError("gain_mode='per_head' requires 'n_heads' or backbone config to define heads")

    replaced_modules: List[str] = []
    if comm_enabled and rank > 0:
        replaced_modules = _apply_commutator_adapters(
            backbone,
            rank=rank,
            alpha=alpha,
            dropout=adapter_dropout,
            target_modules=target_modules,
            skew=skew,
            use_gain=use_gain,
            gain_mode=gain_mode,
            gain_heads=gain_heads,
            init_std_scale=init_std_scale,
        )
    else:
        LOGGER.info("Commutator adapters disabled or rank=0; using frozen backbone only")

    model = ViTClassifier(backbone, backbone.config.hidden_size, num_classes, dropout)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    extras: Dict[str, Any] = {
        "backbone": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": 100 * trainable_params / total_params if total_params else 0.0,
        "commutator": {
            "enabled": comm_enabled,
            "rank": rank,
            "alpha": alpha,
            "skew": skew,
            "dropout": adapter_dropout,
            "target_modules": list(target_modules),
            "gain": {
                "enabled": use_gain,
                "mode": gain_mode,
                "n_heads": gain_heads,
            },
            "replaced_modules": replaced_modules,
        },
    }

    return model, extras


__all__ = ["build_model"]
