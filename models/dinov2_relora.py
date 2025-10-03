"""
DINOv2 model with ReLoRA (Reinitialized Low-Rank Adaptation) support.

This module implements Algorithm 1 from the provided ReLoRA specification. It
replaces linear layers with ReLoRA adapters, freezes the base weights, and
exposes helpers to run the merge/reinitialize cycle during training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import Dinov2Model

logger = logging.getLogger(__name__)


@dataclass
class ReLoRAConfig:
    """Configuration options for ReLoRA adapters."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    merge_scale: Optional[float] = None  # If None, defaults to alpha / rank
    target_modules: Optional[List[str]] = None
    prune_b_state: bool = True

    def resolve_scale(self) -> float:
        if self.merge_scale is not None:
            return float(self.merge_scale)
        if self.rank <= 0:
            raise ValueError("ReLoRA rank must be positive before computing scale")
        return float(self.alpha) / float(self.rank)


class ReLoRALinear(nn.Module):
    """Wrapper around ``nn.Linear`` that adds ReLoRA adapters."""

    def __init__(self, linear: nn.Linear, config: ReLoRAConfig):
        super().__init__()

        # Keep the original Linear module so its Parameter objects (and optimizer state)
        # remain intact. We only freeze the weight as part of the ReLoRA algorithm.
        self.base = linear
        self.base.weight.requires_grad_(False)
        # Bias stays trainable by default to mirror typical LoRA practice.

        self.config = config
        self.rank = int(config.rank)
        self.scale = config.resolve_scale()
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        factory_kwargs = {
            "device": self.base.weight.device,
            "dtype": self.base.weight.dtype,
        }
        self.A = nn.Parameter(torch.empty(self.rank, self.base.in_features, **factory_kwargs))
        self.B = nn.Parameter(torch.empty(self.base.out_features, self.rank, **factory_kwargs))

        self.reset_adapters()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.base(x)

        adapter_input = x
        if self.dropout is not None and self.training:
            adapter_input = self.dropout(adapter_input)

        down = F.linear(adapter_input, self.A)
        up = F.linear(down, self.B)
        return residual + self.scale * up

    @torch.no_grad()
    def merge_into_base(self) -> None:
        assert self.base.weight.shape == (
            self.B.size(0),
            self.A.size(1),
        ), (
            f"Shape mismatch: base {self.base.weight.shape} vs delta "
            f"{(self.B.size(0), self.A.size(1))}"
        )
        delta_w = (self.B.float() @ self.A.float()) * float(self.scale)
        self.base.weight.data.add_(delta_w.to(self.base.weight.dtype))

    @torch.no_grad()
    def reset_adapters(self) -> None:
        nn.init.kaiming_uniform_(self.A, a=0.0, nonlinearity="linear")
        nn.init.zeros_(self.B)

    @torch.no_grad()
    def merge_and_reinit(self) -> None:
        self.merge_into_base()
        self.reset_adapters()

    def prune_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        state = optimizer.state
        for param, prune_bias in ((self.A, False), (self.B, self.config.prune_b_state)):
            if not prune_bias and param is self.B:
                continue
            if param in state:
                for key, value in state[param].items():
                    if isinstance(value, Tensor):
                        value.zero_()

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        return (self.A, self.B)

    def base_parameter(self) -> nn.Parameter:
        return self.base.weight


def _module_matches(name: str, targets: Optional[List[str]]) -> bool:
    if not targets:
        return True
    return any(target in name for target in targets)


def replace_linear_with_relora(
    module: nn.Module,
    config: ReLoRAConfig,
    prefix: str = "",
    replaced: Optional[List[str]] = None,
) -> List[str]:
    """Recursively replace ``nn.Linear`` layers with ``ReLoRALinear`` wrappers."""

    if replaced is None:
        replaced = []

    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear) and _module_matches(child_prefix, config.target_modules):
            relora_child = ReLoRALinear(child, config)
            setattr(module, name, relora_child)
            replaced.append(child_prefix)
            logger.debug(f"Replaced linear layer {child_prefix} with ReLoRA")
        else:
            replace_linear_with_relora(child, config, child_prefix, replaced)

    return replaced


class DinoV2ReLoRAClassifier(nn.Module):
    """DINOv2 classifier with ReLoRA adapters."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        num_classes: int = 1000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        self.backbone = Dinov2Model.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_classes),
        )
        nn.init.xavier_uniform_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, pixel_values: Tensor) -> Tensor:
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        return self.classifier(cls_token)

    def apply_relora(self, config: ReLoRAConfig) -> List[str]:
        return replace_linear_with_relora(self.backbone, config)

    def iter_relora_layers(self) -> Iterable[ReLoRALinear]:
        for module in self.modules():
            if isinstance(module, ReLoRALinear):
                yield module

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        for layer in self.iter_relora_layers():
            yield from layer.adapter_parameters()

    def base_parameters(self) -> Iterable[nn.Parameter]:
        for layer in self.iter_relora_layers():
            yield layer.base_parameter()


def collect_adapter_parameters(module: nn.Module) -> List[nn.Parameter]:
    adapters: List[nn.Parameter] = []
    for submodule in module.modules():
        if isinstance(submodule, ReLoRALinear):
            adapters.extend(list(submodule.adapter_parameters()))
    return adapters


def merge_relora_layers(module: nn.Module) -> int:
    count = 0
    for submodule in module.modules():
        if isinstance(submodule, ReLoRALinear):
            submodule.merge_and_reinit()
            count += 1
    return count


def prune_relora_optimizer_states(
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    for submodule in module.modules():
        if isinstance(submodule, ReLoRALinear):
            submodule.prune_optimizer_state(optimizer)
