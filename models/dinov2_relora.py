"""
DINOv2 model with ReLoRA (Reinitialized Low-Rank Adaptation) support.

This module implements Algorithm 1 from the provided ReLoRA specification. It
replaces linear layers with ReLoRA adapters, freezes the base weights, and
exposes helpers to run the merge/reinitialize cycle during training.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import Dinov2Model

logger = logging.getLogger(__name__)


def _power_of_two_blocks(dim: int) -> List[int]:
    """Decompose ``dim`` into a sum of descending powers of two."""

    if dim <= 0:
        raise ValueError("Dimension for orthonormal basis must be positive")

    blocks: List[int] = []
    bit = 1 << (dim.bit_length() - 1)
    remaining = dim
    while remaining:
        if remaining >= bit:
            blocks.append(bit)
            remaining -= bit
        bit >>= 1
    return blocks


def _fwht_rows(x: Tensor) -> Tensor:
    """Fast Walsh-Hadamard transform on the last dimension of ``x``."""

    dim = x.size(-1)
    if dim & (dim - 1) != 0:
        raise ValueError("FWHT requires power-of-two dimension")

    y = x.clone()
    h = 1
    while h < dim:
        new_shape = y.shape[:-1] + (dim // (2 * h), 2, h)
        y = y.reshape(new_shape)
        a = y[..., 0, :].clone()
        b = y[..., 1, :].clone()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        y = y.reshape(*x.shape)
        h <<= 1

    return y / math.sqrt(dim)


def _apply_block_fwht(x: Tensor, block_sizes: List[int]) -> Tensor:
    """Apply block-wise FWHT for non power-of-two dimensions."""

    outputs: List[Tensor] = []
    start = 0
    for size in block_sizes:
        end = start + size
        segment = x[..., start:end]
        if size == 1:
            outputs.append(segment)
        else:
            outputs.append(_fwht_rows(segment))
        start = end
    return torch.cat(outputs, dim=-1)


class SRHTBasisTransform:
    """Structured orthonormal transform using (block) Hadamard, permutation, and sign flips."""

    def __init__(
        self,
        dim: int,
        device: torch.device,
        generator: torch.Generator,
    ) -> None:
        self.dim = dim
        self.block_sizes = _power_of_two_blocks(dim)

        permutation = torch.randperm(dim, generator=generator, device=device)
        inv_permutation = torch.empty_like(permutation)
        inv_permutation[permutation] = torch.arange(dim, device=device)

        signs = torch.randint(0, 2, (dim,), generator=generator, device=device, dtype=torch.int8)
        signs = signs.mul_(2).sub_(1).to(torch.float32)

        self.permutation = permutation
        self.inv_permutation = inv_permutation
        self.signs = signs

    def right_multiply(self, x: Tensor) -> Tensor:
        """Compute ``x @ P`` for row-major ``x``."""

        orig_shape = x.shape
        rows = x.reshape(-1, self.dim).to(torch.float32)
        rows = rows * self.signs
        rows = rows.index_select(-1, self.permutation)
        rows = _apply_block_fwht(rows, self.block_sizes)
        return rows.reshape(orig_shape).to(x.dtype)

    def right_multiply_transpose(self, x: Tensor) -> Tensor:
        """Compute ``x @ Pᵀ`` for row-major ``x``."""

        orig_shape = x.shape
        rows = x.reshape(-1, self.dim).to(torch.float32)
        rows = _apply_block_fwht(rows, self.block_sizes)
        rows = rows.index_select(-1, self.inv_permutation)
        rows = rows * self.signs
        return rows.reshape(orig_shape).to(x.dtype)


class CycleBasisManager:
    """Sample and cache orthonormal bases per width class for a cycle."""

    def __init__(
        self,
        seed: int,
        cycle_index: int,
        device: torch.device,
    ) -> None:
        self.seed = int(seed)
        self.cycle_index = int(cycle_index)
        self.device = device
        self._cache: Dict[Tuple[str, int], SRHTBasisTransform] = {}

    def _make_generator(self, key: Tuple[str, int]) -> torch.Generator:
        dim_key = key[1]
        composite = (
            (self.seed & 0xFFFFFFFF)
            + 0x9E3779B9 * self.cycle_index
            + 0x7F4A7C15 * dim_key
        ) & 0xFFFFFFFFFFFFFFFF
        device_str = "cuda" if self.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=device_str)
        generator.manual_seed(int(composite))
        return generator

    def get(self, kind: str, dim: int) -> SRHTBasisTransform:
        cache_key = (kind, dim)
        transform = self._cache.get(cache_key)
        if transform is None:
            generator = self._make_generator(cache_key)
            transform = SRHTBasisTransform(dim, self.device, generator)
            self._cache[cache_key] = transform
        return transform

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

        self.input_basis: Optional[SRHTBasisTransform] = None
        self.output_basis: Optional[SRHTBasisTransform] = None

        self.reset_adapters()

    def assign_cycle_basis(
        self,
        input_basis: Optional[SRHTBasisTransform],
        output_basis: Optional[SRHTBasisTransform],
    ) -> None:
        self.input_basis = input_basis
        self.output_basis = output_basis

    def forward(self, x: Tensor) -> Tensor:
        residual = self.base(x)

        adapter_input = x
        if self.input_basis is not None:
            adapter_input = self.input_basis.right_multiply_transpose(adapter_input)
        if self.dropout is not None and self.training:
            adapter_input = self.dropout(adapter_input)

        down = F.linear(adapter_input, self.A)
        up = F.linear(down, self.B)
        if self.output_basis is not None:
            up = self.output_basis.right_multiply(up)
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
        delta_w = (self.B.float() @ self.A.float())
        if self.input_basis is not None:
            delta_w = self.input_basis.right_multiply(delta_w)
        if self.output_basis is not None:
            delta_w = self.output_basis.right_multiply(delta_w.T).T
        delta_w = delta_w * float(self.scale)
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


def assign_relora_cycle_bases(
    module: nn.Module,
    manager: CycleBasisManager,
) -> int:
    assigned = 0
    for submodule in module.modules():
        if isinstance(submodule, ReLoRALinear):
            input_basis = manager.get("in", submodule.base.in_features)
            output_basis = manager.get("out", submodule.base.out_features)
            submodule.assign_cycle_basis(input_basis, output_basis)
            assigned += 1
    return assigned


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
