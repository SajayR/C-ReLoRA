"""Bi-sided BracketAdapter for parameter-efficient fine-tuning."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

SignMode = Literal["commutator", "anticommutator"]


class BracketAdapter(nn.Module):
    """Wrap a frozen linear layer with simultaneous left/right low-rank updates."""

    version: int = 1

    def __init__(
        self,
        base_linear: nn.Linear,
        rank_out: int,
        rank_in: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        lora_alpha: float | None = None,
        scaling: float | None = None,
        mode: SignMode = "commutator",
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be an instance of nn.Linear")
        if rank_out <= 0:
            raise ValueError(
                "rank_out must be a positive integer; both sides must remain active"
            )
        if rank_in <= 0:
            raise ValueError(
                "rank_in must be a positive integer; both sides must remain active"
            )
        if mode not in ("commutator", "anticommutator"):
            raise ValueError("mode must be either 'commutator' or 'anticommutator'")

        self.base_linear = base_linear
        self.in_features = int(base_linear.in_features)
        self.out_features = int(base_linear.out_features)
        self.rank_out = int(rank_out)
        self.rank_in = int(rank_in)
        self.adapter_dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None

        self.base_linear.weight.requires_grad_(False)
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad_(False)

        if scaling is not None and lora_alpha is not None:
            raise ValueError("Provide at most one of scaling or lora_alpha")
        if scaling is None:
            effective_rank = self.rank_out + self.rank_in
            if effective_rank <= 0:
                raise ValueError("Effective rank must be positive")
            lora_alpha_val = lora_alpha if lora_alpha is not None else 1.0
            scaling = float(lora_alpha_val) / float(effective_rank)
        self.register_buffer(
            "scaling",
            torch.tensor(float(scaling), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "alpha", torch.tensor(float(alpha), dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "beta", torch.tensor(float(beta), dtype=torch.float32), persistent=True
        )
        sign = -1.0 if mode == "commutator" else 1.0
        self.register_buffer(
            "sign", torch.tensor(sign, dtype=torch.float32), persistent=True
        )
        self.mode: SignMode = mode

        self.U_out = nn.Parameter(torch.empty(self.out_features, self.rank_out))
        self.V_out = nn.Parameter(torch.empty(self.out_features, self.rank_out))
        self.U_in = nn.Parameter(torch.empty(self.in_features, self.rank_in))
        self.V_in = nn.Parameter(torch.empty(self.in_features, self.rank_in))

        self.register_buffer(
            "adapter_version",
            torch.tensor(self.version, dtype=torch.int32),
            persistent=True,
        )

        self._enabled: bool = True
        self._merged: bool = False
        self._cached_delta: Tensor | None = None

        self._init_parameters()

        self._register_state_dict_hook(self._strip_base_from_state_dict)
        self._register_load_state_dict_pre_hook(self._strip_base_from_load)

    def _init_parameters(self):
        nn.init.kaiming_uniform_(self.U_out, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.U_in, a=math.sqrt(5))
        nn.init.zeros_(self.V_out)
        nn.init.zeros_(self.V_in)
        with torch.no_grad():
            self.U_out.mul_(1.0 / math.sqrt(self.rank_out))
            self.U_in.mul_(1.0 / math.sqrt(self.rank_in))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank_out={self.rank_out}, rank_in={self.rank_in}, mode='{self.mode}', "
            f"dropout={self.adapter_dropout.p if self.adapter_dropout else 0.0}"
        )

    @property
    def scaling_value(self) -> float:
        return float(self.scaling.item())

    def forward(self, x: Tensor) -> Tensor:
        if self._merged or not self._enabled:
            return self.base_linear(x)

        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        weight = self.base_linear.weight
        compute_dtype = torch.promote_types(weight.dtype, x_flat.dtype)
        x_compute = x_flat.to(compute_dtype)
        weight_compute = weight.to(compute_dtype)

        y = F.linear(x_compute, weight_compute, bias=None)

        V_out = self.V_out.to(compute_dtype)
        U_out = self.U_out.to(compute_dtype)
        V_in = self.V_in.to(compute_dtype)
        U_in = self.U_in.to(compute_dtype)

        t_out = x_compute.new_empty(x_compute.shape[0], self.rank_out)
        t_out = torch.matmul(y, V_out)
        if self.adapter_dropout is not None:
            t_out = self.adapter_dropout(t_out)
        y_out = torch.matmul(t_out, U_out.t())

        t_in = torch.matmul(x_compute, V_in)
        if self.adapter_dropout is not None:
            t_in = self.adapter_dropout(t_in)
        z = torch.matmul(t_in, U_in.t())
        y_in = F.linear(z, weight_compute, bias=None)

        delta = self.scaling.to(compute_dtype) * (
            self.alpha.to(compute_dtype) * y_out
            + self.sign.to(compute_dtype) * self.beta.to(compute_dtype) * y_in
        )
        y_hat = y + delta

        if self.base_linear.bias is not None:
            y_hat = y_hat + self.base_linear.bias.to(compute_dtype)

        y_hat = y_hat.reshape(*orig_shape[:-1], self.out_features)
        return y_hat.to(x.dtype)

    def enable_adapter(self) -> None:
        self._enabled = True

    def disable_adapter(self) -> None:
        self._enabled = False

    def merge_into_base(self) -> None:
        if self._merged:
            return
        delta = self.delta_weight().to(self.base_linear.weight.dtype)
        self.base_linear.weight.data.add_(delta)
        self._cached_delta = delta
        self._merged = True
        self.disable_adapter()

    def unmerge_from_base(self) -> None:
        if not self._merged:
            return
        if self._cached_delta is None:
            raise RuntimeError("Cached delta weight missing; cannot unmerge")
        self.base_linear.weight.data.sub_(self._cached_delta)
        self._merged = False
        self.enable_adapter()
        self._cached_delta = None

    def delta_weight(self) -> Tensor:
        weight = self.base_linear.weight.detach()
        compute_dtype = weight.dtype
        U_out = self.U_out.to(compute_dtype)
        V_out = self.V_out.to(compute_dtype)
        U_in = self.U_in.to(compute_dtype)
        V_in = self.V_in.to(compute_dtype)

        left = torch.matmul(U_out, torch.matmul(V_out.t(), weight))
        right = torch.matmul(weight, torch.matmul(U_in, V_in.t()))
        delta = self.scaling.to(compute_dtype) * (
            self.alpha.to(compute_dtype) * left
            + self.sign.to(compute_dtype) * self.beta.to(compute_dtype) * right
        )
        return delta

    def effective_weight(self) -> Tensor:
        return self.base_linear.weight + self.delta_weight()

    def num_adapter_parameters(self, include_buffers: bool = True) -> int:
        param_count = (
            self.U_out.numel()
            + self.V_out.numel()
            + self.U_in.numel()
            + self.V_in.numel()
        )
        if include_buffers:
            param_count += 3  # alpha, beta, scaling (sign is derived)
        return param_count

    @classmethod
    def wrap_linear(
        cls,
        base_linear: nn.Linear,
        rank_out: int,
        rank_in: int,
        **kwargs,
    ) -> "BracketAdapter":
        return cls(
            base_linear=base_linear, rank_out=rank_out, rank_in=rank_in, **kwargs
        )

    @staticmethod
    def _strip_base_from_state_dict(state_dict, prefix, _local_metadata) -> None:
        keys = [key for key in state_dict if key.startswith(prefix + "base_linear.")]
        for key in keys:
            del state_dict[key]

    @staticmethod
    def _strip_base_from_load(
        state_dict,
        prefix,
        _local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        keys = [
            key
            for key in list(state_dict.keys())
            if key.startswith(prefix + "base_linear.")
        ]
        for key in keys:
            state_dict.pop(key)


__all__ = ["BracketAdapter"]
