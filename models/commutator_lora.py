"""Two-sided Commutator-LoRA adapter."""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

GainMode = Literal["per_out", "per_head", "none"]


def _skew_from_square(param: Tensor) -> Tensor:
    return 0.5 * param - param.transpose(-1, -2)


class CommutatorLoRA(nn.Module):
    """Rotation-first low-rank adapter that perturbs a frozen linear weight."""

    version: int = 2

    def __init__(
        self,
        base_linear: nn.Linear,
        rank_lora: int,
        alpha: float = 16.0,
        skew: bool = False,
        use_gain: bool = False,
        gain_mode: GainMode = "per_out",
        n_heads: Optional[int] = None,
        init_std_scale: float = 1.0,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be an nn.Linear instance")
        if rank_lora < 0:
            raise ValueError("rank_lora must be non-negative")

        self.base_linear = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.rank_lora = int(rank_lora)
        self.rank_com = int(math.ceil(self.rank_lora / 2)) if self.rank_lora > 0 else 0
        self.alpha = float(alpha)
        self.skew = bool(skew)
        self.init_std_scale = float(init_std_scale)
        self.adapter_dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None

        self.base_linear.weight.requires_grad_(False)
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad_(False)

        self.scaling = self.alpha / self.rank_com if self.rank_com > 0 else 0.0

        self.use_gain = bool(use_gain) and gain_mode != "none"
        self.gain_mode: GainMode = gain_mode if self.use_gain else "none"
        self.n_heads = int(n_heads) if n_heads is not None else None
        self.head_dim: Optional[int] = None
        if self.gain_mode == "per_head":
            if self.n_heads is None:
                raise ValueError("n_heads must be provided when gain_mode='per_head'")
            if self.out_features % self.n_heads != 0:
                raise ValueError(
                    "out_features must be divisible by n_heads for per_head gains"
                )
            self.head_dim = self.out_features // self.n_heads

        if self.rank_com > 0:
            if self.skew:
                self.U_o = nn.Parameter(torch.zeros(self.out_features, self.rank_com))
                self.Omega_o = nn.Parameter(torch.zeros(self.rank_com, self.rank_com))
                self.U_i = nn.Parameter(torch.zeros(self.in_features, self.rank_com))
                self.Omega_i = nn.Parameter(torch.zeros(self.rank_com, self.rank_com))
                self.V_o = None
                self.V_i = None
            else:
                self.U_o = nn.Parameter(torch.zeros(self.out_features, self.rank_com))
                self.V_o = nn.Parameter(torch.zeros(self.out_features, self.rank_com))
                self.U_i = nn.Parameter(torch.zeros(self.in_features, self.rank_com))
                self.V_i = nn.Parameter(torch.zeros(self.in_features, self.rank_com))
                self.Omega_o = None
                self.Omega_i = None
        else:
            self.U_o = None
            self.V_o = None
            self.U_i = None
            self.V_i = None
            self.Omega_o = None
            self.Omega_i = None

        if self.use_gain:
            if self.gain_mode == "per_out":
                self.gamma = nn.Parameter(torch.zeros(self.out_features))
            elif self.gain_mode == "per_head":
                assert self.head_dim is not None
                self.gamma = nn.Parameter(torch.zeros(self.n_heads, self.head_dim))
            else:
                raise ValueError(f"Unsupported gain mode '{gain_mode}'")
        else:
            self.register_parameter("gamma", None)

        self.register_buffer(
            "adapter_version",
            torch.tensor(self.version, dtype=torch.int32),
            persistent=True,
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        if self.rank_com == 0:
            return
        std_out = self.init_std_scale / math.sqrt(self.out_features)
        std_in = self.init_std_scale / math.sqrt(self.in_features)
        nn.init.normal_(self.U_o, mean=0.0, std=std_out)
        nn.init.normal_(self.U_i, mean=0.0, std=std_in)
        if self.skew:
            nn.init.normal_(self.Omega_o, mean=0.0, std=1e-3)
            nn.init.normal_(self.Omega_i, mean=0.0, std=1e-3)
        else:
            nn.init.normal_(self.V_o, mean=0.0, std=std_out)
            nn.init.normal_(self.V_i, mean=0.0, std=std_in)
        if self.use_gain:
            nn.init.zeros_(self.gamma)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank_lora={self.rank_lora}, rank_com={self.rank_com}, skew={self.skew}, "
            f"gain_mode='{self.gain_mode}'"
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.rank_com == 0 and not self.use_gain:
            return self.base_linear(x)

        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        weight = self.base_linear.weight
        base_output = F.linear(x_flat, weight, bias=None)
        y_flat = base_output

        if self.rank_com > 0:
            if self.skew:
                omega_out = _skew_from_square(self.Omega_o)
                proj_out = base_output.matmul(self.U_o)
                if self.adapter_dropout is not None:
                    proj_out = self.adapter_dropout(proj_out)
                proj_out = proj_out.matmul(omega_out.t())
                delta_out = proj_out.matmul(self.U_o.t())

                omega_in = _skew_from_square(self.Omega_i)
                proj_in = x_flat.matmul(self.U_i)
                if self.adapter_dropout is not None:
                    proj_in = self.adapter_dropout(proj_in)
                proj_in = proj_in.matmul(omega_in.t())
                w_u_i = weight.matmul(self.U_i)
                delta_in = proj_in.matmul(w_u_i.t())
            else:
                proj_out = base_output.matmul(self.V_o)
                if self.adapter_dropout is not None:
                    proj_out = self.adapter_dropout(proj_out)
                delta_out = proj_out.matmul(self.U_o.t())

                proj_in = x_flat.matmul(self.V_i)
                if self.adapter_dropout is not None:
                    proj_in = self.adapter_dropout(proj_in)
                w_u_i = weight.matmul(self.U_i)
                delta_in = proj_in.matmul(w_u_i.t())

            y_flat = y_flat + self.scaling * delta_out - self.scaling * delta_in

        if self.use_gain:
            y_flat = self._apply_gain(y_flat, base_output)

        if self.base_linear.bias is not None:
            y_flat = y_flat + self.base_linear.bias

        return y_flat.view(*orig_shape[:-1], self.out_features)

    def _apply_gain(self, y: Tensor, base_output: Tensor) -> Tensor:
        if self.gain_mode == "per_out":
            return y + base_output * self.gamma.unsqueeze(0)
        if self.gain_mode == "per_head":
            assert self.n_heads is not None and self.head_dim is not None
            new_shape = (-1, self.n_heads, self.head_dim)
            y_view = y.view(new_shape)
            base_view = base_output.view(new_shape)
            y_view = y_view + base_view * self.gamma.unsqueeze(0)
            return y_view.view_as(y)
        return y

    def materialize_delta_weight(self) -> Tensor:
        weight = self.base_linear.weight.detach()
        if self.rank_com == 0:
            delta = torch.zeros_like(weight)
        else:
            if self.skew:
                omega_out = _skew_from_square(self.Omega_o)
                omega_in = _skew_from_square(self.Omega_i)
                s_out = self.U_o @ omega_out @ self.U_o.transpose(0, 1)
                s_in = self.U_i @ omega_in @ self.U_i.transpose(0, 1)
            else:
                s_out = self.U_o @ self.V_o.t()
                s_in = self.U_i @ self.V_i.t()
            delta = self.scaling * (s_out @ weight - weight @ s_in)
        if self.use_gain:
            gamma_vec = self._gamma_vector().unsqueeze(1)
            delta = delta + gamma_vec * weight
        return delta

    def _gamma_vector(self) -> Tensor:
        if not self.use_gain:
            raise RuntimeError("gain not enabled")
        if self.gain_mode == "per_out":
            return self.gamma
        if self.gain_mode == "per_head":
            return self.gamma.reshape(-1)
        return torch.zeros(
            self.out_features,
            dtype=self.base_linear.weight.dtype,
            device=self.base_linear.weight.device,
        )

    def delta_weight(self) -> Tensor:
        return self.materialize_delta_weight()

    def effective_weight(self) -> Tensor:
        return self.base_linear.weight + self.materialize_delta_weight()

    @classmethod
    def wrap_linear(
        cls, base_linear: nn.Linear, rank_lora: int, **kwargs
    ) -> "CommutatorLoRA":
        return cls(base_linear=base_linear, rank_lora=rank_lora, **kwargs)


__all__ = ["CommutatorLoRA"]
