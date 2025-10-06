import os
import math
import json
import hashlib
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoConfig


@dataclass
class ReLoRaConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    keep_original_weights: bool
    lora_only: bool = False
    trainable_scaling: bool = False
    use_c_relora: bool = False
    c_relora_seed: int = 1234567
    c_relora_share_per_width: bool = True

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

    def _ensure_device(self, device: torch.device | str) -> None:
        target = device if isinstance(device, torch.device) else torch.device(device)
        if self.permutation.device != target:
            self.permutation = self.permutation.to(target)
            self.inv_permutation = self.inv_permutation.to(target)
            self.signs = self.signs.to(target)

    def right_multiply(self, x: Tensor) -> Tensor:
        """Compute ``x @ P`` for row-major ``x``."""

        self._ensure_device(x.device)
        orig_shape = x.shape
        rows = x.reshape(-1, self.dim).to(torch.float32)
        rows = rows * self.signs
        rows = rows.index_select(-1, self.permutation)
        rows = _apply_block_fwht(rows, self.block_sizes)
        return rows.reshape(orig_shape).to(x.dtype)

    def right_multiply_transpose(self, x: Tensor) -> Tensor:
        """Compute ``x @ Pᵀ`` for row-major ``x``."""

        self._ensure_device(x.device)
        orig_shape = x.shape
        rows = x.reshape(-1, self.dim).to(torch.float32)
        rows = _apply_block_fwht(rows, self.block_sizes)
        rows = rows.index_select(-1, self.inv_permutation)
        rows = rows * self.signs
        return rows.reshape(orig_shape).to(x.dtype)


def merge_and_reinit_functional(module):
    if not isinstance(module, ReLoRaLinear):
        return

    if getattr(module, "use_c_relora", False) and module.has_srht:
        A_eff = module._effective_A()
        B_eff = module._effective_B()
        _delta = B_eff @ A_eff
    else:
        _delta = module.lora_B.weight @ module.lora_A.weight
    _delta = _delta * module._post_lora_scale()
    module.weight.data += _delta
    nn.init.kaiming_uniform_(module.lora_A.weight, a=math.sqrt(5))

    nn.init.zeros_(module.lora_B.weight)
    if module.trainable_scaling:
        nn.init.zeros_(module.scaling)


class ReLoRaModel(torch.nn.Module):
    def __init__(
        self,
        model,
        *,
        target_modules,
        r=128,
        lora_alpha=32,
        lora_dropout=0.1,
        keep_original_weights=True,
        lora_only=False,
        trainable_scaling=False,
        use_c_relora=False,
        c_relora_seed: int = 1234567,
        c_relora_share_per_width: bool = True,
    ):
        if r <= 0:
            raise ValueError("r must be positive. If you want r == 0, use the original model.")

        super().__init__()
        self.wrapped_model: nn.Module = model
        # expose the underlying model config so downstream code using HuggingFace models keeps working
        if hasattr(model, "config"):
            self.config = model.config
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.keep_original_weights = keep_original_weights
        self.lora_only = lora_only
        self.trainable_scaling = trainable_scaling
        self.use_c_relora = bool(use_c_relora)
        self.c_relora_seed = int(c_relora_seed)
        self.c_relora_share_per_width = bool(c_relora_share_per_width)

        self._config = ReLoRaConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            keep_original_weights=keep_original_weights,
            lora_only=lora_only,
            trainable_scaling=trainable_scaling,
            use_c_relora=self.use_c_relora,
            c_relora_seed=self.c_relora_seed,
            c_relora_share_per_width=self.c_relora_share_per_width,
        )

        # patch methods
        self.forward = self.wrapped_model.forward

        target_modules_list = target_modules
        if isinstance(target_modules_list, str):
            target_modules_list = [target_modules_list]

        for module_name, module in self.wrapped_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            weight_data = module.weight.data if keep_original_weights else None
            bias_data = None
            if module.bias is not None:
                bias_data = module.bias.data if keep_original_weights else None

            new_module = ReLoRaLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                lora_only=self.lora_only,
                trainable_scaling=self.trainable_scaling,
                weight_data=weight_data,
                bias_data=bias_data,
                use_c_relora=self.use_c_relora,
                c_relora_seed=self.c_relora_seed,
                c_relora_share_per_width=self.c_relora_share_per_width,
                module_name=module_name,
            )
            if self.keep_original_weights:
                # make lora'ed network to be exacty the same as the original network at initialization
                assert new_module.lora_A.bias is None
                assert new_module.lora_B.bias is None

            if self.lora_only:
                assert not self.keep_original_weights
                module.weight = None

            del module

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

        torch.cuda.empty_cache()

    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.wrapped_model.get_submodule(parent_name)
        return parent

    def merge_and_reinit(self):
        for module in self.modules():
            if isinstance(module, ReLoRaLinear):
                module.merge_and_reinit()

    def save_pretrained(self, path):
        self.wrapped_model.save_pretrained(path)
        with open(os.path.join(path, "relora_config.json"), "w") as f:
            json.dump(self._config.__dict__, f, indent=4)

    @classmethod
    def from_pretrained(cls, path):
        with open(os.path.join(path, "relora_config.json"), "r") as f:
            relora_config = json.load(f)

        config = AutoConfig.from_pretrained(path)

        base_model = AutoModelForCausalLM.from_config(config)
        if "keep_original" in relora_config:
            print("WARNING: keep_original is deprecated. Use lora_only instead.")
            print(f"keep_original: {relora_config['keep_original']}")
            relora_config["lora_only"] = not relora_config.pop("keep_original")
            relora_config["keep_original_weights"] = not relora_config["lora_only"]

        if "trainable_scaling" not in relora_config:
            relora_config["trainable_scaling"] = False

        model = cls(base_model, **relora_config)

        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        model.wrapped_model.load_state_dict(state_dict, strict=True)
        return model


# The code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
class ReLoRaLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        *,
        lora_alpha: int = 1,
        lora_dropout: float = 0.1,
        lora_only: bool = False,
        weight_data=None,
        bias_data=None,
        trainable_scaling: bool = False,
        bias=True,
        device=None,
        dtype=None,
        use_c_relora: bool = False,
        c_relora_seed: int = 1234567,
        c_relora_share_per_width: bool = True,
        module_name: Optional[str] = None,
    ):
        """Wrap a linear layer W with LoRA (and optional C-ReLoRA rotations)."""

        super().__init__()
        if r <= 0:
            raise ValueError("r must be positive. If you want r == 0, use the original model.")

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.lora_only = lora_only
        self.trainable_scaling = trainable_scaling
        self.use_c_relora = bool(use_c_relora)
        self.c_relora_seed = int(c_relora_seed)
        self.c_relora_share_per_width = bool(c_relora_share_per_width)
        self.module_name = module_name or ""
        self._cycle_index = 0
        self.input_basis: Optional[SRHTBasisTransform] = None
        self.output_basis: Optional[SRHTBasisTransform] = None

        self.register_buffer("_has_srht_flag", torch.tensor(0, dtype=torch.int8), persistent=True)

        if lora_only:
            self.weight = None
            self.bias = None
        else:
            if bias_data is None:
                bias_data = (
                    torch.zeros(out_features, device=device, dtype=dtype, requires_grad=False)
                    if bias
                    else None
                )
            self.bias = nn.Parameter(bias_data, requires_grad=False) if bias else None

            if weight_data is None:
                weight_data = torch.zeros(
                    out_features,
                    in_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=False,
                )

            self.weight = nn.Parameter(weight_data, requires_grad=False)

        w_dev = None
        w_dtype = None
        if weight_data is not None:
            w_dev = weight_data.device
            w_dtype = weight_data.dtype
        elif device is not None:
            w_dev = torch.device(device) if not isinstance(device, torch.device) else device
            w_dtype = dtype

        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            self.lora_B = nn.Linear(r, out_features, bias=False)
            nn.init.zeros_(self.lora_B.weight)
            if trainable_scaling:
                self.scaling = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
            else:
                self.scaling = self.lora_alpha / self.r

            if w_dev is not None or w_dtype is not None:
                self.lora_A.to(device=w_dev, dtype=w_dtype)
                self.lora_B.to(device=w_dev, dtype=w_dtype)
                if self.trainable_scaling and isinstance(self.scaling, nn.Parameter):
                    self.scaling.data = self.scaling.data.to(device=w_dev, dtype=w_dtype)

            if not self.lora_only:
                self.weight.requires_grad = False

        transform_device = None
        if weight_data is not None:
            transform_device = weight_data.device
        elif not self.lora_only and hasattr(self, "weight") and self.weight is not None:
            transform_device = self.weight.device
        elif r > 0:
            transform_device = self.lora_A.weight.device
        else:
            transform_device = device

        self._build_transforms(device=transform_device)

    def _post_lora_scale(self):
        if self.trainable_scaling:
            return self.scaling.tanh()
        return self.scaling

    def _stable_hash64(self, s: str) -> int:
        h = hashlib.sha256(s.encode("utf-8")).digest()
        return int.from_bytes(h[:8], "little", signed=False)

    def _seed_for(self, role: str, dim: int, cycle_index: int) -> int:
        base = f"{self.c_relora_seed}|{cycle_index}|{role}|{dim}"
        if not self.c_relora_share_per_width:
            base += f"|{self.module_name}"
        return self._stable_hash64(base) & 0x7FFFFFFF

    def _build_transforms(self, device=None):
        if not self.use_c_relora:
            self.input_basis = None
            self.output_basis = None
            self._has_srht_flag.fill_(0)
            return

        if device is None:
            if not self.lora_only and getattr(self, "weight", None) is not None:
                device = self.weight.device
            elif hasattr(self, "lora_A"):
                device = self.lora_A.weight.device
            else:
                device = torch.device("cpu")

        dev = device if isinstance(device, torch.device) else torch.device(device)
        gen_device = dev.type

        g_in = torch.Generator(device=gen_device)
        g_in.manual_seed(self._seed_for("in", self.in_features, self._cycle_index))
        g_out = torch.Generator(device=gen_device)
        g_out.manual_seed(self._seed_for("out", self.out_features, self._cycle_index))

        self.input_basis = SRHTBasisTransform(self.in_features, dev, g_in)
        self.output_basis = SRHTBasisTransform(self.out_features, dev, g_out)
        self._has_srht_flag.fill_(1)

    @property
    def has_srht(self) -> bool:
        return bool(self._has_srht_flag.item())

    def _effective_A(self) -> Tensor:
        if self.use_c_relora and self.has_srht and self.input_basis is not None:
            return self.input_basis.right_multiply(self.lora_A.weight)
        return self.lora_A.weight

    def _effective_B(self) -> Tensor:
        if self.use_c_relora and self.has_srht and self.output_basis is not None:
            return self.output_basis.right_multiply(self.lora_B.weight.T).T
        return self.lora_B.weight

    @torch.no_grad()
    def merge_and_reinit(self):
        if self.lora_only:
            print("WARNING: Skipping merge and reinit, because only lora parameters are used")
            if self.use_c_relora:
                self._cycle_index += 1
                self._build_transforms(device=self.lora_A.weight.device)
            return

        A_eff = self._effective_A()
        B_eff = self._effective_B()
        delta = (B_eff @ A_eff) * self._post_lora_scale()
        self.weight.data.add_(delta)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        if self.trainable_scaling:
            nn.init.zeros_(self.scaling)

        if self.use_c_relora:
            self._cycle_index += 1
            target_device = self.weight.device if self.weight is not None else self.lora_A.weight.device
            self._build_transforms(device=target_device)

    def forward(self, x: torch.Tensor):
        if self.lora_only:
            if self.r <= 0:
                return x
            adapter_input = (
                self.lora_dropout(x)
                if self.training and getattr(self.lora_dropout, "p", 0.0) > 0.0
                else x
            )
            A_eff = self._effective_A()
            down = F.linear(adapter_input, A_eff)
            B_eff = self._effective_B()
            up = F.linear(down, B_eff)
            return up * self._post_lora_scale()

        result = F.linear(x, self.weight, bias=self.bias)

        if self.r > 0:
            adapter_input = (
                self.lora_dropout(x)
                if self.training and getattr(self.lora_dropout, "p", 0.0) > 0.0
                else x
            )
            A_eff = self._effective_A()
            down = F.linear(adapter_input, A_eff)
            B_eff = self._effective_B()
            up = F.linear(down, B_eff)
            result = result + up * self._post_lora_scale()
        return result
