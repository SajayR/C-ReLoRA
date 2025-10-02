import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.commutator_lora import CommutatorLoRA


def _make_adapter(d_in: int, d_out: int, rank: int, **kwargs) -> CommutatorLoRA:
    base = nn.Linear(d_in, d_out, bias=True)
    return CommutatorLoRA(base_linear=base, rank_lora=rank, **kwargs)


def test_forward_matches_manual() -> None:
    torch.manual_seed(0)
    d_in, d_out, rank = 7, 5, 4
    adapter = _make_adapter(d_in, d_out, rank, skew=False)
    x = torch.randn(3, d_in)

    with torch.no_grad():
        delta = adapter.materialize_delta_weight()
        weight_eff = adapter.base_linear.weight + delta
        expected = F.linear(x, weight_eff, bias=adapter.base_linear.bias)
        actual = adapter(x)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_forward_supports_3d_inputs() -> None:
    torch.manual_seed(1)
    d_in, d_out, rank = 4, 6, 4
    adapter = _make_adapter(d_in, d_out, rank)
    x = torch.randn(2, 5, d_in)

    with torch.no_grad():
        delta = adapter.materialize_delta_weight()
        weight_eff = adapter.base_linear.weight + delta
        expected = F.linear(x, weight_eff, bias=adapter.base_linear.bias)
        actual = adapter(x)

    assert actual.shape == (2, 5, d_out)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_grad_flow_freezes_base_weight() -> None:
    torch.manual_seed(2)
    adapter = _make_adapter(6, 7, rank=4)
    x = torch.randn(3, 6)
    loss = adapter(x).pow(2).mean()
    loss.backward()

    assert adapter.base_linear.weight.grad is None
    assert adapter.base_linear.bias.grad is None
    grads = [p.grad for p in adapter.parameters() if p is not None and p.requires_grad]
    assert all(g is not None for g in grads)


def test_forward_handles_dtype_cast() -> None:
    torch.manual_seed(3)
    adapter = _make_adapter(5, 4, rank=4).to(torch.float16)
    x = torch.randn(2, 5, dtype=torch.float16)
    out = adapter(x)
    assert out.dtype == torch.float16


def test_parameter_count_matches_lora_budget() -> None:
    torch.manual_seed(4)
    d_in, d_out, rank = 8, 10, 6
    adapter = _make_adapter(d_in, d_out, rank)
    trainable = sum(p.numel() for p in adapter.parameters() if p is not None and p.requires_grad)
    expected = rank * (d_in + d_out)
    assert trainable == expected


def test_skew_variant_preserves_singular_values_first_order() -> None:
    torch.manual_seed(5)
    d_in = d_out = 6
    adapter = _make_adapter(d_in, d_out, rank=4, skew=True)
    eps = 1e-3

    with torch.no_grad():
        base_weight = adapter.base_linear.weight.clone()
        delta = adapter.materialize_delta_weight()
        sv_base = torch.linalg.svdvals(base_weight)
        sv_eps = torch.linalg.svdvals(base_weight + eps * delta)
        relative = torch.norm(sv_eps - sv_base) / torch.norm(sv_base)

    assert relative < 5e-3


def test_effective_rank_bounded() -> None:
    torch.manual_seed(6)
    adapter = _make_adapter(9, 8, rank=6)
    with torch.no_grad():
        delta = adapter.materialize_delta_weight()
        rank_delta = torch.linalg.matrix_rank(delta, tol=1e-5).item()
    assert rank_delta <= min(2 * adapter.rank_com, adapter.in_features, adapter.out_features)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
