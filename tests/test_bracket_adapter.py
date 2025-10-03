import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.bracket_adapter import BracketAdapter


def _make_adapter(
    d_out: int,
    d_in: int,
    r_out: int,
    r_in: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    mode: str = "commutator",
    dropout: float = 0.0,
) -> BracketAdapter:
    base = nn.Linear(d_in, d_out, bias=True)
    adapter = BracketAdapter(
        base_linear=base,
        rank_out=r_out,
        rank_in=r_in,
        alpha=alpha,
        beta=beta,
        lora_alpha=16.0,
        mode=mode,
        dropout_p=dropout,
    )
    return adapter


def test_forward_shape_rectangular():
    torch.manual_seed(0)
    adapter = _make_adapter(d_out=128, d_in=96, r_out=8, r_in=6)
    x = torch.randn(4, 96)
    y = adapter(x)
    assert y.shape == (4, 128)


def test_delta_weight_rank_bound():
    torch.manual_seed(1)
    r_out = 5
    r_in = 7
    adapter = _make_adapter(d_out=64, d_in=48, r_out=r_out, r_in=r_in)
    delta = adapter.delta_weight()
    rank = torch.linalg.matrix_rank(delta, tol=1e-5)
    assert rank <= r_out + r_in


def test_merge_round_trip_matches_forward():
    torch.manual_seed(2)
    adapter = _make_adapter(d_out=32, d_in=40, r_out=4, r_in=3, beta=0.6)
    x = torch.randn(3, 40)
    y_before = adapter(x)
    adapter.merge_into_base()
    y_merged = adapter(x)
    torch.testing.assert_close(y_before, y_merged, atol=1e-5, rtol=1e-5)
    adapter.unmerge_from_base()
    y_after = adapter(x)
    torch.testing.assert_close(y_before, y_after, atol=1e-5, rtol=1e-5)


def test_parameter_count_matches_formula():
    d_out, d_in = 96, 64
    r_out, r_in = 6, 5
    adapter = _make_adapter(d_out=d_out, d_in=d_in, r_out=r_out, r_in=r_in)
    expected = 2 * d_out * r_out + 2 * d_in * r_in + 3
    assert adapter.num_adapter_parameters() == expected


def test_grad_flow_only_adapter_params():
    torch.manual_seed(4)
    adapter = _make_adapter(d_out=48, d_in=32, r_out=4, r_in=4)
    x = torch.randn(10, 32)
    target = torch.randn(10, 48)
    loss = torch.nn.functional.mse_loss(adapter(x), target)
    loss.backward()

    assert adapter.U_out.grad is not None
    assert adapter.V_out.grad is not None
    assert adapter.U_in.grad is not None
    assert adapter.V_in.grad is not None
    assert adapter.base_linear.weight.grad is None
    if adapter.base_linear.bias is not None:
        assert adapter.base_linear.bias.grad is None
