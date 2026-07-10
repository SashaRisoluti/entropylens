"""A2 bounds + Difetto 3 anchors through the library API. 'shift' must be
rejected (removed per MATH_DECISIONS.md D3)."""
import math

import pytest
import torch

from entropylens import core

ANCHOR = torch.tensor([[5.0, -5.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]])


def test_difetto3_anchors():
    assert abs(float(core.per_example_entropy(ANCHOR, "abs").entropy[0])
               - 0.7362) < 5e-4
    assert abs(float(core.per_example_entropy(ANCHOR, "relu").entropy[0])
               - 0.0856) < 5e-4


def test_shift_rejected_with_rationale():
    with pytest.raises(ValueError, match="shift.*removed"):
        core.per_example_entropy(ANCHOR, "shift")


def test_a2_one_hot_zero_uniform():
    m = 16
    one_hot = torch.zeros(1, m); one_hot[0, 5] = 7.3
    r = core.per_example_entropy(one_hot, "relu")
    assert abs(float(r.entropy[0])) < 1e-7 and bool(r.valid[0])

    uniform = torch.full((1, m), 0.25)
    r = core.per_example_entropy(uniform, "relu")
    assert abs(float(r.entropy[0]) - math.log(m)) < 1e-5
    assert abs(float(r.normalized[0]) - 1.0) < 1e-5

    zero = torch.zeros(1, m)
    r = core.per_example_entropy(zero, "relu")
    assert torch.isnan(r.entropy[0]) and not bool(r.valid[0])


def test_a2_inequality_random_rows():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(64, 16, generator=g)
    for strat in ["relu", "abs"]:
        r = core.per_example_entropy(x, strat)
        h = r.entropy[r.valid]
        assert bool((h >= -1e-9).all() and (h <= math.log(16) + 1e-6).all())
        nrm = r.normalized[r.valid]
        assert bool((nrm >= 0).all() and (nrm <= 1.0 + 1e-9).all())


def test_requires_2d():
    with pytest.raises(ValueError, match="expected at least 2D"):
        core.per_example_entropy(torch.randn(8), "relu")


def test_strategy_is_mandatory():
    with pytest.raises(TypeError):
        core.per_example_entropy(torch.randn(4, 8))  # no strategy arg


def test_outside_autograd():
    x = torch.randn(8, 8, requires_grad=True)
    r = core.per_example_entropy(x, "relu")
    assert not r.entropy.requires_grad
