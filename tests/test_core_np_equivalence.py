"""E1: frozen regression of exp6 — numpy core must match the torch core.

Reduced grid (the full sweep lives in validation/exp6_np_equivalence.py,
run in CI); gates identical: counts bitwise, |dH| <= 1e-5, flags identical.
"""
import numpy as np
import pytest
import torch

from entropylens import core, core_np


def _cases():
    g = torch.Generator().manual_seed(3)
    zeroinf = torch.randn(500, 6, generator=g)
    zeroinf[torch.rand(500, 6, generator=g) < 0.9] = 0.0
    outlier = torch.randn(500, 6, generator=g)
    outlier[0] = 1000.0
    return {
        "gauss": torch.randn(500, 6, generator=g),
        "lognormal": torch.randn(500, 6, generator=g).mul(2.0).exp(),
        "zeroinf90": zeroinf,
        "outlier": outlier,
        "constant": torch.full((500, 6), 3.14),
    }


@pytest.mark.parametrize("dist", list(_cases()))
@pytest.mark.parametrize("n", [1, 8, 128, 500])
@pytest.mark.parametrize("bins", [8, 32])
def test_batch_wise_equivalence(dist, n, bins):
    xt = _cases()[dist][:n]
    bt = core.batch_wise_entropy(xt, bins=bins)
    bn = core_np.batch_wise_entropy(xt.numpy(), bins=bins)
    # AD-2's strongest gate frozen here, not only in exp6: bitwise counts
    ct, dt = core._quantile_hist_counts(xt.float(), bins, 0.01, 0.99)
    cn, dn = core_np._quantile_hist_counts(
        xt.numpy().astype(np.float32), bins, 0.01, 0.99)
    assert np.array_equal(ct.numpy(), cn)
    assert np.array_equal(dt.numpy(), dn)
    assert np.nanmax(np.abs(bt.entropy.numpy() - bn.entropy)) <= 1e-5
    assert np.nanmax(np.abs(bt.normalized.numpy() - bn.normalized)) <= 5e-6
    assert np.array_equal(bt.reliable.numpy(), bn.reliable)
    assert np.array_equal(bt.reason.numpy(), bn.reason)
    assert bt.n == bn.n and bt.bins == bn.bins


@pytest.mark.parametrize("bins", [8, 32])
def test_batch_wise_equivalence_value_range(bins):
    g = torch.Generator().manual_seed(9)
    xt = torch.randn(256, 6, generator=g)
    vr = (-3.0, 3.0)
    ct, dt = core._quantile_hist_counts(
        xt.float(), bins, 0.01, 0.99, value_range=vr)
    cn, dn = core_np._quantile_hist_counts(
        xt.numpy().astype(np.float32), bins, 0.01, 0.99, value_range=vr)
    assert np.array_equal(ct.numpy(), cn)
    assert np.array_equal(dt.numpy(), dn)
    bt = core.batch_wise_entropy(xt, bins=bins, value_range=vr)
    bn = core_np.batch_wise_entropy(xt.numpy(), bins=bins, value_range=vr)
    assert np.nanmax(np.abs(bt.entropy.numpy() - bn.entropy)) <= 1e-5
    assert np.array_equal(bt.reliable.numpy(), bn.reliable)


@pytest.mark.parametrize("strategy", ["relu", "abs"])
def test_per_example_equivalence(strategy):
    g = torch.Generator().manual_seed(5)
    xt = torch.randn(128, 16, generator=g)
    xt[0] = 0.0
    pt = core.per_example_entropy(xt, strategy)
    pn = core_np.per_example_entropy(xt.numpy(), strategy)
    assert np.array_equal(np.isnan(pt.entropy.numpy()), np.isnan(pn.entropy))
    assert np.array_equal(pt.valid.numpy(), pn.valid)
    assert np.nanmax(np.abs(pt.entropy.numpy() - pn.entropy)) <= 1e-6


@pytest.mark.parametrize("scale", [1e-30, 1e-39, 1e-40, 1e-44])
@pytest.mark.parametrize("strategy", ["relu", "abs"])
def test_per_example_underflow_equivalence(scale, strategy):
    """D10 / exp9 G1 in the numpy mirror: both cores had the same floored
    denominator, so both had to be fixed or E1 would break."""
    xt = torch.full((1, 8), scale)
    pt = core.per_example_entropy(xt, strategy)
    pn = core_np.per_example_entropy(xt.numpy(), strategy)
    assert bool(pt.valid[0]) and bool(pn.valid[0])
    assert abs(float(pn.entropy[0]) - np.log(8)) < 1e-5
    assert abs(float(pt.entropy[0]) - float(pn.entropy[0])) <= 1e-6


def test_per_example_shift_rejected():
    with pytest.raises(ValueError, match="shift"):
        core_np.per_example_entropy(np.ones((4, 4), dtype=np.float32), "shift")


def test_ci_shapes_and_bounds():
    x = np.random.default_rng(0).standard_normal((400, 3)).astype(np.float32)
    est, lo, hi = core_np.confidence_interval(x, n_boot=50,
                                              rng=np.random.default_rng(1))
    assert (lo <= est).all() and (est <= hi).all()
    assert (lo >= 0.0).all()


def test_edge_cases_match_torch_policy():
    with pytest.raises(ValueError, match="empty"):
        core_np.batch_wise_entropy(np.empty((0,), dtype=np.float32))
    with pytest.raises(ValueError, match="NaN or Inf"):
        core_np.batch_wise_entropy(np.array([1.0, np.nan], dtype=np.float32))
    with pytest.raises(ValueError, match="unsupported in v1"):
        core_np.batch_wise_entropy(np.zeros((2, 2, 2, 2), dtype=np.float32))
