"""E1: frozen regression of exp6 — numpy core must match the torch core.

Reduced grid (the full sweep lives in validation/exp6_np_equivalence.py,
run in CI); gates identical: counts bitwise, |dH| <= 1e-5, flags identical.
"""
import warnings

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


@pytest.mark.parametrize("scale", [
    pytest.param(1e-30, id="control-1e-30"), 1e-39, 1e-40, 1e-44])
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


@pytest.mark.parametrize("strategy", ["relu", "abs"])
def test_per_example_overflow_equivalence(strategy):
    """D12 / exp11 G1 in the numpy mirror (row sum overflowing float32)."""
    xt = torch.full((1, 8), 3e38)
    pt = core.per_example_entropy(xt, strategy)
    pn = core_np.per_example_entropy(xt.numpy(), strategy)
    assert bool(pt.valid[0]) and bool(pn.valid[0])
    assert abs(float(pn.entropy[0]) - np.log(8)) < 1e-5
    assert abs(float(pt.entropy[0]) - float(pn.entropy[0])) <= 1e-6


def _overflow_cases():
    g = torch.Generator().manual_seed(11)
    return {   # name -> (unscaled input, power-of-two exponent)
        "G2-hi-minus-lo": (
            torch.randint(-1000, 1001, (512, 3), generator=g).float(), 118),
        "G3-quantile-lerp": (
            torch.cat([torch.full((2,), -1.75),
                       torch.full((126,), 1.75)]).unsqueeze(1), 127),
    }


@pytest.mark.parametrize("case", list(_overflow_cases()))
def test_batch_wise_overflow_equivalence(case):
    """D12 / exp11 G2-G3 in the numpy mirror: where float32 overflows
    (G2: hi - lo; G3: the quantile lerp itself), counts are bitwise equal
    between the two cores AND to the unscaled input's (power-of-two
    scaling is exact), so a revert in both cores is caught too."""
    base, k = _overflow_cases()[case]
    xt = base * 2.0 ** k
    ct, dt = core._quantile_hist_counts(xt, 32, 0.01, 0.99)
    cn, dn = core_np._quantile_hist_counts(xt.numpy(), 32, 0.01, 0.99)
    ref, _ = core_np._quantile_hist_counts(base.numpy(), 32, 0.01, 0.99)
    assert np.array_equal(ct.numpy(), cn) and np.array_equal(dt.numpy(), dn)
    assert np.array_equal(cn, ref) and not dn.any()


def test_overflow_rejection_same_error_in_both_cores():
    """D11 contract under warnings-as-errors: both cores raise the same
    ValueError. numpy's cast used to emit RuntimeWarning('overflow
    encountered in cast') first, so such callers got the wrong exception
    from core_np, and its message did not name the overflow case."""
    x = torch.full((128, 3), 1e39, dtype=torch.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="outside float32 range"):
            core.batch_wise_entropy(x)
        with pytest.raises(ValueError, match="outside float32 range"):
            core_np.batch_wise_entropy(x.numpy())


def test_per_example_shift_rejected():
    with pytest.raises(ValueError, match="shift"):
        core_np.per_example_entropy(np.ones((4, 4), dtype=np.float32), "shift")


def test_ci_shapes_and_bounds():
    x = np.random.default_rng(0).standard_normal((400, 3)).astype(np.float32)
    est, lo, hi = core_np.confidence_interval(x, n_boot=50,
                                              rng=np.random.default_rng(1))
    assert (lo <= est).all() and (est <= hi).all()
    assert (lo >= 0.0).all()


def test_dtype_contract_matches_torch_policy():
    """exp10 G6 (D11): both cores must agree accept/reject on the dtype
    battery. Before D11 the torch core accepted overflowing float64 while
    core_np rejected it — the two had different input contracts."""
    over = torch.full((128, 3), 1e39, dtype=torch.float64)
    over[0, 0] = 1e38
    ok64 = torch.randn(128, 3).double()
    for label, t, should_raise in [("overflowing", over, True),
                                   ("in-range float64", ok64, False)]:
        t_raised = n_raised = False
        try:
            core.batch_wise_entropy(t)
        except ValueError:
            t_raised = True
        try:
            core_np.batch_wise_entropy(t.numpy())
        except ValueError:
            n_raised = True
        assert t_raised == n_raised == should_raise, label


def test_edge_cases_match_torch_policy():
    with pytest.raises(ValueError, match="empty"):
        core_np.batch_wise_entropy(np.empty((0,), dtype=np.float32))
    with pytest.raises(ValueError, match="NaN or Inf"):
        core_np.batch_wise_entropy(np.array([1.0, np.nan], dtype=np.float32))
    with pytest.raises(ValueError, match="unsupported in v1"):
        core_np.batch_wise_entropy(np.zeros((2, 2, 2, 2), dtype=np.float32))
