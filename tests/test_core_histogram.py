"""Internal histogram machinery: per-column quantile-winsorized counts
must agree with the torch.histc reference used by validation/exp1."""
import math

import pytest
import torch

from entropylens import core


def _histc_reference(values, bins, q_lo, q_hi):
    """Scalar reference: identical to validation/common.py quantile_hist_counts."""
    x = values.float().flatten()
    lo = torch.quantile(x, q_lo).item()
    hi = torch.quantile(x, q_hi).item()
    if hi <= lo:
        return None
    return torch.histc(x.clamp(lo, hi), bins=bins, min=lo, max=hi)


def test_counts_match_histc_reference_per_column():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(500, 7, generator=g)
    counts, degenerate = core._quantile_hist_counts(x, bins=32, q_lo=0.01, q_hi=0.99)
    assert counts.shape == (7, 32)
    assert not degenerate.any()
    for j in range(7):
        ref = _histc_reference(x[:, j], 32, 0.01, 0.99)
        assert torch.equal(counts[j], ref), f"column {j} disagrees with histc"


def test_degenerate_column_flagged():
    x = torch.randn(100, 3)
    x[:, 1] = 4.2  # constant column
    counts, degenerate = core._quantile_hist_counts(x, bins=32, q_lo=0.01, q_hi=0.99)
    assert degenerate.tolist() == [False, True, False]


def test_validation_rejects_nan_inf():
    bad_nan = torch.tensor([[1.0, float("nan")]])
    bad_inf = torch.tensor([[1.0, float("inf")]])
    for bad in (bad_nan, bad_inf):
        with pytest.raises(ValueError, match="NaN or Inf"):
            core.batch_wise_entropy(bad)


@pytest.mark.parametrize("fn,args", [
    (core.batch_wise_entropy, ()),
    (core.per_example_entropy, ("abs",)),
    (core.confidence_interval, ()),
])
def test_validation_rejects_float64_overflowing_float32(fn, args):
    """Frozen regression of validation/exp10 G1 (MATH_DECISIONS.md D11).

    A float64 finite above float32's max used to pass the finiteness check
    (run on the ORIGINAL dtype) and become inf on the downcast, after which
    batch_wise_entropy reported H=0 with reliable=True — a silent
    'maximally concentrated neuron' reading."""
    x = torch.full((128, 4), 1e39, dtype=torch.float64)
    x[0, 0] = 1e38
    assert torch.isfinite(x).all(), "input must be finite in float64"
    with pytest.raises(ValueError, match="NaN or Inf"):
        fn(x, *args)


def test_validation_accepts_in_range_float64():
    """exp10 G3: the fix must not reject ordinary float64 — only values
    that actually overflow float32."""
    g = torch.Generator().manual_seed(10)
    x32 = torch.randn(128, 4, generator=g)
    b32 = core.batch_wise_entropy(x32)
    b64 = core.batch_wise_entropy(x32.double())
    assert torch.equal(b32.entropy, b64.entropy)
    assert torch.equal(b32.reliable, b64.reliable)
    # 1e37 < float32 max: large but representable, must be accepted
    core.batch_wise_entropy(torch.full((128, 2), 1e37, dtype=torch.float64))


def test_validation_rejects_4d():
    with pytest.raises(ValueError, match="unsupported in v1"):
        core.batch_wise_entropy(torch.randn(2, 3, 8, 8))


def test_validation_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        core.batch_wise_entropy(torch.empty(0, 4))
