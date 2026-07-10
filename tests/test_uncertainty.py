"""C4: full-pipeline bootstrap CI (MATH_DECISIONS.md D4 as amended,
validation/exp4b): resample RAW values, re-estimate quantile edges per
replicate, normal interval H_MM +- z*sd_boot. The closed-form (Basharin)
was REJECTED: it misses edge re-estimation variance and under-covers
(exp4b: min 62.5% vs the >=85% gate)."""
import math

import torch

from entropylens import core


def _analytic_h(bins=32, q_lo=0.01, q_hi=0.99):
    z_lo = float(torch.special.ndtri(torch.tensor(q_lo)))
    z_hi = float(torch.special.ndtri(torch.tensor(q_hi)))
    edges = torch.linspace(z_lo, z_hi, bins + 1)
    cdf = torch.special.ndtr(edges)
    probs = (cdf[1:] - cdf[:-1]).clone()
    probs[0] += cdf[0]
    probs[-1] += 1.0 - cdf[-1]
    return float(-(probs * probs.log()).sum())


def test_ci_shape_ordering_and_centering():
    x = torch.randn(500, 3, generator=torch.Generator().manual_seed(0))
    est, lo, hi = core.confidence_interval(
        x, n_boot=100, generator=torch.Generator().manual_seed(1))
    r = core.batch_wise_entropy(x)
    assert est.shape == lo.shape == hi.shape == (3,)
    assert torch.equal(est, r.entropy)  # centered on the point estimate
    assert bool((lo <= est).all() and (est <= hi).all())
    assert bool((hi > lo).all())  # nondegenerate data -> positive width


def test_ci_deterministic_given_generator():
    x = torch.randn(300, 2, generator=torch.Generator().manual_seed(2))
    a = core.confidence_interval(
        x, n_boot=50, generator=torch.Generator().manual_seed(3))
    b = core.confidence_interval(
        x, n_boot=50, generator=torch.Generator().manual_seed(3))
    assert all(torch.equal(u, v) for u, v in zip(a, b))


def test_ci_coverage_full_pipeline_smoke():
    """Scaled-down exp4b anchor: gaussian N=400, 40 reps, coverage of the
    analytic binned reference >= 85% (exp4b at full scale: 94.5-96.0% on
    normal; at true coverage 94.5%, P(observed < 0.85 over 40 reps) < 1%)."""
    h_true = _analytic_h()
    hits = 0
    reps = 40
    for seed in range(reps):
        x = torch.randn(400, 1, generator=torch.Generator().manual_seed(seed))
        est, lo, hi = core.confidence_interval(
            x, n_boot=100, generator=torch.Generator().manual_seed(10_000 + seed))
        hits += bool(lo[0] <= h_true <= hi[0])
    assert hits / reps >= 0.85, f"coverage {hits / reps}"


def test_ci_upper_bound_clamped_to_theoretical_max():
    """Adversarial finding: on near-uniform data the raw normal interval
    exceeded log(min(N,B)) in ~100% of runs; the estimand cannot, so the
    interval is truncated to the bound (never lowers coverage)."""
    x = torch.rand(500, 2, generator=torch.Generator().manual_seed(4))
    est, lo, hi = core.confidence_interval(
        x, n_boot=100, generator=torch.Generator().manual_seed(5))
    bound = math.log(min(500, 32))
    assert bool((hi <= torch.maximum(torch.full_like(est, bound), est)).all())
    assert bool((hi >= est).all())


def test_ci_sparsity_warning_below_validated_floor():
    """Adversarial finding: at >90% single-bin mass the coverage floor
    rises to N~400 (measured 83% at N=200); the function must warn."""
    import pytest
    g = torch.Generator().manual_seed(6)
    x = torch.cat([torch.zeros(190, 1), torch.rand(10, 1, generator=g) + 0.5])
    with pytest.warns(UserWarning, match="single bin"):
        core.confidence_interval(x, n_boot=20,
                                 generator=torch.Generator().manual_seed(7))
    # large N with the same sparsity: no warning
    x_big = torch.cat([torch.zeros(450, 1),
                       torch.rand(50, 1, generator=g) + 0.5])
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        core.confidence_interval(x_big, n_boot=20,
                                 generator=torch.Generator().manual_seed(8))


def test_degenerate_zero_width_and_flagged():
    """Documented blind spot (D4): degenerate sample -> every bootstrap
    replicate identical -> zero-width CI; the reliable flag (not the CI)
    must carry the warning."""
    x = torch.full((500, 1), 2.0)
    est, lo, hi = core.confidence_interval(x, n_boot=20)
    assert float(est[0]) == 0.0 and float(lo[0]) == 0.0 and float(hi[0]) == 0.0
    assert not bool(core.batch_wise_entropy(x).reliable[0])
