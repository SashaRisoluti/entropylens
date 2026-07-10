"""A1 (bounds), A3 (outlier gate <10%), A4 (N-invariance <5%), plus the
exp1/exp2 anchors, now exercised through the library API."""
import math

import torch

from entropylens import core


def test_a1_bound_inequality_across_regimes():
    g = torch.Generator().manual_seed(0)
    for n in [8, 32, 100, 1000]:
        r = core.batch_wise_entropy(torch.randn(n, 1, generator=g))
        bound = math.log(min(n, 32))
        h = float(r.entropy[0])
        assert -1e-9 <= h, f"N={n}: H={h} negative"
        # Miller-Madow can exceed the plugin bound; normalized is clamped
        assert 0.0 <= float(r.normalized[0]) <= 1.0


def test_a1_attainment_n_below_b_no_clipping():
    # 8 distinct spread values, 32 bins, no winsorization -> H_plugin=log 8.
    x = torch.linspace(0.0, 1.0, 8).unsqueeze(1)
    r = core.batch_wise_entropy(x, q_lo=0.0, q_hi=1.0)
    h_plugin = float(r.entropy[0]) - (8 - 1) / (2 * 8)  # subtract MM term
    assert abs(h_plugin - math.log(8)) < 1e-5


def test_a1_attainment_uniform_divisible():
    x = torch.arange(32, dtype=torch.float).repeat_interleave(10).unsqueeze(1)
    r = core.batch_wise_entropy(x, q_lo=0.0, q_hi=1.0)
    h_plugin = float(r.entropy[0]) - (32 - 1) / (2 * 320)
    assert abs(h_plugin - math.log(32)) < 1e-5
    assert float(r.normalized[0]) == 1.0  # MM pushes over; clamp holds


def test_a1_degenerate_and_n1():
    r = core.batch_wise_entropy(torch.full((500, 2), -7.0))
    assert torch.all(r.entropy == 0.0) and torch.all(r.normalized == 0.0)
    assert not r.reliable.any()
    assert torch.all(r.reason == core.REASON_DEGENERATE)
    r1 = core.batch_wise_entropy(torch.tensor([[0.3]]))
    assert float(r1.entropy[0]) == 0.0 and not bool(r1.reliable[0])
    assert int(r1.reason[0]) == core.REASON_N1


def test_a3_outlier_gate():
    """Difetto 2 regression: 1 outlier on 1000 gaussians moves H by <10%
    (measured 0.18% in validation/exp1)."""
    rels = []
    for seed in range(10):
        g = torch.Generator().manual_seed(seed)
        base = torch.randn(1000, 1, generator=g)
        with_out = torch.cat([base, torch.tensor([[1000.0]])])
        h0 = float(core.batch_wise_entropy(base).entropy[0])
        h1 = float(core.batch_wise_entropy(with_out).entropy[0])
        rels.append(abs(h1 - h0) / abs(h0))
    assert sum(rels) / len(rels) < 0.10, f"mean rel change {sum(rels)/len(rels)}"


def _analytic_normalized_href(bins=32, q_lo=0.01, q_hi=0.99):
    """Binned-clipped N(0,1) reference from validation/exp2."""
    z_lo = float(torch.special.ndtri(torch.tensor(q_lo)))
    z_hi = float(torch.special.ndtri(torch.tensor(q_hi)))
    edges = torch.linspace(z_lo, z_hi, bins + 1)
    cdf = torch.special.ndtr(edges)
    probs = (cdf[1:] - cdf[:-1]).clone()
    probs[0] += cdf[0]
    probs[-1] += 1.0 - cdf[-1]
    h = float(-(probs * probs.log()).sum())
    return h / math.log(bins)


def test_a4_n_invariance_of_normalized_entropy():
    """Difetto 1 regression: same distribution, 4 batch sizes, normalized
    entropy within 5% of the analytic reference at every N (exp2: <=0.34%)."""
    href = _analytic_normalized_href()
    for n in [128, 512, 2048, 8192]:
        vals = []
        for seed in range(20):
            g = torch.Generator().manual_seed(seed)
            r = core.batch_wise_entropy(torch.randn(n, 1, generator=g))
            vals.append(float(r.normalized[0]))
        mean = sum(vals) / len(vals)
        assert abs(mean - href) / href < 0.05, f"N={n}: {mean} vs {href}"


def test_reliability_flag_thresholds():
    # B=32 -> threshold N ~ 90 (MATH_DECISIONS.md D2)
    for n, expected in [(32, False), (50, False), (128, True), (512, True)]:
        g = torch.Generator().manual_seed(0)
        r = core.batch_wise_entropy(torch.randn(n, 1, generator=g))
        assert bool(r.reliable[0]) == expected, f"N={n}"
        if not expected:
            assert int(r.reason[0]) == core.REASON_SMALL_N


def test_1d_input_treated_as_single_neuron():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(200, generator=g)
    r1 = core.batch_wise_entropy(x)
    r2 = core.batch_wise_entropy(x.unsqueeze(1))
    assert torch.equal(r1.entropy, r2.entropy)


def test_entropy_outside_autograd():
    x = torch.randn(100, 4, requires_grad=True)
    r = core.batch_wise_entropy(x)
    assert not r.entropy.requires_grad
    assert not r.normalized.requires_grad
