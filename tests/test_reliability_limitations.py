"""Frozen documentation of the reliability flag's KNOWN limitations
(adversarial review of MATH_DECISIONS.md D2, reproduced at 400 seeds).

These tests assert the DOCUMENTED behavior, not that it is desirable:
the flag is a necessary condition (bin-count-bias model), NOT sufficient.
If a future criterion fixes these cases, update D2 and flip the asserts.
"""
import math

import torch

from entropylens import core


def _analytic_zeroinf_exp_ref(pi=0.9, bins=32):
    """Population binned-clipped normalized entropy of the mixture
    pi*delta_0 + (1-pi)*Exp(1). Q(0.01)=0 (atom), Q(0.99)=ln(10) for
    pi=0.9; atom + lower-clip land in bin 1, upper clip in the last bin."""
    q_hi = -math.log((1.0 - 0.99) / (1.0 - pi))  # F(x)=pi+(1-pi)(1-e^-x)
    edges = torch.linspace(0.0, q_hi, bins + 1, dtype=torch.float64)
    cont = (1.0 - pi) * (1.0 - torch.exp(-edges))  # continuous part CDF
    probs = (cont[1:] - cont[:-1]).clone()
    probs[0] += pi                       # atom at 0
    probs[-1] += 1.0 - pi - float(cont[-1])   # clipped upper tail
    probs = probs[probs > 0]
    h = float(-(probs * probs.log()).sum())
    return h / math.log(bins)


def test_flag_green_while_biased_on_sparse_data():
    """zero-inflated 90% + Exp(1) at N=128, B=32: reliable=True but the
    mean normalized entropy under-estimates the population value by ~7%
    (probe: -7.43% +- 1.16 sem at 400 seeds) — beyond the 5% target the
    flag's prose used to promise. Asserted at a safe -3% margin."""
    ref = _analytic_zeroinf_exp_ref()
    n, seeds = 128, 200
    vals = []
    for s in range(seeds):
        g = torch.Generator().manual_seed(s)
        u = torch.rand(n, 1, generator=g)
        e = -(1.0 - torch.rand(n, 1, generator=g)).log()
        x = torch.where(u < 0.9, torch.zeros_like(e), e)
        r = core.batch_wise_entropy(x)
        assert bool(r.reliable[0]), "flag must be True here (the point!)"
        vals.append(float(r.normalized[0]))
    dev = (sum(vals) / seeds - ref) / ref
    assert dev < -0.03, (
        f"documented limitation vanished? dev={dev:+.4f}; if a new "
        "criterion fixed this, update MATH_DECISIONS.md D2 and this test")


def test_flag_conservative_side_is_safe_direction():
    """N=50 < 90: flag red even though the corrected estimate is decent
    on gaussian data (exp2: 1.7% dev) — conservative by design."""
    r = core.batch_wise_entropy(
        torch.randn(50, 1, generator=torch.Generator().manual_seed(0)))
    assert not bool(r.reliable[0])
