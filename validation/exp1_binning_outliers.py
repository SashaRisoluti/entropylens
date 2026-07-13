"""EXP1 (Q1 / Difetto 2 / A3): outlier robustness of binning candidates.

Protocol (mirrors the documented Difetto-2 defect analysis):
  - 10 seeds, 1000 values ~ N(0,1), then append ONE outlier (1000.0).
  - Gate (A3, user-approved): mean relative change of H < 10%.
  - Baseline (min-max) must REPRODUCE the documented collapse (~99.7%),
    proving the test actually captures the defect.

Extra limit cases per candidate:
  - degenerate (all values identical)      -> H = 0
  - N < B (10 samples, 32 bins)            -> H <= log(10) + tol
  - uniform exact attainment, min-max      -> H = log(32) (analytic)
  - uniform exact attainment, quantile     -> H = analytic clipped value
  - ASYMMETRIC quantiles (q_lo != 1-q_hi)  -> distinct edge-bin masses
    (regression for the D6 formula bug found in adversarial review)
  - KL constant convention                 -> H(N(0,1)) ~= 1.4189
"""

import math

import torch

from common import (hist_entropy_minmax, hist_entropy_quantile,
                    hist_entropy_fixed, knn_entropy_kl)

GATE = 0.10  # A3: relative change < 10%
failures = []


def check(name, cond, detail):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not cond:
        failures.append(name)


# ---------------------------------------------------------- outlier battery
print("=== outlier robustness: 1000 N(0,1) values + one outlier (1000.0) ===")
rel_changes = {"minmax": [], "quantile": [], "fixed": [], "knn": []}
for seed in range(10):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(1000, generator=g)
    with_out = torch.cat([base, torch.tensor([1000.0])])

    for name, fn in [
        ("minmax", lambda v: hist_entropy_minmax(v, bins=32)),
        ("quantile", lambda v: hist_entropy_quantile(v, bins=32)),
        ("fixed", lambda v: hist_entropy_fixed(v, lo=-4.0, hi=4.0, bins=32)),
        ("knn", lambda v: knn_entropy_kl(v, k=3)),
    ]:
        h0, h1 = fn(base), fn(with_out)
        rel_changes[name].append(abs(h1 - h0) / abs(h0))

for name, vals in rel_changes.items():
    mean_rel = sum(vals) / len(vals)
    print(f"  {name:9s} mean relative change = {mean_rel * 100:7.3f}%  "
          f"(max {max(vals) * 100:.3f}%)")

check("baseline reproduces Difetto 2",
      sum(rel_changes["minmax"]) / 10 > 0.90,
      f"min-max collapses by {sum(rel_changes['minmax']) / 10 * 100:.1f}% "
      "(doc: ~99.7%)")
for cand in ["quantile", "fixed", "knn"]:
    mean_rel = sum(rel_changes[cand]) / 10
    check(f"A3 gate ({cand})", mean_rel < GATE,
          f"mean rel change {mean_rel * 100:.3f}% < {GATE * 100:.0f}%")

# ------------------------------------------------------------- limit cases
print("\n=== limit cases ===")

h = hist_entropy_quantile(torch.full((1000,), 3.14), bins=32)
check("quantile: degenerate -> 0", h == 0.0, f"H={h}")

h = hist_entropy_minmax(torch.full((1000,), 3.14), bins=32)
check("minmax: degenerate -> 0", h == 0.0, f"H={h}")

g = torch.Generator().manual_seed(0)
h = hist_entropy_quantile(torch.randn(10, generator=g), bins=32)
check("quantile: N=10 < B=32 -> H <= log(10)",
      h <= math.log(10) + 1e-6, f"H={h:.4f}, log(10)={math.log(10):.4f}")

# uniform attainment, min-max: 3200 points evenly spread over 32 bins
vals = torch.linspace(0.0, 1.0, 3200)
h = hist_entropy_minmax(vals, bins=32)
check("minmax: uniform -> log(32)",
      abs(h - math.log(32)) < 1e-3, f"H={h:.5f}, log(32)={math.log(32):.5f}")

# uniform attainment, quantile: analytic expected value under clipping.
# In-range mass 98% spread over 32 bins; the 1% clipped on each side is
# added to the corresponding edge bin.
p_mid = 0.98 / 32
p_edge = p_mid + 0.01
h_expected = -(30 * p_mid * math.log(p_mid) + 2 * p_edge * math.log(p_edge))
h = hist_entropy_quantile(vals, bins=32)
check("quantile: uniform -> analytic clipped max",
      abs(h - h_expected) / h_expected < 1e-2,
      f"H={h:.5f}, expected {h_expected:.5f} "
      f"(= {h_expected / math.log(32) * 100:.2f}% of log(32))")

# asymmetric quantiles: the two edge bins receive DIFFERENT clipped masses
# (p_lo = p_m + q_lo, p_hi = p_m + (1 - q_hi)); the earlier symmetric-only
# formula (p_e = p_m + q_lo for both edges) sums to 1.04 at q=(0.05, 0.99)
# and even exceeds log(32) — caught in adversarial review of D6.
for q_lo, q_hi in [(0.05, 0.99), (0.0, 0.95)]:
    p_m = (q_hi - q_lo) / 32
    p_lo_e = p_m + q_lo
    p_hi_e = p_m + (1.0 - q_hi)
    h_expected = -(30 * p_m * math.log(p_m)
                   + p_lo_e * math.log(p_lo_e) + p_hi_e * math.log(p_hi_e))
    assert abs(30 * p_m + p_lo_e + p_hi_e - 1.0) < 1e-12
    h = hist_entropy_quantile(vals, bins=32, q_lo=q_lo, q_hi=q_hi)
    check(f"quantile: uniform, asymmetric q=({q_lo},{q_hi}) -> analytic",
          abs(h - h_expected) / h_expected < 1e-2,
          f"H={h:.5f}, expected {h_expected:.5f}")

# discrete attainability under DEFAULT winsorization: with all mass on B
# exact levels, the sample quantiles land on the extreme levels, nothing is
# clipped, and H = log(B) exactly — so the "H* < log B" ceiling applies to
# uniform-DENSITY data, not to discrete-valued data (D6 scope fix).
disc = torch.arange(32, dtype=torch.float).repeat_interleave(100)
h = hist_entropy_quantile(disc, bins=32)  # default q=0.01/0.99
check("quantile: 32 discrete levels, default clipping -> log(32) exactly",
      abs(h - math.log(32)) < 1e-5, f"H={h:.5f}")

# KL constant convention vs N(0,1) ground truth
g = torch.Generator().manual_seed(0)
h_true = 0.5 * math.log(2 * math.pi * math.e)
hs = [knn_entropy_kl(torch.randn(4000, generator=torch.Generator().manual_seed(s)), k=3)
      for s in range(5)]
h_mean = sum(hs) / len(hs)
check("knn: constant convention vs N(0,1)",
      abs(h_mean - h_true) < 0.05, f"H={h_mean:.4f}, true={h_true:.4f}")

# KL failure mode on duplicates (documented, not fixed here)
h = knn_entropy_kl(torch.full((100,), 1.0), k=3)
check("knn: degenerate -> -inf (documented failure)",
      h == float("-inf"), f"H={h}")

print("\n" + ("ALL PASS" if not failures else f"FAILURES: {failures}"))
raise SystemExit(0 if not failures else 1)
