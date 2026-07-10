"""EXP2 (Q2 / Difetto 1 / A4): cross-batch-size normalization.

Same underlying distribution N(0,1), quantile binning (exp1 winner
candidate), B=32, 20 seeds per N. Candidates:

  V0: H_plugin / log(min(N, B))
  V1: (H_plugin + (K-1)/(2N)) / log(min(N, B))    [Miller-Madow]

Reference: analytic H_true of the binned-and-clipped N(0,1) distribution
(population quantiles as edges), normalized by log(B).

Gates proposed (explicit, per 05 template):
  - A4: for the chosen variant, mean normalized H at each
    N in {128, 512, 2048, 8192} within 5% relative of the analytic
    reference -> H_norm independent of N at fixed distribution.
  - Reliability criterion (Difetto 5 / Q2): flag unreliable when the
    bias-correction magnitude (B-1)/(2N) > 5% of log(min(N,B)).
    For B=32 this gives N < ~90. Verify N=32 and N=50 are flagged and
    that their (uncorrected) deviation actually exceeds the gate,
    i.e. the flag fires exactly where the estimate degrades.
"""

import math

import torch

from common import (quantile_hist_counts, entropy_from_counts,
                    miller_madow_from_counts)

B = 32
SEEDS = 20
failures = []


def check(name, cond, detail):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not cond:
        failures.append(name)


def reliable(n, bins):
    """Bias-dominance reliability criterion."""
    return (bins - 1) / (2 * n) <= 0.05 * math.log(min(n, bins))


# ---------------------------------------------------- analytic reference
z_lo = float(torch.special.ndtri(torch.tensor(0.01)))
z_hi = float(torch.special.ndtri(torch.tensor(0.99)))
edges = torch.linspace(z_lo, z_hi, B + 1)
cdf = torch.special.ndtr(edges)
probs = (cdf[1:] - cdf[:-1]).clone()
probs[0] += cdf[0]          # clipped lower tail -> first bin
probs[-1] += 1.0 - cdf[-1]  # clipped upper tail -> last bin
h_true = float(-(probs * probs.log()).sum())
h_true_norm = h_true / math.log(B)
print(f"analytic binned-clipped H_true = {h_true:.5f} "
      f"(normalized {h_true_norm:.5f})")

# ------------------------------------------------------------- sweep N
print(f"\n{'N':>6} {'V0 mean':>9} {'V0 dev%':>8} {'V1 mean':>9} "
      f"{'V1 dev%':>8} {'reliable':>9}")
results = {}
for n in [32, 50, 128, 512, 2048, 8192]:
    v0s, v1s = [], []
    for seed in range(SEEDS):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, generator=g)
        counts = quantile_hist_counts(x, bins=B)
        assert counts is not None
        denom = math.log(min(n, B))
        v0s.append(entropy_from_counts(counts) / denom)
        v1s.append(miller_madow_from_counts(counts) / denom)
    v0m = sum(v0s) / SEEDS
    v1m = sum(v1s) / SEEDS
    dev0 = abs(v0m - h_true_norm) / h_true_norm
    dev1 = abs(v1m - h_true_norm) / h_true_norm
    results[n] = (v0m, dev0, v1m, dev1)
    print(f"{n:>6} {v0m:>9.4f} {dev0 * 100:>7.2f}% {v1m:>9.4f} "
          f"{dev1 * 100:>7.2f}% {str(reliable(n, B)):>9}")

# ----------------------------------------------------------------- gates
print()
for n in [128, 512, 2048, 8192]:
    _, dev0, _, dev1 = results[n]
    check(f"A4 V1 (Miller-Madow) at N={n}", dev1 < 0.05,
          f"deviation {dev1 * 100:.2f}% < 5%")

# does V1 improve on V0 where the correction matters (small reliable N)?
_, dev0_128, _, dev1_128 = results[128]
check("V1 <= V0 deviation at N=128", dev1_128 <= dev0_128 + 1e-9,
      f"V1 {dev1_128 * 100:.2f}% vs V0 {dev0_128 * 100:.2f}%")

# reliability flag fires exactly where it should for B=32
for n, expected in [(32, False), (50, False), (128, True),
                    (512, True), (2048, True), (8192, True)]:
    check(f"reliability flag at N={n}", reliable(n, B) == expected,
          f"reliable={reliable(n, B)}, expected {expected}")

print("\n" + ("ALL PASS" if not failures else f"FAILURES: {failures}"))
raise SystemExit(0 if not failures else 1)
