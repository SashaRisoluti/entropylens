"""EXP5 (A1 / A2): bound attainment for both entropies.

A1 — batch-wise (quantile-binned histogram), claim:
      0 <= H <= log(min(N, B)),
   with H = 0 iff all mass in one bin, and the upper bound attained iff
   mass is uniform over min(N,B) bins (requires B | N when N > B; each
   sample in its own bin when N <= B). Proof sketch recorded in
   MATH_DECISIONS.md; here every attainment condition is exercised.

A2 — per-example, claim: 0 <= H <= log(M); H = 0 iff one-hot;
   H = log(M) iff uniform; all-zero row -> NaN (undefined distribution).
"""

import math

import torch

from common import hist_entropy_quantile, per_example_entropy

failures = []


def check(name, cond, detail):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not cond:
        failures.append(name)


print("=== A1: batch-wise bounds ===")

# N > B, uniform over bins, B | N -> H = log(B) is NOT attainable exactly
# under quantile clipping (2% mass folded into edge bins); the attainable
# max is the analytic clipped value (exp1). Here we verify the INEQUALITY
# and the low end; exact attainment under clipping is covered in exp1.
g = torch.Generator().manual_seed(0)
for n in [8, 32, 100, 1000]:
    x = torch.randn(n, generator=g)
    h = hist_entropy_quantile(x, bins=32)
    bound = math.log(min(n, 32))
    check(f"0 <= H <= log(min(N,B)) at N={n}",
          -1e-9 <= h <= bound + 1e-6, f"H={h:.4f}, bound={bound:.4f}")

check("degenerate -> H=0 exactly",
      hist_entropy_quantile(torch.full((500,), -7.0), bins=32) == 0.0, "")
check("N=1 -> H=0 (degenerate by construction)",
      hist_entropy_quantile(torch.tensor([0.3]), bins=32) == 0.0, "")

# N <= B with all-distinct spread values -> each sample its own bin,
# H = log(N) exactly (upper bound attained in the N < B regime).
# Use q_lo=0, q_hi=1 to remove clipping from this attainment check.
x = torch.linspace(0.0, 1.0, 8)
h = hist_entropy_quantile(x, bins=32, q_lo=0.0, q_hi=1.0)
check("N=8 <= B=32, distinct spread values -> H = log(8)",
      abs(h - math.log(8)) < 1e-5, f"H={h:.5f}, log(8)={math.log(8):.5f}")

# N > B, B | N, exactly uniform over bins, no clipping -> H = log(B):
# ten copies of each integer 0..31 -> exactly 10 samples per bin.
x = torch.arange(32, dtype=torch.float).repeat_interleave(10)
h = hist_entropy_quantile(x, bins=32, q_lo=0.0, q_hi=1.0)
check("N=320, B=32, uniform, no clipping -> H = log(32)",
      abs(h - math.log(32)) < 1e-5, f"H={h:.5f}")

print("\n=== A2: per-example bounds ===")
M = 16
one_hot = torch.zeros(1, M)
one_hot[0, 5] = 7.3
h = float(per_example_entropy(one_hot, "relu")[0])
check("one-hot -> H=0", abs(h) < 1e-7, f"H={h}")

uniform = torch.full((1, M), 0.25)
h = float(per_example_entropy(uniform, "relu")[0])
check("uniform -> H=log(M)", abs(h - math.log(M)) < 1e-5,
      f"H={h:.5f}, log({M})={math.log(M):.5f}")

zero = torch.zeros(1, M)
h = per_example_entropy(zero, "relu")
check("all-zero row -> NaN", bool(torch.isnan(h[0])), f"H={float(h[0])}")

# inequality on random data, all three strategies
g = torch.Generator().manual_seed(1)
x = torch.randn(64, M, generator=g)
for strat in ["relu", "abs", "shift"]:
    h = per_example_entropy(x, strat)
    finite = h[~torch.isnan(h)]
    check(f"0 <= H <= log(M) for all rows ({strat})",
          bool((finite >= -1e-9).all() and
               (finite <= math.log(M) + 1e-6).all()),
          f"min={float(finite.min()):.4f}, max={float(finite.max()):.4f}")

print("\n" + ("ALL PASS" if not failures else f"FAILURES: {failures}"))
raise SystemExit(0 if not failures else 1)
