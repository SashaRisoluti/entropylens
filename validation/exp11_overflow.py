"""EXP11 (A1 / A2 / E1, follow-up of D10/D11): float32 overflow INSIDE the
computation, on finite float32 input.

Defect. D11 rejects values outside float32's range at the cast. Values
inside the range still overflow later, in three places:

  per-example  s = a.sum(1) -> inf for a large row; p = a / inf = 0
               -> H = 0 with valid = True
  batch-wise   hi - lo and xc - lo -> inf; (xc - lo) / width = NaN; the
               NaN bin index lands in bin 0 for every sample
               -> H = 0 with reliable = True
  batch-wise   the quantile lerp a + w * (b - a): b - a = inf when the two
               order statistics straddle zero near the max -> lo = +inf,
               the column is flagged degenerate (honest flag, wrong answer)

Same false-specialisation class as Difetto 2, D10 and D11.

Decision (user, 2026-11-02): option 1 of three presented — sums, quantile
interpolation and bin-index arithmetic in float64 internally. Rejected:
per-row / per-column max rescale in float32 (more code, same gates);
rejecting at validation any input whose intermediates could overflow (a
second contract narrowing after D11, with a conservative bound). float64
holds the difference of any two float32 values and the sum of up to ~1e270
of them, so no finite float32 input can overflow these intermediates; it
also turns D10's subnormal row sums into normal numbers. Cost: transient
float64 copies of the activations (~2x memory on those intermediates).

Gates (declared before the sweep, per 05_ITERATION_LOOP.md):
  G1 per-example uniform rows near float32's max (1e37, 1e38, 3e38, and a
     sign-alternating +-3e38 row): H = log(M_eff) within 1e-5 and valid,
     both strategies.
  G2 batch-wise exact scale invariance: integer columns times 2**k,
     k in {60, 100, 118}: entropy and reliable flag BITWISE equal to k=0.
     Power-of-two scaling is exact, so any change is the defect.
  G3 quantile straddle: 2 x -1.75 and 126 x +1.75, times 2**127: H and
     reliable bitwise equal to the unscaled column.
  G4 no regression on ordinary input, C1 vs C0: per-example |dH| <= 1e-6;
     batch-wise counts change only by edge flips — at most 1% of columns
     touched and at most 2 samples moved in any touched column (a sample
     within float32 rounding of a bin edge: its float32 bin was itself a
     rounding artefact). D3 anchors (0.7362 / 0.0856) and the D10
     subnormal uniform row (log 8) unmoved.
  G5 the CURRENT float32 formula (C0) must FAIL G1, G2 and G3 — proof the
     gates capture the defect (exp1/exp9/exp10 discipline).
  G6 library: both cores pass G1-G3 through the public API, agree with each
     other (counts bitwise, |dH| <= 1e-6), and confidence_interval keeps
     lo <= estimate <= hi with a non-zero width on the G2 input.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from entropylens import core, core_np

BINS, Q_LO, Q_HI = 32, 0.01, 0.99
ANCHOR = torch.tensor([[5.0, -5.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]])
STRATEGIES = {"relu": lambda x: x.clamp_min(0.0), "abs": lambda x: x.abs()}
failures = []


def check(name, cond, detail):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not cond:
        failures.append(name)


# ---------------------------------------------------------------- candidates
# C0 = current library arithmetic (float32), C1 = the fix (float64). Both
# share everything downstream (Miller-Madow, reliability) with the library.
def per_example(a, dtype):
    a = a.to(dtype)
    s = a.sum(1)
    valid = s > 0
    p = torch.where(valid.unsqueeze(1), a / s.unsqueeze(1),
                    torch.zeros_like(a)).float()
    plogp = torch.where(p > 0, p * p.log(), torch.zeros_like(p))
    h = -plogp.sum(1)
    return torch.where(valid, h, torch.full_like(h, float("nan"))), valid


def hist_counts(x, dtype):
    x = x.to(dtype)
    n, m = x.shape
    lo = torch.quantile(x, Q_LO, dim=0)
    hi = torch.quantile(x, Q_HI, dim=0)
    degenerate = hi <= lo
    width = torch.where(degenerate, torch.ones_like(hi), hi - lo)
    idx = ((x.clamp(lo, hi) - lo) / width * BINS).long().clamp_(0, BINS - 1)
    counts = torch.zeros(m, BINS).scatter_add_(1, idx.t(), torch.ones(m, n))
    counts[degenerate] = 0.0
    return counts, degenerate


def batch_wise(x, dtype):
    counts, degenerate = hist_counts(x, dtype)
    n = x.shape[0]
    h = core._miller_madow(counts, n, degenerate)
    return h, core.reliability_threshold(n, BINS) & ~degenerate


CANDIDATES = [("C0 float32", torch.float32), ("C1 float64", torch.float64)]

# ------------------------------------------------------------ gate inputs
G1_ROWS = {
    "1e37": torch.full((1, 8), 1e37),
    "1e38": torch.full((1, 8), 1e38),
    "3e38": torch.full((1, 8), 3e38),
    "+-3e38": torch.tensor([[3e38, -3e38] * 4]),
}


def m_eff(row, strategy):
    return int((STRATEGIES[strategy](row) > 0).sum())


g = torch.Generator().manual_seed(11)
INTEGERS = torch.randint(-1000, 1001, (512, 3), generator=g).float()
STRADDLE = torch.cat([torch.full((2,), -1.75), torch.full((126,), 1.75)])[:, None]

# ------------------------------------------------------------- G1 / G2 / G3
ok = {name: {"G1": True, "G2": True, "G3": True} for name, _ in CANDIDATES}
print("=== G1: per-example uniform rows near float32 max, |H - log M_eff| ===")
for name, dtype in CANDIDATES:
    for label, row in G1_ROWS.items():
        for strat, pre in STRATEGIES.items():
            h, valid = per_example(pre(row), dtype)
            err = abs(float(h[0]) - math.log(m_eff(row, strat)))
            good = bool(valid[0]) and err < 1e-5
            ok[name]["G1"] &= good
            if not good:
                print(f"     {name} {label:>6} {strat}: H={float(h[0]):.6f} "
                      f"valid={bool(valid[0])} err={err:.3e}")

print("\n=== G2: integer columns x 2**k, bitwise vs k=0 ===")
for name, dtype in CANDIDATES:
    h0, r0 = batch_wise(INTEGERS, dtype)
    for k in [60, 100, 118]:
        h, r = batch_wise(INTEGERS * 2.0 ** k, dtype)
        good = torch.equal(h, h0) and torch.equal(r, r0)
        ok[name]["G2"] &= good
        print(f"     {name} k={k:>3}: H={[round(float(v), 4) for v in h]} "
              f"(k=0: {[round(float(v), 4) for v in h0]}) "
              f"reliable={r.tolist()} {'ok' if good else 'MOVED'}")

print("\n=== G3: quantile straddle x 2**127, bitwise vs unscaled ===")
for name, dtype in CANDIDATES:
    h0, r0 = batch_wise(STRADDLE, dtype)
    h, r = batch_wise(STRADDLE * 2.0 ** 127, dtype)
    good = torch.equal(h, h0) and torch.equal(r, r0)
    ok[name]["G3"] &= good
    print(f"     {name}: H={float(h[0]):.6f} reliable={bool(r[0])} "
          f"(unscaled H={float(h0[0]):.6f} reliable={bool(r0[0])})")

for gate in ["G1", "G2", "G3"]:
    check(f"{gate} C1 float64", ok["C1 float64"][gate], "passes")
check("G5 baseline C0 reproduces the defect",
      not any(ok["C0 float32"].values()),
      f"C0 fails G1/G2/G3: {[k for k, v in ok['C0 float32'].items() if not v]}")

# ---------------------------------------------------------------------- G4
print("\n=== G4: no regression on ordinary input (C1 vs C0) ===")
worst_pe, touched, cols, worst_moved = 0.0, 0, 0, 0
for seed in range(20):
    g.manual_seed(seed)
    for x in [torch.randn(512, 64, generator=g),
              torch.randn(512, 64, generator=g).mul(2.0).exp(),
              torch.rand(128, 64, generator=g),
              torch.randn(4096, 16, generator=g)]:
        c32, _ = hist_counts(x, torch.float32)
        c64, _ = hist_counts(x, torch.float64)
        moved = (c32 - c64).abs().sum(1) / 2          # samples moved, per column
        touched += int((moved > 0).sum())
        cols += x.shape[1]
        worst_moved = max(worst_moved, int(moved.max()))
        for pre in STRATEGIES.values():
            h32, v32 = per_example(pre(x), torch.float32)
            h64, v64 = per_example(pre(x), torch.float64)
            assert torch.equal(v32, v64)
            worst_pe = max(worst_pe, float((h32 - h64)[v32].abs().max()))
check("G4 per-example |dH| <= 1e-6", worst_pe <= 1e-6,
      f"max |dH| = {worst_pe:.2e} over 160 batteries x 2 strategies")
check("G4 batch-wise edge flips only", touched <= 0.01 * cols
      and worst_moved <= 2,
      f"{touched}/{cols} columns touched, at most {worst_moved} sample(s) "
      "moved in any column")
for strat, expected in [("abs", 0.7362), ("relu", 0.0856)]:
    h, _ = per_example(STRATEGIES[strat](ANCHOR), torch.float64)
    check(f"G4 D3 anchor {strat}", abs(float(h[0]) - expected) < 5e-4,
          f"H={float(h[0]):.4f}, documented {expected:.4f}")
h, v = per_example(torch.full((1, 8), 1e-40), torch.float64)
check("G4 D10 subnormal uniform row", bool(v[0])
      and abs(float(h[0]) - math.log(8)) < 1e-5, f"H={float(h[0]):.6f}")

# ---------------------------------------------------------------------- G6
print("\n=== G6: library, both cores ===")
for label, row in G1_ROWS.items():
    for strat in STRATEGIES:
        target = math.log(m_eff(row, strat))
        pt = core.per_example_entropy(row, strat)
        pn = core_np.per_example_entropy(row.numpy(), strat)
        check(f"G6 G1 {label} {strat}",
              bool(pt.valid[0]) and bool(pn.valid[0])
              and abs(float(pt.entropy[0]) - target) < 1e-5
              and abs(float(pt.entropy[0]) - float(pn.entropy[0])) <= 1e-6,
              f"torch H={float(pt.entropy[0]):.6f} numpy "
              f"H={float(pn.entropy[0]):.6f} target {target:.6f}")
for label, base, k in [("G2", INTEGERS, 118), ("G3", STRADDLE, 127)]:
    x = base * 2.0 ** k
    ref, bt = core.batch_wise_entropy(base), core.batch_wise_entropy(x)
    bn = core_np.batch_wise_entropy(x.numpy())
    ct, dt = core._quantile_hist_counts(x, BINS, Q_LO, Q_HI)
    cn, dn = core_np._quantile_hist_counts(x.numpy(), BINS, Q_LO, Q_HI)
    check(f"G6 {label} torch bitwise vs unscaled",
          torch.equal(bt.entropy, ref.entropy)
          and torch.equal(bt.reliable, ref.reliable),
          f"H={bt.entropy.tolist()} reliable={bt.reliable.tolist()}")
    check(f"G6 {label} cores agree",
          np.array_equal(ct.numpy(), cn) and np.array_equal(dt.numpy(), dn)
          and float(np.abs(bt.entropy.numpy() - bn.entropy).max()) <= 1e-6,
          "counts bitwise, |dH| <= 1e-6")
est, lo, hi = core.confidence_interval(
    INTEGERS * 2.0 ** 118, generator=torch.Generator().manual_seed(0))
check("G6 confidence_interval on G2 input",
      bool((lo <= est).all() and (est <= hi).all() and (hi > lo).all()),
      f"est={[round(float(v), 4) for v in est]} "
      f"width={[round(float(v), 4) for v in hi - lo]}")

print("\n" + ("ALL PASS" if not failures else f"FAILURES: {failures}"))
raise SystemExit(0 if not failures else 1)
