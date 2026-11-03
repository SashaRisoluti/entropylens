"""EXP10 (A7 / E1 / review finding M5): float64 -> float32 overflow gap.

Defect. entropylens/core.py validates finiteness on the ORIGINAL dtype and
downcasts afterwards:

    if not x.dtype.is_floating_point: x = x.float()
    if not torch.isfinite(x).all(): raise ValueError(...)
    return x.float()                      # <- overflow happens HERE

A float64 value that is finite but larger than float32's max (~3.4028e38)
passes the check and becomes `inf` in the returned tensor. It does not
crash: batch_wise_entropy then reports H = 0.0 with reliable = True, i.e.
"maximally concentrated neuron, no a priori red flag" — the same false
specialisation signal as Difetto 2 and D10.

core_np.py casts BEFORE checking (D7) and rejects the same input, so the
two cores disagree on their input contract. The numpy behaviour is the
correct one; this experiment gates aligning torch to it.

Decision (user, 2026-08-03): option 1 — cast first, then check once. An
original NaN/Inf is still NaN/Inf after the cast, so a single post-cast
check catches both causes; no second pass is needed. Option 3 (check
before AND after, to distinguish "contains NaN/Inf" from "overflows
float32" in the message) was considered and NOT chosen: the extra
reduction on every call was judged not worth the better diagnostic.
Recorded here because the rejected alternative is part of the decision.

Gates (declared before the sweep, per 05_ITERATION_LOOP.md):
  G1 finite-float64-overflowing-float32 input -> ValueError in BOTH cores,
     for all three public entry points (batch_wise_entropy,
     per_example_entropy, confidence_interval).
  G2 no regression on the existing contract: NaN and +-Inf in the input
     are still rejected, empty / 4D / wrong-dim errors unchanged.
  G3 ordinary float64 input (in float32 range) is still ACCEPTED and gives
     the same answer as the equivalent float32 input.
  G4 the float32 path keeps its documented anchors: D3 (0.7362 / 0.0856),
     the D10 subnormal uniform row (1e-40 -> log 8), normalized in bounds.
     (At the D11 commit, 4a6b29f, float32 results were also bitwise
     identical to pre-D11; D12 later changed float32 intermediates on
     purpose, and that change is gated by exp11 G4, not here.)
  G5 the PRE-FIX validation order (frozen below as c0_validate) must FAIL
     G1 — proof the gate captures the defect rather than passing
     vacuously (exp1/exp9 discipline), re-checked on every run.
  G6 torch and numpy cores agree accept/reject on the whole battery.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from entropylens import core, core_np

FLOAT32_MAX = 3.4028234663852886e38
failures = []


def check(name, cond, detail):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not cond:
        failures.append(name)


def _raises(fn, *a, **k):
    """Returns True if fn raised ValueError, False if it returned."""
    try:
        fn(*a, **k)
        return False
    except ValueError:
        return True


# --------------------------------------------- the offending input family
def overflowing(n=128, m=4):
    """Finite in float64, overflows float32 on cast."""
    x = torch.full((n, m), 1e39, dtype=torch.float64)
    x[0, 0] = 1e38            # one in-range value, so it is not uniform
    assert bool(torch.isfinite(x).all()), "input must be finite in float64"
    assert float(x.max()) > FLOAT32_MAX
    return x


print("=== G5: the pre-fix validation order must exhibit the defect ===")


def c0_validate(x):
    """Frozen pre-D11 torch order: finiteness checked on the ORIGINAL dtype,
    downcast afterwards. Kept so G5 re-proves the defect on every run."""
    if not torch.isfinite(x).all():
        raise ValueError("input contains NaN or Inf")
    return x.float()


x = overflowing()
accepted = not _raises(c0_validate, x)
leaks_inf = accepted and not bool(torch.isfinite(c0_validate(x)).all())
check("G5 pre-fix order accepts the G1 input and leaks inf",
      accepted and leaks_inf,
      "C0 passes the finiteness check, then returns inf after the cast")

# ------------------------------------------------------------------ G1
print("\n=== G1: overflowing float64 rejected by both cores ===")
x = overflowing()
xn = x.numpy()
for label, fn, args in [
    ("torch batch_wise", core.batch_wise_entropy, (x,)),
    ("torch per_example", core.per_example_entropy, (x, "abs")),
    ("torch confidence_interval", core.confidence_interval, (x,)),
    ("numpy batch_wise", core_np.batch_wise_entropy, (xn,)),
    ("numpy per_example", core_np.per_example_entropy, (xn, "abs")),
    ("numpy confidence_interval", core_np.confidence_interval, (xn,)),
]:
    check(f"G1 {label}", _raises(fn, *args), "raises ValueError")

# ------------------------------------------------------------------ G2
print("\n=== G2: existing rejections unchanged ===")
nan = torch.tensor([[1.0, float("nan")], [2.0, 3.0]])
inf = torch.tensor([[1.0, float("inf")], [2.0, 3.0]])
ninf = torch.tensor([[1.0, float("-inf")], [2.0, 3.0]])
for label, t in [("NaN", nan), ("+Inf", inf), ("-Inf", ninf)]:
    check(f"G2 torch rejects {label}",
          _raises(core.batch_wise_entropy, t), "raises ValueError")
    check(f"G2 numpy rejects {label}",
          _raises(core_np.batch_wise_entropy, t.numpy()), "raises ValueError")
check("G2 torch rejects empty",
      _raises(core.batch_wise_entropy, torch.empty(0)), "raises ValueError")
check("G2 torch rejects 4D",
      _raises(core.batch_wise_entropy, torch.zeros(2, 2, 2, 2)),
      "raises ValueError")
check("G2 per_example still rejects 1D",
      _raises(core.per_example_entropy, torch.randn(8), "relu"),
      "raises ValueError")

# ------------------------------------------------------------------ G3
print("\n=== G3: ordinary float64 still accepted, same answer as float32 ===")
g = torch.Generator().manual_seed(10)
x32 = torch.randn(256, 6, generator=g)
x64 = x32.double()
b32 = core.batch_wise_entropy(x32)
b64 = core.batch_wise_entropy(x64)
check("G3 float64 accepted", True, "no exception")
check("G3 float64 == float32 (batch-wise entropy)",
      torch.equal(b32.entropy, b64.entropy),
      f"max |d| = {float((b32.entropy - b64.entropy).abs().max()):.2e}")
check("G3 float64 == float32 (normalized)",
      torch.equal(b32.normalized, b64.normalized), "bitwise equal")
check("G3 float64 == float32 (reliable/reason)",
      torch.equal(b32.reliable, b64.reliable)
      and torch.equal(b32.reason, b64.reason), "bitwise equal")
for strat in ["relu", "abs"]:
    p32 = core.per_example_entropy(x32, strat)
    p64 = core.per_example_entropy(x64, strat)
    # NaN-aware: `relu` legitimately produces NaN on all-negative rows
    # (zero mass, D5), and torch.equal is False on NaN by definition —
    # compare the NaN masks and the finite values separately.
    nan32, nan64 = torch.isnan(p32.entropy), torch.isnan(p64.entropy)
    same = (torch.equal(nan32, nan64)
            and torch.equal(p32.entropy[~nan32], p64.entropy[~nan64])
            and torch.equal(p32.valid, p64.valid))
    check(f"G3 per_example float64 == float32 ({strat})", same,
          f"NaN masks identical ({int(nan32.sum())} invalid rows), "
          "finite values bitwise equal")

# large-but-in-range float64 must still pass (boundary, not overflow)
big = torch.full((128, 2), 1e37, dtype=torch.float64)
big[0, 0] = 1e36
check("G3 large in-range float64 accepted",
      not _raises(core.batch_wise_entropy, big),
      "1e37 < float32 max -> no exception")

# ------------------------------------------------------------------ G4
print("\n=== G4: float32 path unaffected (frozen anchors) ===")
anchor = torch.tensor([[5.0, -5.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]])
for strat, expected in [("abs", 0.7362), ("relu", 0.0856)]:
    h = float(core.per_example_entropy(anchor, strat).entropy[0])
    check(f"G4 D3 anchor {strat}", abs(h - expected) < 5e-4,
          f"H={h:.4f}, documented {expected:.4f}")
# 1e-40: the row sum is subnormal, where the pre-D10 clamp read H=0.368
uni = torch.full((1, 8), 1e-40)
h = float(core.per_example_entropy(uni, "relu").entropy[0])
check("G4 subnormal uniform row -> log 8 (D10 regression)",
      abs(h - math.log(8)) < 1e-5, f"H={h:.6f}")
g = torch.Generator().manual_seed(4)
b = core.batch_wise_entropy(torch.randn(512, 4, generator=g))
check("G4 float32 batch-wise in bounds",
      bool((b.normalized >= 0).all() and (b.normalized <= 1).all()),
      f"normalized in [{float(b.normalized.min()):.4f}, "
      f"{float(b.normalized.max()):.4f}]")

# ------------------------------------------------------------------ G6
print("\n=== G6: torch and numpy cores agree accept/reject ===")
battery = {
    "overflowing float64": overflowing(),
    "NaN": nan,
    "+Inf": inf,
    "ordinary float32": x32,
    "ordinary float64": x64,
    "large in-range float64": big,
    "degenerate constant": torch.full((128, 3), 3.14),
}
for label, t in battery.items():
    t_rej = _raises(core.batch_wise_entropy, t)
    n_rej = _raises(core_np.batch_wise_entropy, t.numpy())
    check(f"G6 {label}", t_rej == n_rej,
          f"torch {'reject' if t_rej else 'accept'}, "
          f"numpy {'reject' if n_rej else 'accept'}")

print("\n" + ("ALL PASS" if not failures else f"FAILURES: {failures}"))
raise SystemExit(0 if not failures else 1)
