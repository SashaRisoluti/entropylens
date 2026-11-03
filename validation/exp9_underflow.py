"""EXP9 (A2 regression / review finding M1): per-example row-sum underflow.

Defect. entropylens/core.py normalises each row by `s.clamp(min=1e-38)`
(core_np.py: `np.maximum(s, 1e-38)`). When the post-strategy row sum falls
below 1e-38 the row is divided by 1e-38 instead of by its own sum, so `p`
no longer sums to 1 and H collapses towards 0 while `valid` stays True.
A UNIFORM row — the maximum-entropy case — reads as near-one-hot.

This is the same failure class that removed `shift` in D3 ("a strategy
that turns the maximum-entropy case into an undefined one does not
measure what it claims"), so the same standard applies to the surviving
path. Note the clamp does NOT come from the validated formula: the
pre-implementation reference in common.py:147 divides by `s` directly.
It was introduced during implementation and no gate probed subnormal
rows, which is why exp9 exists.

Gates (declared before the sweep, per 05_ITERATION_LOOP.md):
  G1 uniform row of M=8 at element scale 1e-30 ... 1e-44:
     H = log(M) within 1e-5, valid=True, for BOTH strategies.
  G2 true zero row -> NaN + valid=False (D5 convention preserved).
  G3 D3 anchors unmoved: abs -> 0.7362, relu -> 0.0856 within 5e-4.
  G4 one-hot row -> H = 0 within 1e-7, valid=True (A2).
  G5 no NaN/Inf leaks into rows marked valid, on a mixed-scale battery.
  G6 the CURRENT formula must FAIL G1 — proof the gate captures the
     defect rather than passing vacuously (same discipline as exp1's
     min-max baseline).

Candidates (only the winner is implemented; 05 step 4):
  C0 current   p = a / max(s, 1e-38)                      [baseline]
  C1 no-clamp  p = a / s                                   [= common.py]
  C2 rescale   a <- a / rowmax, then p = a / s
  C3 float64   reduce in float64, cast back to float32
"""

import math

import torch

M = 8
LOG_M = math.log(M)
SCALES = [1e-30, 1e-36, 1e-38, 1e-39, 1e-40, 1e-42, 1e-44]
ANCHOR = torch.tensor([[5.0, -5.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]])
failures = []


def check(name, cond, detail):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not cond:
        failures.append(name)


def _entropy_from_p(p, valid):
    plogp = torch.where(p > 0, p * p.log(), torch.zeros_like(p))
    h = -plogp.sum(1)
    return torch.where(valid, h, torch.full_like(h, float("nan")))


def c0_current(a):
    s = a.sum(1)
    valid = s > 0
    p = torch.where(valid.unsqueeze(1),
                    a / s.clamp(min=1e-38).unsqueeze(1), torch.zeros_like(a))
    return _entropy_from_p(p, valid), valid


def c1_no_clamp(a):
    s = a.sum(1)
    valid = s > 0
    p = torch.where(valid.unsqueeze(1), a / s.unsqueeze(1),
                    torch.zeros_like(a))
    return _entropy_from_p(p, valid), valid


def c2_rescale(a):
    mx = a.max(1, keepdim=True).values
    a = torch.where(mx > 0, a / mx, a)   # H is invariant to positive scaling
    s = a.sum(1)
    valid = s > 0
    p = torch.where(valid.unsqueeze(1), a / s.unsqueeze(1),
                    torch.zeros_like(a))
    return _entropy_from_p(p, valid), valid


def c3_float64(a):
    a64 = a.double()
    s = a64.sum(1)
    valid = s > 0
    p = torch.where(valid.unsqueeze(1), a64 / s.unsqueeze(1),
                    torch.zeros_like(a64))
    return _entropy_from_p(p, valid).float(), valid


CANDIDATES = [("C0 current", c0_current), ("C1 no-clamp", c1_no_clamp),
              ("C2 rescale", c2_rescale), ("C3 float64", c3_float64)]
STRATEGIES = {"relu": lambda x: x.clamp_min(0.0), "abs": lambda x: x.abs()}


# ------------------------------------------------- G1: uniform-row sweep
print("=== G1: uniform row (true H = log 8 = %.6f) ===" % LOG_M)
print(f"{'element scale':>14} " + " ".join(f"{n:>12}" for n, _ in CANDIDATES))
g1_ok = {name: True for name, _ in CANDIDATES}
for scale in SCALES:
    row = []
    for name, fn in CANDIDATES:
        worst = 0.0
        for strat, pre in STRATEGIES.items():
            h, valid = fn(pre(torch.full((1, M), scale)))
            err = abs(float(h[0]) - LOG_M)   # NaN must fail, not vanish in max()
            worst = max(worst, err) if math.isfinite(err) else math.inf
            if not bool(valid[0]):
                worst = float("inf")
        g1_ok[name] &= worst < 1e-5
        row.append(f"{worst:>12.2e}")
    print(f"{scale:>14.0e} " + " ".join(row))

for name, _ in CANDIDATES:
    if name.startswith("C0"):
        continue
    check(f"G1 {name}", g1_ok[name], "max |H - log M| < 1e-5 over all scales")

# G6: the defect must be visible, or the gate proves nothing
check("G6 baseline C0 reproduces the defect", not g1_ok["C0 current"],
      "current formula FAILS G1 as expected (review finding M1)")
h_c0, v_c0 = c0_current(torch.full((1, M), 1e-40).abs())
print(f"     C0 at 1e-40: H={float(h_c0[0]):.6f} valid={bool(v_c0[0])} "
      f"(true {LOG_M:.6f}) -> reads as near-one-hot")

# ------------------------------------------------------ G2 / G4: A2 cases
print("\n=== G2 zero row / G4 one-hot (A2 conventions) ===")
zero = torch.zeros(1, M)
one_hot = torch.zeros(1, M)
one_hot[0, 5] = 7.3
for name, fn in CANDIDATES:
    if name.startswith("C0"):
        continue
    h, valid = fn(zero)
    check(f"G2 {name}: zero row -> NaN + invalid",
          bool(torch.isnan(h[0])) and not bool(valid[0]),
          f"H={float(h[0])}, valid={bool(valid[0])}")
    h, valid = fn(one_hot)
    check(f"G4 {name}: one-hot -> 0", abs(float(h[0])) < 1e-7
          and bool(valid[0]), f"H={float(h[0]):.2e}")

# ------------------------------------------------------- G3: D3 anchors
print("\n=== G3: Difetto 3 anchors must not move ===")
for name, fn in CANDIDATES:
    if name.startswith("C0"):
        continue
    for strat, expected in [("abs", 0.7362), ("relu", 0.0856)]:
        h, _ = fn(STRATEGIES[strat](ANCHOR))
        check(f"G3 {name} {strat}", abs(float(h[0]) - expected) < 5e-4,
              f"H={float(h[0]):.4f}, documented {expected:.4f}")

# ------------------------------------------- G5: mixed-scale battery
print("\n=== G5: mixed-scale battery, no NaN/Inf in valid rows ===")
g = torch.Generator().manual_seed(9)
battery = torch.cat([
    torch.randn(32, M, generator=g),
    torch.randn(32, M, generator=g) * 1e-38,
    torch.randn(32, M, generator=g) * 1e20,
    torch.zeros(4, M),
])
for name, fn in CANDIDATES:
    if name.startswith("C0"):
        continue
    bad = 0
    for strat, pre in STRATEGIES.items():
        h, valid = fn(pre(battery))
        hv = h[valid]
        bad += int((~torch.isfinite(hv)).sum())
        bad += int((hv < -1e-6).sum()) + int((hv > LOG_M + 1e-5).sum())
    check(f"G5 {name}", bad == 0,
          f"{bad} valid rows with non-finite or out-of-bound H "
          f"(both strategies, {battery.shape[0]} rows each)")

print("\n" + ("ALL PASS" if not failures else f"FAILURES: {failures}"))
raise SystemExit(0 if not failures else 1)
