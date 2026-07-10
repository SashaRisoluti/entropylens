"""EXP3 (Q3 / Difetto 3 / A5): negative-handling strategies.

Part 1 — anchor: reproduce the EXACT numbers documented in
02_MATH_FOUNDATIONS.md for the vector [5, -5, 0.01 x 6]:
    shift -> 1.9064, abs -> 0.7362, relu -> 0.0856
This proves our implementation matches the one that produced the
documented evidence, so the defect analysis transfers.

Part 2 — quantitative motivation for the activation->strategy table:
for each activation, measure the fraction of total |output| mass that is
negative under N(0,1) and N(0,3) inputs. Rationale:
  - ReLU family: 0% negative mass -> strategy is a no-op ("relu" safe).
  - GELU/SiLU: small bounded negative tail -> "relu" discards little
    mass (quantified here); negatives mean "weakly inhibited" ~ off.
  - Tanh/Softsign: ~50% negative mass, sign = direction -> "abs" is the
    only strategy that does not destroy or misread half the signal.
  - Identity (raw Linear output): ~50% negative mass, sign semantics
    unknown -> NO safe default exists; library must require an explicit
    choice (documented fallback per C2).

Part 3 — zero-row limit case (A2): all-zero row -> NaN + flag.
"""

import torch

from common import per_example_entropy

failures = []


def check(name, cond, detail):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not cond:
        failures.append(name)


# ------------------------------------------------- Part 1: doc anchors
x = torch.tensor([[5.0, -5.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]])
expected = {"shift": 1.9064, "abs": 0.7362, "relu": 0.0856}
print("=== Difetto 3 anchor (02_MATH_FOUNDATIONS.md) ===")
for strat, exp_h in expected.items():
    h = float(per_example_entropy(x, strat)[0])
    check(f"anchor {strat}", abs(h - exp_h) < 5e-4,
          f"H={h:.4f}, documented {exp_h:.4f}")

# ------------------------------- Part 2: negative-mass quantification
print("\n=== negative |output| mass fraction per activation ===")
acts = {
    "relu": torch.nn.functional.relu,
    "gelu": torch.nn.functional.gelu,
    "silu": torch.nn.functional.silu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "identity": lambda t: t,
}
print(f"{'activation':>10} {'sigma=1':>9} {'sigma=3':>9}")
neg_mass = {}
for name, fn in acts.items():
    row = []
    for sigma in [1.0, 3.0]:
        g = torch.Generator().manual_seed(0)
        z = torch.randn(200_000, generator=g) * sigma
        y = fn(z)
        frac = float(y.clamp_max(0).abs().sum() / y.abs().sum())
        row.append(frac)
    neg_mass[name] = row
    print(f"{name:>10} {row[0] * 100:>8.2f}% {row[1] * 100:>8.2f}%")

check("relu family: zero negative mass",
      max(neg_mass["relu"]) == 0.0, f"{neg_mass['relu']}")
# The theoretical basis for mapping GELU/SiLU -> "relu" is NOT that the
# negative mass is negligible (at sigma=1 it is ~15% for GELU), but that
# the negative OUTPUT RANGE is bounded (GELU >= -0.17, SiLU >= -0.28)
# while the positive range is unbounded: sign encodes "nearly off", not
# direction. Consequences tested: (a) output lower bound, (b) negative
# mass fraction DECREASES with input scale. The residual mass discarded
# at small scales is a documented caveat in the A5 table, not hidden.
z_grid = torch.linspace(-20.0, 20.0, 2_000_001)
gelu_min = float(torch.nn.functional.gelu(z_grid).min())
silu_min = float(torch.nn.functional.silu(z_grid).min())
check("gelu output bounded below (>= -0.18)", gelu_min >= -0.18,
      f"min={gelu_min:.4f}")
check("silu output bounded below (>= -0.30)", silu_min >= -0.30,
      f"min={silu_min:.4f}")
check("gelu/silu negative mass decreases with input scale",
      neg_mass["gelu"][1] < neg_mass["gelu"][0]
      and neg_mass["silu"][1] < neg_mass["silu"][0],
      f"gelu {neg_mass['gelu']}, silu {neg_mass['silu']}")
check("tanh: negative mass ~50% (sign is direction)",
      abs(neg_mass["tanh"][0] - 0.5) < 0.02, f"{neg_mass['tanh']}")
check("identity: negative mass ~50% (no safe default)",
      abs(neg_mass["identity"][0] - 0.5) < 0.02, f"{neg_mass['identity']}")
check("sigmoid: zero negative mass (no handling needed)",
      max(neg_mass["sigmoid"]) == 0.0, f"{neg_mass['sigmoid']}")

# --------------------------------------- Part 3: zero-row limit (A2)
print("\n=== zero-row limit case ===")
xz = torch.zeros(2, 8)
xz[1, 3] = 1.0  # second row one-hot, first row all-zero
for strat in ["relu", "abs", "shift"]:
    h = per_example_entropy(xz, strat)
    check(f"zero row -> NaN ({strat})", bool(torch.isnan(h[0])),
          f"H[0]={float(h[0])}")
h = per_example_entropy(xz, "relu")
check("one-hot row -> 0", abs(float(h[1])) < 1e-7, f"H[1]={float(h[1])}")

# note: under "shift", a one-hot POSITIVE row stays one-hot (min=0);
# uniform row becomes all-zero after shift -> NaN. Demonstrate:
xu = torch.full((1, 8), 2.5)
h = per_example_entropy(xu, "shift")
check("shift: constant row -> NaN (shift annihilates constant rows)",
      bool(torch.isnan(h[0])), f"H={float(h[0])} "
      "(one more reason shift is not a sound default, Difetto 3)")

print("\n" + ("ALL PASS" if not failures else f"FAILURES: {failures}"))
raise SystemExit(0 if not failures else 1)
