"""EXP4 (Q5 / C4, user decision: IN SCOPE v1): uncertainty method choice.

*** SUPERSEDED FOR THE D4 DECISION by exp4b_pipeline_uncertainty.py. ***
This experiment tests a strictly EASIER problem than the real pipeline:
categorical data with FIXED bins. Adversarial verification showed the
closed-form winner here fails on the full pipeline (edges re-estimated
from the sample): coverage 62.5-88% vs the >=85% gate. exp4 remains valid
as evidence that (a) percentile/basic bootstrap collapse at the entropy
ceiling and (b) all methods are blind on degenerate samples — but the
adopted method is the one selected by exp4b (full-pipeline bootstrap,
normal interval).

Two candidate methods for the std error / CI of batch-wise entropy:

  A. Closed-form (Basharin 1959 delta-method):
     Var(H_hat) ~= (1/N)(sum p_k log^2 p_k - H^2), CI = H_MM +- 1.96 s.
     Known degeneracy: the first-order term VANISHES for uniform and
     one-point distributions, so the CI collapses exactly where honest
     uncertainty is most needed.
  B. Nonparametric bootstrap (R=200 multinomial resamples of the
     empirical distribution, Miller-Madow per replicate,
     percentile CI [2.5%, 97.5%]).

Protocol: categorical distributions with KNOWN true entropy.
  - "skewed":  p_k proportional to 0.8^k over 32 categories
  - "uniform": p_k = 1/32                       (delta-method blind spot)
  Sample N in {50, 200, 1000}, 200 replications; measure 95% CI COVERAGE.

Gate (proposed): chosen method achieves coverage >= 85% on ALL cells with
N >= 200. Values reported, not just pass/fail.

Also demonstrates (documented, drives the Difetto-5 flag): both methods
report ZERO width on a degenerate one-category sample -> uncertainty
machinery cannot replace the reliability flag.
"""

import math

import torch

from common import (entropy_from_counts, miller_madow_from_counts,
                    basharin_stderr_from_counts)

K = 32
REPS = 200
R_BOOT = 200
failures = []


def check(name, cond, detail):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not cond:
        failures.append(name)


def counts_from_sample(idx, k=K):
    return torch.bincount(idx, minlength=k).float()


def mm_rows(counts):
    """Vectorized Miller-Madow entropy per row. counts: (r, K) -> (r,)."""
    n = counts.sum(1)
    p = counts / n.unsqueeze(1)
    plogp = torch.where(p > 0, p * p.log(), torch.zeros_like(p))
    h = -plogp.sum(1)
    k = (counts > 0).sum(1).float()
    return h + (k - 1.0) / (2.0 * n)


def bootstrap_cis(counts, r=R_BOOT, gen=None):
    """Returns (percentile CI, basic CI). The basic/reversed CI
    [2*H_hat - q97.5, 2*H_hat - q2.5] corrects for the bias direction that
    sinks the percentile CI when the estimator sits near a boundary."""
    n = int(counts.sum())
    p_hat = counts / counts.sum()
    idx = torch.multinomial(p_hat.expand(r, -1), n, replacement=True,
                            generator=gen)
    boot_counts = torch.zeros(r, K).scatter_add_(
        1, idx, torch.ones_like(idx, dtype=torch.float))
    hs = mm_rows(boot_counts)
    q_lo, q_hi = (float(torch.quantile(hs, 0.025)),
                  float(torch.quantile(hs, 0.975)))
    h_hat = float(mm_rows(counts.unsqueeze(0))[0])
    return (q_lo, q_hi), (2 * h_hat - q_hi, 2 * h_hat - q_lo)


dists = {
    "skewed": (0.8 ** torch.arange(K, dtype=torch.float)),
    "uniform": torch.ones(K),
}

print(f"{'dist':>8} {'N':>6} {'closed-form':>12} {'boot-perc':>10} "
      f"{'boot-basic':>11}")
coverage = {}
for dname, w in dists.items():
    p = w / w.sum()
    h_true = float(-(p * p.log()).sum())
    for n in [50, 200, 1000]:
        cov_a = cov_b = cov_c = 0
        for rep in range(REPS):
            g = torch.Generator().manual_seed(rep)
            idx = torch.multinomial(p, n, replacement=True, generator=g)
            counts = counts_from_sample(idx)
            # A: closed-form around the Miller-Madow point estimate
            h_mm = miller_madow_from_counts(counts)
            s = basharin_stderr_from_counts(counts)
            cov_a += (abs(h_true - h_mm) <= 1.96 * s)
            # B/C: bootstrap percentile and basic CIs
            (plo, phi), (blo, bhi) = bootstrap_cis(counts, gen=g)
            cov_b += (plo <= h_true <= phi)
            cov_c += (blo <= h_true <= bhi)
        coverage[(dname, n)] = (cov_a / REPS, cov_b / REPS, cov_c / REPS)
        print(f"{dname:>8} {n:>6} {cov_a / REPS * 100:>11.1f}% "
              f"{cov_b / REPS * 100:>9.1f}% {cov_c / REPS * 100:>10.1f}%")

print()
# Gate: the CHOSEN method must reach >= 85% on every cell with N >= 200.
# (N=50 sits below the exp2 reliability threshold N~90 for B=32: the
# library flags those estimates as unreliable anyway.)
methods = {"closed-form": 0, "boot-percentile": 1, "boot-basic": 2}
passing = []
for mname, i in methods.items():
    covs = [coverage[(d, n)][i] for d in dists for n in [200, 1000]]
    ok = all(c >= 0.85 for c in covs)
    print(f"  {mname:16s} min coverage (N>=200) = {min(covs) * 100:.1f}% "
          f"-> {'eligible' if ok else 'rejected'}")
    if ok:
        passing.append(mname)
check("at least one method passes the 85% gate on all N>=200 cells",
      len(passing) > 0, f"eligible: {passing}")

# degenerate sample: both methods claim certainty -> flag is irreplaceable
counts = torch.zeros(K)
counts[0] = 500.0
s = basharin_stderr_from_counts(counts)
(lo, hi), _ = bootstrap_cis(counts, gen=torch.Generator().manual_seed(0))
check("degenerate: closed-form stderr = 0", s == 0.0, f"s={s}")
check("degenerate: bootstrap CI width = 0", hi - lo == 0.0,
      f"width={hi - lo}")

print("\n" + ("ALL PASS" if not failures else f"FAILURES: {failures}"))
raise SystemExit(0 if not failures else 1)
