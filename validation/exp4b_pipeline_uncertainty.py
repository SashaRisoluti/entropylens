"""EXP4b (Q5 / C4 — D4 REOPENED): uncertainty on the FULL pipeline.

Adversarial verification of D4 found a CRITICAL gap: exp4 validated the
closed-form (Basharin) CI on a strictly easier problem — categorical data
with FIXED bins. The real pipeline estimates the bin edges from the same
sample (per-batch quantiles), and that edge re-estimation contributes
variance the Basharin formula cannot see. Verifier evidence: coverage
78.9% / 81.6% / 78.2% at N = 200 / 1000 / 5000 on N(0,1), all below the
85% gate, with claimed-stderr/actual-sd drifting 0.72 -> 0.64 as N grows
(the deficit is asymptotically permanent, not small-sample). Oracle fixed
edges recover 91.9% / 95.8%.

This experiment re-runs the coverage study END TO END: continuous data ->
per-replication quantile edges -> winsorized histogram -> Miller-Madow.

Estimand: the POPULATION binned-clipped entropy H_pop (edges at population
quantiles Q(0.01), Q(0.99), 32 equal bins, tails clamped into edge bins),
computed analytically from each distribution's CDF. The sample estimator
converges to H_pop as N -> inf, so H_pop is the correct coverage target.

Candidates:
  A. closed-form Basharin around H_MM      (current D4 — expected to FAIL,
                                            reproducing the verifier)
  B. full-pipeline bootstrap, percentile   (resample RAW values,
                                            re-estimate edges per replicate)
  C. full-pipeline bootstrap, basic        (2*H_hat - q_hi, 2*H_hat - q_lo)
  D. full-pipeline bootstrap, normal       (H_MM +- 1.96 * sd(replicates))

"Calibrated stderr inflation" is NOT tested: the deficit ratio drifts with
N (0.72 -> 0.64) and depends on the distribution, so no principled
constant exists — rejected on those grounds, documented in D4.

Distributions (analytic H_pop from the CDF):
  - normal N(0,1)      (the verifier's failing case)
  - uniform(0,1)       (binned dist near the log-32 ceiling: the regime
                        that sank the fixed-bin bootstrap in exp4)
  - exponential(1)     (skewed, asymmetric tails)

Grid: N in {200, 1000, 5000}, REPS = 200 replications, R = 200 bootstrap
resamples. Gate: the chosen method reaches coverage >= 85% on EVERY cell
(all three distributions, all N >= 200).

Also reported: claimed-stderr / actual-sd for the closed form, to pin the
mechanism (edge re-estimation variance invisible to Basharin).
"""

import math

import torch

from common import basharin_stderr_from_counts

B = 32
Q_LO, Q_HI = 0.01, 0.99
REPS = 200
R_BOOT = 200
NS = [200, 1000, 5000]
failures = []


def check(name, cond, detail):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not cond:
        failures.append(name)


# ------------------------------------------------- analytic H_pop per dist
def h_pop_from_cdf(cdf, inv_cdf):
    lo, hi = inv_cdf(Q_LO), inv_cdf(Q_HI)
    edges = torch.linspace(lo, hi, B + 1, dtype=torch.float64)
    c = cdf(edges)
    probs = (c[1:] - c[:-1]).clone()
    probs[0] += c[0]           # clamped lower tail -> first bin
    probs[-1] += 1.0 - c[-1]   # clamped upper tail -> last bin
    return float(-(probs * probs.log()).sum())


def sample_normal(n, g):
    return torch.randn(n, generator=g)


def sample_uniform(n, g):
    return torch.rand(n, generator=g)


def sample_exponential(n, g):
    return -(1.0 - torch.rand(n, generator=g)).log()


DISTS = {
    "normal": (
        sample_normal,
        h_pop_from_cdf(lambda e: torch.special.ndtr(e),
                       lambda q: float(torch.special.ndtri(
                           torch.tensor(q, dtype=torch.float64)))),
    ),
    "uniform": (
        sample_uniform,
        h_pop_from_cdf(lambda e: e.clamp(0.0, 1.0), lambda q: q),
    ),
    "exponential": (
        sample_exponential,
        h_pop_from_cdf(lambda e: 1.0 - (-e).exp(),
                       lambda q: -math.log(1.0 - q)),
    ),
}


# ------------------------------------------- vectorized full pipeline
def hist_rows(m):
    """Quantile-winsorized histogram per ROW. m: (R, N) -> counts (R, B).

    Same formula as D1 / plan Task 1: edges at the row's [Q(0.01), Q(0.99)],
    out-of-range values clamped into the edge bins. Asserts no degenerate
    rows (continuous data, N >= 200: quantile collapse is impossible).
    """
    lo = torch.quantile(m, Q_LO, dim=1, keepdim=True)
    hi = torch.quantile(m, Q_HI, dim=1, keepdim=True)
    assert bool((hi > lo).all()), "degenerate quantile range in bootstrap"
    xc = m.clamp(lo, hi)
    idx = ((xc - lo) / (hi - lo) * B).long().clamp_(0, B - 1)
    return torch.zeros(m.shape[0], B).scatter_add_(
        1, idx, torch.ones_like(m))


def mm_rows(counts):
    """Vectorized Miller-Madow entropy per row. counts: (r, B) -> (r,)."""
    n = counts.sum(1)
    p = counts / n.unsqueeze(1)
    plogp = torch.where(p > 0, p * p.log(), torch.zeros_like(p))
    h = -plogp.sum(1)
    k = (counts > 0).sum(1).float()
    return h + (k - 1.0) / (2.0 * n)


# ------------------------------------------------------- coverage study
print(f"{'dist':>12} {'N':>6} {'H_pop':>8} {'closed':>7} {'b-perc':>7} "
      f"{'b-basic':>8} {'b-normal':>9} {'s_claim/s_act':>14}")
coverage = {}
for dname, (sampler, h_pop) in DISTS.items():
    for n in NS:
        cov = [0, 0, 0, 0]
        h_mms, s_claims = [], []
        for rep in range(REPS):
            g = torch.Generator().manual_seed(rep)
            x = sampler(n, g)
            counts = hist_rows(x.unsqueeze(0))
            h_mm = float(mm_rows(counts)[0])
            s = basharin_stderr_from_counts(counts[0])
            h_mms.append(h_mm)
            s_claims.append(s)
            # A: closed-form
            cov[0] += (abs(h_pop - h_mm) <= 1.96 * s)
            # full-pipeline bootstrap: resample RAW x, re-estimate edges
            bidx = torch.randint(0, n, (R_BOOT, n), generator=g)
            hs = mm_rows(hist_rows(x[bidx]))
            q_lo_b = float(torch.quantile(hs, 0.025))
            q_hi_b = float(torch.quantile(hs, 0.975))
            sd_b = float(hs.std())
            cov[1] += (q_lo_b <= h_pop <= q_hi_b)                 # B
            cov[2] += (2 * h_mm - q_hi_b <= h_pop <= 2 * h_mm - q_lo_b)  # C
            cov[3] += (abs(h_pop - h_mm) <= 1.96 * sd_b)          # D
        coverage[(dname, n)] = [c / REPS for c in cov]
        actual_sd = float(torch.tensor(h_mms).std())
        ratio = (sum(s_claims) / REPS) / actual_sd
        print(f"{dname:>12} {n:>6} {h_pop:>8.5f} "
              f"{cov[0] / REPS * 100:>6.1f}% {cov[1] / REPS * 100:>6.1f}% "
              f"{cov[2] / REPS * 100:>7.1f}% {cov[3] / REPS * 100:>8.1f}% "
              f"{ratio:>14.3f}")

# ------------------------------------------------------------------ gates
print()
methods = {"closed-form": 0, "boot-percentile": 1, "boot-basic": 2,
           "boot-normal": 3}
passing = []
for mname, i in methods.items():
    covs = [coverage[(d, n)][i] for d in DISTS for n in NS]
    ok = all(c >= 0.85 for c in covs)
    print(f"  {mname:16s} min coverage = {min(covs) * 100:.1f}% "
          f"-> {'eligible' if ok else 'rejected'}")
    if ok:
        passing.append(mname)

check("verifier's finding reproduces: closed-form FAILS the gate",
      "closed-form" not in passing,
      f"closed-form min = "
      f"{min(coverage[(d, n)][0] for d in DISTS for n in NS) * 100:.1f}%")
check("at least one full-pipeline method passes >= 85% on ALL cells",
      len(passing) > 0, f"eligible: {passing}")

print("\n" + ("ALL PASS" if not failures else f"FAILURES: {failures}"))
raise SystemExit(0 if not failures else 1)
