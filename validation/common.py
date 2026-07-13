"""Candidate estimator implementations for PRE-IMPLEMENTATION validation.

These are NOT the library. They exist to validate formulas numerically
BEFORE anything is integrated into entropylens (the project's mandated
order). The winning formulas move into entropylens/core.py;
these scripts then serve as the reference for regression tests.

All entropies are in NATS (natural log) throughout.
"""

import math

import torch


# --------------------------------------------------------------- baseline
def hist_entropy_minmax(values: torch.Tensor, bins: int = 32) -> float:
    """Old-draft estimator (Difetto 2): raw min-max binning.

    Kept ONLY as the baseline the candidates must beat. Reproduces the
    documented outlier collapse.
    """
    x = values.float().flatten()
    lo, hi = x.min().item(), x.max().item()
    if hi <= lo:
        return 0.0
    counts = torch.histc(x, bins=bins, min=lo, max=hi)
    return entropy_from_counts(counts)


# ------------------------------------------------------------ candidate 1
def quantile_hist_counts(values: torch.Tensor, bins: int = 32,
                         q_lo: float = 0.01, q_hi: float = 0.99):
    """Histogram counts with percentile-range binning.

    Bin edges span [Q(q_lo), Q(q_hi)] of the sample; values outside the
    range are clamped into the first/last bin (winsorisation).
    Returns None when the quantile range is degenerate (Q(q_hi) <= Q(q_lo)),
    i.e. >= 98% of the mass sits at a single point: entropy is 0 by
    convention there, and the caller must flag it as a degenerate case.
    """
    x = values.float().flatten()
    lo = torch.quantile(x, q_lo).item()
    hi = torch.quantile(x, q_hi).item()
    if hi <= lo:
        return None
    counts = torch.histc(x.clamp(lo, hi), bins=bins, min=lo, max=hi)
    return counts


def hist_entropy_quantile(values: torch.Tensor, bins: int = 32,
                          q_lo: float = 0.01, q_hi: float = 0.99) -> float:
    counts = quantile_hist_counts(values, bins, q_lo, q_hi)
    if counts is None:
        return 0.0
    return entropy_from_counts(counts)


# ------------------------------------------------------------ candidate 2
def hist_entropy_fixed(values: torch.Tensor, lo: float, hi: float,
                       bins: int = 32) -> float:
    """Fixed user-supplied range; values outside clamped into edge bins."""
    if hi <= lo:
        raise ValueError("fixed range requires hi > lo")
    x = values.float().flatten().clamp(lo, hi)
    counts = torch.histc(x, bins=bins, min=lo, max=hi)
    return entropy_from_counts(counts)


# ------------------------------------------------------------ candidate 3
def knn_entropy_kl(values: torch.Tensor, k: int = 3) -> float:
    """Kozachenko-Leonenko differential-entropy estimator, 1-D, k-th NN.

    H_hat = psi(N) - psi(k) + log(V_1) + (1/N) * sum_i log(r_i,k)
    with V_1 = 2 (volume of the 1-D unit ball) and r_i,k = distance from
    x_i to its k-th nearest neighbour.

    DIFFERENTIAL entropy: unbounded, can be negative, and diverges to -inf
    on duplicated values. The constant convention is validated numerically
    against N(0,1) ground truth (0.5*log(2*pi*e) ~= 1.4189) in exp1.
    """
    x = values.double().flatten()
    n = x.numel()
    if n <= k:
        raise ValueError("need N > k")
    # direct |a-b| in float64: torch.cdist in float32 suffers catastrophic
    # cancellation on near-equal values (distance rounds to 0 -> log -inf)
    d = (x.unsqueeze(1) - x.unsqueeze(0)).abs()
    d.fill_diagonal_(float("inf"))
    r_k = d.kthvalue(k, dim=1).values
    if (r_k <= 0).any():
        return float("-inf")  # duplicates: estimator undefined
    return float(torch.special.digamma(torch.tensor(float(n)))
                 - torch.special.digamma(torch.tensor(float(k)))
                 + math.log(2.0)
                 + r_k.log().mean())


# ----------------------------------------------------------------- shared
def entropy_from_counts(counts: torch.Tensor) -> float:
    """Plug-in Shannon entropy of a histogram, in nats."""
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * p.log()).sum())


def miller_madow_from_counts(counts: torch.Tensor) -> float:
    """Miller-Madow bias-corrected entropy: H_plugin + (K-1)/(2N),
    K = number of non-empty bins. First-order correction of the plug-in
    estimator's negative bias E[H_hat] - H ~= -(K-1)/(2N)."""
    n = float(counts.sum())
    k = float((counts > 0).sum())
    return entropy_from_counts(counts) + (k - 1.0) / (2.0 * n)


def basharin_stderr_from_counts(counts: torch.Tensor) -> float:
    """Asymptotic std error of the plug-in entropy (Basharin 1959):
    Var(H_hat) ~= (1/N) * (sum_k p_k * log(p_k)^2 - H^2).
    Degenerates (variance -> 0) for uniform and one-point distributions,
    where the first-order delta-method term vanishes: exp4 probes this.
    """
    n = float(counts.sum())
    p = counts / counts.sum()
    p = p[p > 0]
    h = float(-(p * p.log()).sum())
    var = (float((p * p.log() ** 2).sum()) - h ** 2) / n
    return math.sqrt(max(var, 0.0))


# ------------------------------------------------- per-example entropy
def per_example_entropy(x: torch.Tensor, strategy: str) -> torch.Tensor:
    """Entropy across neurons for each example. x: (N, M) -> (N,), nats.

    strategy in {"relu", "abs", "shift"} handles negative values before
    normalising each row to a probability distribution. Rows whose
    post-strategy sum is 0 have UNDEFINED entropy -> NaN (caller flags).
    """
    if strategy == "relu":
        a = x.clamp_min(0.0)
    elif strategy == "abs":
        a = x.abs()
    elif strategy == "shift":
        a = x - x.min(dim=1, keepdim=True).values
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    s = a.sum(dim=1, keepdim=True)
    p = torch.where(s > 0, a / s, torch.zeros_like(a))
    plogp = torch.where(p > 0, p * p.log(), torch.zeros_like(p))
    h = -plogp.sum(dim=1)
    return torch.where(s.squeeze(1) > 0, h,
                       torch.full_like(h, float("nan")))
