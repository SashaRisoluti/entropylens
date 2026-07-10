"""Pure math core: tensor-in / tensor-out.

MUST NOT reference torch.nn (checklist B4): everything here is testable
with plain tensors. Formulas are specified and validated in
MATH_DECISIONS.md (D1, D2, D4, D5, D6); do not change them here without
re-running validation/.

All entropies are in nats.
"""
import math
from dataclasses import dataclass

import torch

# Reliability reason codes (MATH_DECISIONS.md D2/D5)
REASON_OK = 0
REASON_SMALL_N = 1      # N below the bias-dominance threshold for this B
REASON_DEGENERATE = 2   # zero quantile range (all mass at ~one point)
REASON_N1 = 3           # single-sample batch
REASON_LABELS = ("ok", "N below reliability threshold",
                 "degenerate (zero quantile range)", "N=1")


def _validate_input(x, fn, min_dim=1, max_dim=2):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{fn}: expected a torch.Tensor, got {type(x).__name__}")
    if x.numel() == 0:
        raise ValueError(f"{fn}: empty tensor")
    if x.dim() > max_dim:
        raise ValueError(
            f"{fn}: got a {x.dim()}D tensor; conv feature maps are unsupported "
            "in v1 (the spatial-independence assumption is not validated, see "
            "MATH_DECISIONS.md D5 and 03_OPEN_MATH_QUESTIONS.md Q4).")
    if x.dim() < min_dim:
        raise ValueError(f"{fn}: expected at least {min_dim}D, got {x.dim()}D")
    if not x.dtype.is_floating_point:
        x = x.float()
    if not torch.isfinite(x).all():
        raise ValueError(f"{fn}: input contains NaN or Inf")
    return x.float()


def reliability_threshold(n: int, bins: int) -> bool:
    """Bias-dominance criterion (MATH_DECISIONS.md D2): reliable iff the
    worst-case first-order bias (B-1)/(2N) is <= 5% of the normalization
    scale log(min(N,B)). For B=32 this requires N >= 90."""
    scale = math.log(min(n, bins))
    if scale <= 0.0:
        return False
    return (bins - 1) / (2 * n) <= 0.05 * scale


def _quantile_hist_counts(x, bins, q_lo, q_hi, value_range=None):
    """Per-column quantile-winsorized histogram (MATH_DECISIONS.md D1).

    x: (N, M) float tensor. Returns (counts (M, bins), degenerate (M,) bool).
    Degenerate columns (quantile range collapsed) get all-zero counts and
    must be handled by the caller.
    """
    n, m = x.shape
    if value_range is not None:
        lo_v, hi_v = float(value_range[0]), float(value_range[1])
        if not hi_v > lo_v:
            raise ValueError("value_range requires hi > lo")
        lo = torch.full((m,), lo_v)
        hi = torch.full((m,), hi_v)
    else:
        if not (0.0 <= q_lo < q_hi <= 1.0):
            raise ValueError("need 0 <= q_lo < q_hi <= 1")
        lo = torch.quantile(x, q_lo, dim=0)
        hi = torch.quantile(x, q_hi, dim=0)
    degenerate = hi <= lo
    width = torch.where(degenerate, torch.ones_like(hi), hi - lo)
    xc = x.clamp(lo, hi)
    # bin index: matches torch.histc semantics (last edge inclusive)
    idx = ((xc - lo) / width * bins).long().clamp_(0, bins - 1)
    counts = torch.zeros(m, bins).scatter_add_(
        1, idx.t(), torch.ones(m, n))
    counts[degenerate] = 0.0
    return counts, degenerate


def _miller_madow(counts, n, degenerate):
    """H_MM per column from counts (M, bins); 0 on degenerate columns.

    H_MM = H_plugin + (K-1)/(2N), K = occupied bins (MATH_DECISIONS.md D2).
    Shared by batch_wise_entropy and confidence_interval (the bootstrap
    replicates must use the exact same estimator)."""
    p = counts / max(n, 1)
    logp = torch.where(p > 0, p.log(), torch.zeros_like(p))
    h_plugin = -(p * logp).sum(1)
    k = (counts > 0).sum(1).float()
    h_mm = h_plugin + (k - 1.0).clamp(min=0.0) / (2.0 * n)
    return torch.where(degenerate, torch.zeros_like(h_mm), h_mm)


@dataclass
class BatchEntropy:
    """Per-neuron batch-wise entropy (MATH_DECISIONS.md D1/D2).

    entropy:    (M,) Miller-Madow-corrected entropy, nats.
    normalized: (M,) entropy / log(min(N, B)), clamped to [0, 1].
    reliable:   (M,) bool — False where the estimate cannot be certified
                (small N, degenerate column, N=1): a low value there means
                "not estimable", NOT "neuron is specialized".
    reason:     (M,) uint8 codes into REASON_LABELS.

    NO stderr field: the closed-form (Basharin) stderr was removed after
    adversarial verification (D4) — it misses the quantile-edge
    re-estimation variance and under-covers (exp4b). Uncertainty requires
    the RAW values: use confidence_interval(activations, ...).
    """
    entropy: torch.Tensor
    normalized: torch.Tensor
    reliable: torch.Tensor
    reason: torch.Tensor
    n: int
    bins: int


def batch_wise_entropy(activations, *, bins=32, q_lo=0.01, q_hi=0.99,
                       value_range=None):
    """Entropy of each neuron's activation distribution over the batch.

    activations: (N, M) or (N,) tensor. Returns BatchEntropy with (M,)
    fields. Computed with no_grad on detached data; never touches autograd.
    """
    with torch.no_grad():
        x = _validate_input(activations, "batch_wise_entropy").detach()
        if x.dim() == 1:
            x = x.unsqueeze(1)
        n, m = x.shape
        counts, degenerate = _quantile_hist_counts(
            x, bins, q_lo, q_hi, value_range)
        h_mm = _miller_madow(counts, n, degenerate)

        scale = math.log(min(n, bins)) if min(n, bins) > 1 else 0.0
        safe = scale if scale > 0.0 else 1.0
        normalized = (h_mm / safe).clamp(0.0, 1.0)
        normalized = torch.where(degenerate, torch.zeros_like(normalized),
                                 normalized)

        global_ok = reliability_threshold(n, bins)
        reliable = torch.full((m,), global_ok, dtype=torch.bool) & ~degenerate
        # precedence: N1 > degenerate > small_N (an N=1 batch is degenerate
        # by construction, but "N=1" is the more informative reason, D5)
        reason = torch.full((m,), REASON_OK, dtype=torch.uint8)
        if not global_ok:
            reason[:] = REASON_SMALL_N
        reason[degenerate] = REASON_DEGENERATE
        if n == 1:
            reason[:] = REASON_N1
        return BatchEntropy(h_mm, normalized, reliable, reason,
                            n=n, bins=bins)
