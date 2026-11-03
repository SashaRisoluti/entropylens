"""Pure math core: tensor-in / tensor-out.

MUST NOT reference torch.nn (checklist B4): everything here is testable
with plain tensors. Formulas are specified and validated in
MATH_DECISIONS.md (D1-D6, D10-D12); do not change them here without
re-running validation/.

All entropies are in nats.
"""
import math
import warnings
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
            "MATH_DECISIONS.md D5).")
    if x.dim() < min_dim:
        raise ValueError(f"{fn}: expected at least {min_dim}D, got {x.dim()}D")
    # cast FIRST, then check (MATH_DECISIONS.md D11, validation/exp10):
    # checking on the original dtype let a finite float64 above float32's
    # max (~3.4e38) pass and become inf on the downcast, after which
    # batch_wise_entropy reported H=0 with reliable=True. One post-cast
    # check catches both causes — an original NaN/Inf is still NaN/Inf
    # after casting — so no second pass is needed. This matches core_np,
    # which has always validated in this order (D7).
    x = x.float()
    if not torch.isfinite(x).all():
        raise ValueError(f"{fn}: input contains NaN or Inf (or a value "
                         "outside float32 range, which becomes Inf on the "
                         "float32 cast this library performs)")
    return x


def reliability_threshold(n: int, bins: int) -> bool:
    """Bias-dominance criterion (MATH_DECISIONS.md D2): flags unreliable
    when the worst-case first-order bias (B-1)/(2N) exceeds 5% of the
    normalization scale log(min(N,B)). For B=32 this requires N >= 90.

    NECESSARY, NOT SUFFICIENT (adversarial review, D2 counterexamples):
    the criterion models bin-count bias only. Heavy-tailed (lognormal
    sigma=2: +10% at N=128, B=32) or highly sparse data (90% zeros:
    -7% at N=128) can exceed the 5% target while this returns True;
    empirically all probed adversarial cases fall below 5% from N >= 256
    at B=32, but B=8 stays miscalibrated for heavy tails up to N~512.
    True = "no a priori red flag", False = "certified unreliable".
    For a data-driven check use confidence_interval."""
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
    # float64 for the edges and the bin index (MATH_DECISIONS.md D12,
    # validation/exp11): in float32 the quantile lerp, hi - lo and xc - lo
    # overflow to inf on finite inputs near float32's max, and the NaN index
    # that follows puts every sample in bin 0 -> H=0 with reliable=True.
    # float64 holds the difference of any two float32 values.
    # ponytail: transient float64 copy of x (~2x memory here); chunk over
    # columns if that ever matters.
    x = x.double()
    if value_range is not None:
        lo_v, hi_v = float(value_range[0]), float(value_range[1])
        if not hi_v > lo_v:
            raise ValueError("value_range requires hi > lo")
        lo = torch.full((m,), lo_v, dtype=torch.float64)
        hi = torch.full((m,), hi_v, dtype=torch.float64)
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
                "not estimable", NOT "neuron is specialized". True means
                "no a priori red flag", NOT a certification: see
                reliability_threshold and MATH_DECISIONS.md D2 for the
                documented counterexamples (sparse / heavy-tailed data at
                N < ~256).
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


_PER_EXAMPLE_STRATEGIES = ("relu", "abs")


@dataclass
class PerExampleEntropy:
    """Entropy across neurons for each example (MATH_DECISIONS.md D3/D6).

    entropy:    (N,) nats; NaN where the row is invalid (zero mass).
    normalized: (N,) entropy / log(M), NaN where invalid.
    valid:      (N,) bool — False means "entropy undefined for this row
                (all-zero after strategy)", not "low entropy".
    NOTE: no stderr here — each row's entropy is a deterministic function
    of that row, not a sample estimate (D4). Uncertainty applies only to
    batch aggregates (SEM of the mean over valid rows).
    """
    entropy: torch.Tensor
    normalized: torch.Tensor
    valid: torch.Tensor
    strategy: str
    m: int


def per_example_entropy(activations, strategy):
    """Entropy of each example's activation pattern across M neurons.

    strategy is MANDATORY ('relu' or 'abs'): the two are not interchangeable
    (Difetto 3: same input gives H=0.0856 vs 0.7362). Use
    entropylens.strategies to derive it from the activation module type.
    'shift' was removed: no theoretical grounding, and it annihilates
    constant rows (the maximum-entropy case) into undefined ones.
    """
    if strategy not in _PER_EXAMPLE_STRATEGIES:
        if strategy == "shift":
            raise ValueError(
                "strategy 'shift' was removed (MATH_DECISIONS.md D3): it has "
                "no theoretical grounding and maps constant rows to an "
                "undefined distribution. Choose 'relu' or 'abs' explicitly.")
        raise ValueError(f"unknown strategy {strategy!r}; "
                         f"choose from {_PER_EXAMPLE_STRATEGIES}")
    with torch.no_grad():
        x = _validate_input(activations, "per_example_entropy",
                            min_dim=2).detach()
        n, m = x.shape
        a = x.clamp_min(0.0) if strategy == "relu" else x.abs()
        # sum and divide in float64 (MATH_DECISIONS.md D12, validation/exp11):
        # a float32 row sum of finite elements near float32's max overflowed
        # to inf, a / inf = 0, and a uniform row read H=0 with valid=True.
        # ponytail: float64 copy of the activations, same ceiling as in
        # _quantile_hist_counts; also why MPS (no float64) cannot run this.
        a = a.double()
        s = a.sum(1)
        valid = s > 0
        # divide by s itself, NOT by a clamped s (MATH_DECISIONS.md D10,
        # validation/exp9): clamping to 1e-38 silently renormalised rows
        # whose sum was subnormal, collapsing a uniform row to H~0.37 with
        # valid=True. s is a sum of non-negative terms, so s >= max(a) and
        # s > 0 holds exactly whenever any element is non-zero.
        p = torch.where(valid.unsqueeze(1), a / s.unsqueeze(1),
                        torch.zeros_like(a)).float()
        plogp = torch.where(p > 0, p * p.log(), torch.zeros_like(p))
        h = -plogp.sum(1)
        nan = torch.full_like(h, float("nan"))
        h = torch.where(valid, h, nan)
        scale = math.log(m) if m > 1 else 1.0
        normalized = torch.where(valid, (h / scale).clamp(0.0, 1.0), nan)
        return PerExampleEntropy(h, normalized, valid, strategy, m)


def confidence_interval(activations, *, bins=32, q_lo=0.01, q_hi=0.99,
                        value_range=None, level=0.95, n_boot=200,
                        generator=None):
    """Full-pipeline bootstrap CI for batch-wise entropy (MATH_DECISIONS.md
    D4 as amended after adversarial verification; validation/exp4b).

    Returns (estimate, lo, hi), each (M,). estimate = Miller-Madow point
    estimate (bitwise identical to batch_wise_entropy). [lo, hi] is the
    normal interval estimate +- z * sd over n_boot bootstrap replicates in
    which the RAW values are resampled with replacement and the quantile
    edges RE-estimated per replicate. Edge re-estimation variance is what
    the closed-form (Basharin) stderr cannot see: it under-covered
    (62.5-88% vs 95% nominal, exp4b) and was rejected; percentile/basic
    bootstrap intervals collapse near the entropy ceiling (min 4%/31.5%)
    and were rejected too. boot-normal coverage: 90.5-100% on
    normal/uniform/exponential at N in {200, 1000, 5000}.

    Cost: n_boot full histogram passes over the input — call on demand,
    not per tracked batch (this is why BatchEntropy carries no stderr and
    why this function takes raw activations, not a BatchEntropy: the CI is
    not derivable from counts alone).

    Validity domain (adversarial review, MATH_DECISIONS.md D4): coverage
    is ">= 85%, conservative", not exact 95% (measured 86-100% across the
    validated grid), for N >= 200 — the reliable flag's N >= 90 does NOT
    transfer to the CI on sparse or heavily skewed data. Under extreme
    sparsity (>90% of mass in one bin) the floor rises to N ~ 400; this
    function detects that from the histogram and warns. Degenerate
    columns (and degenerate bootstrap replicates, which do occur on
    sparse data) use the H=0 convention; a fully degenerate column yields
    a zero-width interval — a documented blind spot: the `reliable` flag
    of batch_wise_entropy signals those, the CI cannot.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    if n_boot < 2:
        raise ValueError("n_boot must be >= 2")
    with torch.no_grad():
        x = _validate_input(activations, "confidence_interval").detach()
        if x.dim() == 1:
            x = x.unsqueeze(1)
        n, m = x.shape
        counts, degenerate = _quantile_hist_counts(
            x, bins, q_lo, q_hi, value_range)
        est = _miller_madow(counts, n, degenerate)
        p_max = counts.max(dim=1).values / max(n, 1)
        if n < 400 and bool((p_max > 0.9).any()):
            warnings.warn(
                "entropylens.confidence_interval: >90% of the mass of at "
                "least one column sits in a single bin and N < 400; under "
                "this sparsity the CI coverage was measured at 83% at "
                "N=200 (below the 85% floor), recovering only from "
                "N ~ 400 (MATH_DECISIONS.md D4).")
        hs = torch.empty(n_boot, m)
        for r in range(n_boot):
            ridx = torch.randint(0, n, (n,), generator=generator)
            c_r, d_r = _quantile_hist_counts(
                x[ridx], bins, q_lo, q_hi, value_range)
            hs[r] = _miller_madow(c_r, n, d_r)
        z = float(torch.special.ndtri(torch.tensor(0.5 + level / 2.0)))
        half = z * hs.std(dim=0)
        # the estimand H_pop is bounded by log(min(N,B)) (D6 proof), so
        # truncating the interval to that bound never lowers coverage;
        # keep hi >= est so ordering survives when H_MM itself exceeds
        # the bound (Miller-Madow overshoot at the ceiling, D2)
        bound = math.log(min(n, bins)) if min(n, bins) > 1 else 0.0
        hi = torch.maximum((est + half).clamp(max=bound), est)
        return est, (est - half).clamp(min=0.0), hi
