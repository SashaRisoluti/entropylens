"""Backend-agnostic numpy mirror of entropylens.core (checklist E1/E2).

Formulas are IDENTICAL to entropylens/core.py, locked by MATH_DECISIONS.md
D1-D6 and D7; numerical equivalence with the torch core is gated by
validation/exp6_np_equivalence.py and frozen in
tests/test_core_np_equivalence.py. MUST NOT import torch, tensorflow, or
jax (checklist E2, AST-checked). All entropies are in nats, float32.
"""
import math
import warnings
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

REASON_OK = 0
REASON_SMALL_N = 1
REASON_DEGENERATE = 2
REASON_N1 = 3
REASON_LABELS = ("ok", "N below reliability threshold",
                 "degenerate (zero quantile range)", "N=1")
_PER_EXAMPLE_STRATEGIES = ("relu", "abs")


@dataclass
class BatchEntropy:
    entropy: np.ndarray      # (M,) Miller-Madow entropy, nats
    normalized: np.ndarray   # (M,) entropy / log(min(N,B)), clamped [0,1]
    reliable: np.ndarray     # (M,) bool
    reason: np.ndarray       # (M,) uint8, indexes REASON_LABELS
    n: int
    bins: int


@dataclass
class PerExampleEntropy:
    entropy: np.ndarray      # (N,) nats; NaN where row invalid (zero mass)
    normalized: np.ndarray   # (N,) entropy / log(M), NaN where invalid
    valid: np.ndarray        # (N,) bool
    strategy: str
    m: int


def reliability_threshold(n, bins):
    # duplicated from entropylens.core (3 lines of pure python): core_np must
    # not import the torch module chain (checklist E2)
    scale = math.log(min(n, bins))
    if scale <= 0.0:
        return False
    return (bins - 1) / (2 * n) <= 0.05 * scale


def _validate_input(x, fn, min_dim=1, max_dim=2):
    # np.asarray IS the adapter point: tf.Tensor (eager) and jax.Array both
    # convert through __array__ / buffer protocol
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError(f"{fn}: empty array")
    if x.ndim > max_dim:
        raise ValueError(
            f"{fn}: got a {x.ndim}D array; conv feature maps are "
            f"unsupported in v1 (the spatial-independence assumption is "
            f"not validated, see MATH_DECISIONS.md D5).")
    if x.ndim < min_dim:
        raise ValueError(f"{fn}: expected at least {min_dim}D, got {x.ndim}D")
    x = x.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"{fn}: input contains NaN or Inf")
    return x


def _quantile_hist_counts(x, bins, q_lo, q_hi, value_range=None):
    n, m = x.shape
    if value_range is not None:
        lo_v, hi_v = float(value_range[0]), float(value_range[1])
        if not hi_v > lo_v:
            raise ValueError("value_range requires hi > lo")
        lo = np.full(m, lo_v, dtype=np.float32)
        hi = np.full(m, hi_v, dtype=np.float32)
    else:
        if not (0.0 <= q_lo < q_hi <= 1.0):
            raise ValueError("need 0 <= q_lo < q_hi <= 1")
        lo = np.quantile(x, q_lo, axis=0).astype(np.float32)
        hi = np.quantile(x, q_hi, axis=0).astype(np.float32)
    degenerate = hi <= lo
    width = np.where(degenerate, np.float32(1.0), hi - lo).astype(np.float32)
    xc = np.clip(x, lo, hi)
    # bin index: matches torch.histc semantics (last edge inclusive)
    idx = ((xc - lo) / width * bins).astype(np.int64)
    np.clip(idx, 0, bins - 1, out=idx)
    counts = np.zeros((m, bins), dtype=np.float32)
    cols = np.broadcast_to(np.arange(m), (n, m))
    np.add.at(counts, (cols.ravel(), idx.ravel()), np.float32(1.0))
    counts[degenerate] = 0.0
    return counts, degenerate


def _miller_madow(counts, n, degenerate):
    p = counts / max(n, 1)
    logp = np.zeros_like(p)
    np.log(p, out=logp, where=p > 0)
    h_plugin = -(p * logp).sum(axis=1)
    k = (counts > 0).sum(axis=1).astype(np.float32)
    h_mm = h_plugin + np.maximum(k - 1.0, 0.0) / (2.0 * n)
    return np.where(degenerate, np.float32(0.0), h_mm).astype(np.float32)


def batch_wise_entropy(activations, *, bins=32, q_lo=0.01, q_hi=0.99,
                          value_range=None):
    x = _validate_input(activations, "batch_wise_entropy")
    if x.ndim == 1:
        x = x[:, None]
    n, m = x.shape
    counts, degenerate = _quantile_hist_counts(x, bins, q_lo, q_hi, value_range)
    h_mm = _miller_madow(counts, n, degenerate)
    scale = math.log(min(n, bins)) if min(n, bins) > 1 else 0.0
    safe = scale if scale > 0.0 else 1.0
    normalized = np.clip(h_mm / safe, 0.0, 1.0)
    normalized = np.where(degenerate, np.float32(0.0), normalized).astype(np.float32)
    global_ok = reliability_threshold(n, bins)
    reliable = np.full(m, global_ok, dtype=bool) & ~degenerate
    reason = np.full(m, REASON_OK, dtype=np.uint8)
    if not global_ok:
        reason[:] = REASON_SMALL_N
    reason[degenerate] = REASON_DEGENERATE
    if n == 1:
        reason[:] = REASON_N1
    return BatchEntropy(h_mm, normalized, reliable, reason, n=n, bins=bins)


def per_example_entropy(activations, strategy):
    if strategy == "shift":
        raise ValueError(
            "strategy 'shift' was removed (MATH_DECISIONS.md D3): it has no "
            "theoretical grounding and maps constant rows to an undefined "
            "distribution. Choose 'relu' or 'abs' explicitly.")
    if strategy not in _PER_EXAMPLE_STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r}; choose from "
            f"{_PER_EXAMPLE_STRATEGIES}")
    x = _validate_input(activations, "per_example_entropy", min_dim=2)
    n, m = x.shape
    a = np.maximum(x, 0.0) if strategy == "relu" else np.abs(x)
    s = a.sum(axis=1)
    valid = s > 0
    denom = np.maximum(s, np.float32(1e-38))[:, None]
    p = np.where(valid[:, None], a / denom, np.float32(0.0)).astype(np.float32)
    logp = np.zeros_like(p)
    np.log(p, out=logp, where=p > 0)
    h = -(p * logp).sum(axis=1)
    h = np.where(valid, h, np.float32("nan")).astype(np.float32)
    scale = math.log(m) if m > 1 else 1.0
    with np.errstate(invalid="ignore"):
        normalized = np.where(
            valid, np.clip(h / scale, 0.0, 1.0), np.float32("nan")
        ).astype(np.float32)
    return PerExampleEntropy(h, normalized, valid, strategy, m)


def confidence_interval(activations, *, bins=32, q_lo=0.01, q_hi=0.99,
                           value_range=None, level=0.95, n_boot=200, rng=None):
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    if n_boot < 2:
        raise ValueError("n_boot must be >= 2")
    x = _validate_input(activations, "confidence_interval")
    if x.ndim == 1:
        x = x[:, None]
    n, m = x.shape
    counts, degenerate = _quantile_hist_counts(x, bins, q_lo, q_hi, value_range)
    est = _miller_madow(counts, n, degenerate)
    p_max = counts.max(axis=1) / max(n, 1)
    if n < 400 and bool((p_max > 0.9).any()):
        warnings.warn(
            "entropylens.core_np.confidence_interval: >90% of the mass of at least "
            "one column sits in a single bin and N < 400; under this "
            "sparsity the CI coverage was measured at 83% at N=200 (below "
            "the 85% floor), recovering only from N ~ 400 "
            "(MATH_DECISIONS.md D4).")
    if rng is None:
        rng = np.random.default_rng()
    hs = np.empty((n_boot, m), dtype=np.float32)
    for r in range(n_boot):
        ridx = rng.integers(0, n, size=n)
        c_r, d_r = _quantile_hist_counts(x[ridx], bins, q_lo, q_hi, value_range)
        hs[r] = _miller_madow(c_r, n, d_r)
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    half = z * hs.std(axis=0, ddof=1)   # ddof=1: parity with torch unbiased std
    bound = math.log(min(n, bins)) if min(n, bins) > 1 else 0.0
    hi = np.maximum(np.minimum(est + half, bound), est)
    lo = np.maximum(est - half, 0.0)
    return est, lo, hi
