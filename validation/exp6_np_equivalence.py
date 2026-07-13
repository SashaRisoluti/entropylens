"""EXP6 (E1): equivalenza numerica del core numpy candidato vs entropylens.core.

Protocollo: stesse attivazioni generate in torch, convertite a numpy con
.numpy() (stessi bit float32); stessi parametri (B, q_lo, q_hi). I candidati
numpy vivono QUI (convenzione validation/): se i gate passano, il codice si
sposta VERBATIM in entropylens/core_np.py e i gate diventano regressioni in
tests/test_core_np_equivalence.py.

Gate:
  G1  counts bitwise-uguali e maschera degenerate identica su tutti i casi
  G2  |H_np - H_torch| <= 1e-5 nats; |norm_np - norm_torch| <= 5e-6
      (5e-6 ~= 1e-5 / log 8: coerente col gate su H alla scala minima B=8)
  G3  reliable e reason identici (uint8/bool per elemento)
  G4  per-example: |dH| <= 1e-6, posizioni NaN identiche, strategie relu e abs
  G5  CI (RNG diversi -> gate statistico): semi-ampiezza media entro il 10%
      relativo dalla versione torch; copertura pooled PER-COLONNA >= 0.85
      del riferimento large-sample (N=200_000, ~popolazione a meno di bias
      (K-1)/2N ~ 8e-5) su gaussiana N=400, 40 ripetizioni x 4 colonne.
      Il conteggio e' per colonna come in exp4b, NON congiunto: la copertura
      congiunta di 4 colonne ~ c^4 fallirebbe 0.85 anche con CI perfetti.

Rischio di parita' reale (AD-2): torch.quantile vs np.quantile (interpolazione
lineare in entrambi, possibile differenza di promozione dtype) e ordine di
somma nelle riduzioni. Tutto il resto e' elementwise float32 identico.
"""
import math
import sys
from pathlib import Path

# il pacchetto non e' pip-installato: stesso bootstrap di tests/conftest.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from entropylens import core

# ---------------------------------------------------------------------------
# CANDIDATO NUMPY (destinazione: entropylens/core_np.py, verbatim)
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from statistics import NormalDist
import warnings

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


def np_batch_wise_entropy(activations, *, bins=32, q_lo=0.01, q_hi=0.99,
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


def np_per_example_entropy(activations, strategy):
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


def np_confidence_interval(activations, *, bins=32, q_lo=0.01, q_hi=0.99,
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
            "entropylens.confidence_interval: >90% of the mass of at least "
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


# ---------------------------------------------------------------------------
# HARNESS
# ---------------------------------------------------------------------------
failures = []


def check(name, cond, detail):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}: {detail}")
    if not cond:
        failures.append(name)


def make_cases(seed):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(1000, 8, generator=g)
    zeroinf = torch.randn(1000, 8, generator=g)
    zeroinf[torch.rand(1000, 8, generator=g) < 0.9] = 0.0
    outlier = torch.randn(1000, 8, generator=g)
    outlier[0] = 1000.0   # exp1 protocol: one outlier per column
    return {
        "gauss": base,
        "uniform": torch.rand(1000, 8, generator=g),
        "lognormal": torch.randn(1000, 8, generator=g).mul(2.0).exp(),
        "zeroinf90": zeroinf,
        "outlier": outlier,
        "constant": torch.full((1000, 8), 3.14),
    }


print("=== G1-G3: batch-wise equivalence ===")
max_dh, max_dnorm = 0.0, 0.0
for seed in range(5):
    for dist, xt_full in make_cases(seed).items():
        for n in (1, 8, 128, 1000):
            xt = xt_full[:n]
            xn = xt.numpy()
            for bins in (8, 32):
                bt = core.batch_wise_entropy(xt, bins=bins)
                bn = np_batch_wise_entropy(xn, bins=bins)
                ct, dt = core._quantile_hist_counts(
                    xt.float(), bins, 0.01, 0.99)
                cn, dn = _quantile_hist_counts(
                    xn.astype(np.float32), bins, 0.01, 0.99)
                if not np.array_equal(ct.numpy(), cn):
                    check(f"G1 counts {dist} N={n} B={bins} s={seed}",
                          False, "counts differ")
                if not np.array_equal(dt.numpy(), dn):
                    check(f"G1 degenerate {dist} N={n} B={bins} s={seed}",
                          False, "degenerate mask differs")
                dh = float(np.nanmax(np.abs(bt.entropy.numpy() - bn.entropy)))
                dnorm = float(np.nanmax(np.abs(
                    bt.normalized.numpy() - bn.normalized)))
                max_dh = max(max_dh, dh)
                max_dnorm = max(max_dnorm, dnorm)
                if not np.array_equal(bt.reliable.numpy(), bn.reliable):
                    check(f"G3 reliable {dist} N={n} B={bins} s={seed}",
                          False, "reliable differs")
                if not np.array_equal(bt.reason.numpy(), bn.reason):
                    check(f"G3 reason {dist} N={n} B={bins} s={seed}",
                          False, "reason differs")
check("G1 counts+degenerate", "G1" not in "".join(failures),
      "bitwise-identical across all cases")
check("G2 |dH| <= 1e-5", max_dh <= 1e-5, f"max |dH| = {max_dh:.2e}")
check("G2 |dnorm| <= 5e-6", max_dnorm <= 5e-6, f"max = {max_dnorm:.2e}")
check("G3 reliable+reason", not any(f.startswith("G3") for f in failures),
      "identical across all cases")

print("=== G4: per-example equivalence ===")
max_dpe = 0.0
for seed in range(5):
    g = torch.Generator().manual_seed(100 + seed)
    xt = torch.randn(128, 16, generator=g)
    xt[0] = 0.0   # invalid row (zero mass under both strategies)
    xn = xt.numpy()
    for strat in ("relu", "abs"):
        pt = core.per_example_entropy(xt, strat)
        pn = np_per_example_entropy(xn, strat)
        if not np.array_equal(np.isnan(pt.entropy.numpy()), np.isnan(pn.entropy)):
            check(f"G4 NaN positions {strat} s={seed}", False, "differ")
        if not np.array_equal(pt.valid.numpy(), pn.valid):
            check(f"G4 valid {strat} s={seed}", False, "differ")
        dpe = float(np.nanmax(np.abs(pt.entropy.numpy() - pn.entropy)))
        max_dpe = max(max_dpe, dpe)
check("G4 |dH| <= 1e-6", max_dpe <= 1e-6, f"max |dH| = {max_dpe:.2e}")

print("=== G5: CI statistical parity + coverage ===")
ratios = []
for seed in range(20):
    g = torch.Generator().manual_seed(200 + seed)
    xt = torch.randn(400, 4, generator=g)
    et, lt, ht = core.confidence_interval(
        xt, n_boot=200, generator=torch.Generator().manual_seed(seed))
    en, ln, hn = np_confidence_interval(
        xt.numpy(), n_boot=200, rng=np.random.default_rng(seed))
    hw_t = (ht - lt).numpy() / 2.0
    hw_n = (hn - ln) / 2.0
    ratios.append(float(np.mean(np.abs(hw_n - hw_t) / hw_t)))
mean_ratio = sum(ratios) / len(ratios)
check("G5 half-width parity <= 10%", mean_ratio <= 0.10,
      f"mean relative half-width diff = {mean_ratio:.3f}")

g = torch.Generator().manual_seed(999)
href = core.batch_wise_entropy(torch.randn(200_000, 4, generator=g)).entropy.numpy()
cover = np.zeros(4)
REPS = 40
for rep in range(REPS):
    gr = torch.Generator().manual_seed(1000 + rep)
    x = torch.randn(400, 4, generator=gr).numpy()
    _, lo, hi = np_confidence_interval(x, n_boot=200,
                                       rng=np.random.default_rng(rep))
    cover += (lo <= href) & (href <= hi)
coverage = float(cover.sum() / (REPS * 4))
check("G5 coverage >= 0.85 (gauss N=400, pooled per-column)", coverage >= 0.85,
      f"pooled = {coverage:.3f}; per-column = {(cover / REPS).round(3)}")

print("\n" + ("ALL PASS" if not failures else f"FAILURES: {failures}"))
raise SystemExit(0 if not failures else 1)
