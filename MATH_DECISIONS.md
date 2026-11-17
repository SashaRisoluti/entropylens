# MATH_DECISIONS — validated formulas and closed decisions

Produced by the experiments in `validation/` (exp1–exp5 + exp4b, all PASS,
run 2026-07-10 on torch 2.8.0+cpu, Python 3.13). Each decision below
cites the numeric evidence that determined it. This file is the
specification of the formulas for `entropylens/core.py`; of the original
architectural draft, only the module structure remains valid.

**Revision history.** First version submitted to internal multi-agent
adversarial verification (AI-assisted) (2026-07-10). Outcomes incorporated:
D4 **reopened and re-decided** (the closed-form failed its own gate on the
real pipeline → exp4b, full-pipeline bootstrap); D6/D1 formula of the
winsorized maximum corrected for asymmetric quantiles (regression-tested in
exp1); D5 N=0 row added; D1 scoping of the H* cap and heavy-tails caveat
added.
2026-07-13: added D7–D9 (TF/JAX backends) on the basis of exp6–exp8, before implementation (same validation-first discipline as D1–D6).
2026-08-03: added D10 (per-example denominator) from internal design review (AI-assisted): an implementation defect, not a formula defect — the `1e-38` clamp collapsed a uniform row to H≈0.37 with `valid=True`. Gate in exp9, regression frozen.
2026-08-03: added D11 (input validation order), closes the open point left by the D5 note of 2026-07-13: user decision, option "cast first, check after". Gate in exp10.
2026-11-02: added D12 (overflow in the intermediates on finite float32 input) from code review of the D10/D11 branch: user decision, option "float64 in the intermediates". Gate in exp11; exp6 numpy candidate resynchronized (it was stuck at pre-D10).
2026-11-03: corrections from the internal multi-agent code review (AI-assisted; Brooks-lint, 4 reviewers). Vacuous gates in exp10 (G5 unconditional; G4 declared a bitwise identity that was never verified). NaN-blind accumulators in exp9/exp11. New gate exp6 G0 (candidate ≡ core_np). D11 message and warning of the numpy mirror aligned. Cost of D12 completed (time measured, MPS constraint) and confirmed by the user. Prose of D1 realigned with its table. D7 annotated as superseded by D11.

All entropies are in **nats** (natural logarithm).

---

## D1 — Robust binning (Q1 / Defect 2 / A3 / C1): quantile-winsorized binning

**Adopted formula.** Given a sample `x_1..x_N`, with `B` bins and quantiles
`q_lo = 0.01`, `q_hi = 0.99` by default:

```
lo = Quantile(x, q_lo),  hi = Quantile(x, q_hi)
if hi <= lo:  H = 0, reliable = False, reason = "degenerate"
otherwise:    uniform bins on [lo, hi]; out-of-range values are
              clamped into the first/last bin (winsorization)
H_plugin = - Σ_k p_k log p_k,   p_k = c_k / N
```

**Evidence (exp1).** With 1000 N(0,1) values + 1 outlier (1000.0), 10 seeds:

| estimator | mean relative change | outcome vs gate A3 (<10%) |
|---|---|---|
| min-max (old draft) | **99.735%** | reproduces Defect 2 |
| quantile-winsorized | **0.182%** (max 0.624%) | PASS |
| fixed range [-4,4] | 0.176% | PASS |
| k-NN Kozachenko-Leonenko | 0.932% | PASS |

**Sweep over the outlier magnitude (from internal design review
(AI-assisted) — finding DA "cherry-picking": 1000.0 is the case most
favorable to quantile-winsorized and most unfavorable to min-max).** Same
protocol, outlier at 5/20/100/1000 standard deviations:

| outlier | min-max | quantile-winsorized | fixed range | k-NN |
|---|---|---|---|---|
| 5σ | 7.396% | 0.182% | 0.176% | 0.496% |
| 20σ | 41.116% | 0.182% | 0.176% | 0.644% |
| 100σ | 76.970% | 0.182% | 0.176% | 0.767% |
| 1000σ | 99.735% | 0.182% | 0.176% | 0.932% |

Min-max degrades **continuously** with the magnitude (7.4% → 41% → 77% →
99.7%): 1000.0 is not an isolated pathological case, it is the limit of a
curve that exceeds the 10% gate already at 20σ; at 5σ (a realistic, not
extreme, outlier) it stays below the gate (7.4%), but it is ~40× the
quantile-winsorized value (0.182%). Quantile-winsorized and fixed range are
**flat** over more than two orders of magnitude (5σ → 1000σ) — the
robustness does not depend on the scale of the outlier, consistent with the
mechanism (the clamp ignores the value beyond the quantile, whatever it
is). k-NN grows slightly with the magnitude but stays below the gate. Full
sweep in `validation/exp1_binning_outliers.py` (informative, not gated:
gate A3 stays fixed at the documented anchor to remain comparable over
time).

**Choice: quantile-winsorized as the default.** Fixed range remains as an
explicit option (`range=(lo, hi)`) for those who know the activation range
a priori: it is marginally more robust but shifts onto the user a
choice that, if wrong, cuts off real behavior of the network.
k-NN is **rejected as the default** even though it passes the gate, for
three structural reasons, not robustness reasons: (a) it estimates the
*differential* entropy, which is unbounded and potentially negative, hence
incompatible with the [0,1] normalization required by A6/Defect 4; (b) it
diverges to −∞ on duplicated values (a real case with ReLU activations full
of zeros); (c) O(N²) cost.

**Known cost of winsorization (documented, not hidden).** On continuous
data with **uniform density** the bound `log B` is not exactly attainable
under clipping: the attainable maximum is

```
H* = -( p_lo·log p_lo + p_hi·log p_hi + (B-2)·p_m·log p_m )
     p_m  = (q_hi - q_lo)/B
     p_lo = p_m + q_lo            [massa clampata nel primo bin]
     p_hi = p_m + (1 - q_hi)      [massa clampata nell'ultimo bin]
```

which for the defaults (B=32, q=0.01/0.99) equals 3.46298 = **99.92% of
log(32)** (verified in exp1 against the analytical value, including the
asymmetric cases: q=(0.05,0.99) → 3.43717, q=(0.0,0.95) → 3.43835).
Distortion considered negligible compared with the 99.7% collapse it
resolves.

*Correction from internal adversarial review (AI-assisted):* the first
version used `p_e = p_m + q_lo` for BOTH edge bins — correct only for
symmetric quantiles (right at the defaults by coincidence); with
q=(0.05,0.99) the mass summed to 1.04 and "H*" exceeded log 32. The
corrected formula above is now regression-tested in exp1 (mass-check
Σp = 1 + empirical comparison to 1e-4).

**Scope of the H\* cap (scoping, from internal adversarial review
(AI-assisted)):** it holds for continuous data, not in general. With mass
concentrated on B discrete levels the sample quantiles fall on the extreme
levels, nothing is clamped and H = log B exactly under the defaults
(verified in exp1: 32 levels × 100, q=0.01/0.99 → H = log 32 to 1e-5). The
correct statement is: "clipping reduces the attainable maximum only when
there is mass beyond the quantiles q_lo/q_hi".

**Reproducibility caveat on heavy tails (from internal adversarial review
(AI-assisted), reproduced independently):** with heavy-tailed data (Cauchy)
the quantile edges re-estimated per batch make H_norm more variable from
seed to seed than a fixed-range reference: std 0.036 vs 0.008 at N=2000 and
0.017 vs 0.003 at N=10000 (4–5×, 30 seeds; it does not vanish as N grows).
No gate violated (robustness to outliers and N-invariance on average
remain), but it is a documented reproducibility cost: for distributions
with a known heavy tail the fixed-range option `value_range=(lo, hi)` is
preferable.

---

## D2 — Cross-layer normalization (Q2 / Defects 1+4 / A4 / A6): Miller-Madow + log(min(N,B))

**Adopted formula.**

```
K = numero di bin non vuoti
H_MM   = H_plugin + (K - 1) / (2N)          [correzione Miller-Madow]
H_norm = min(1, H_MM / log(min(N, B)))
```

**Evidence (exp2).** N(0,1), B=32, 20 seeds, analytical reference
H_true = 3.29296 (population quantiles, tails clamped):

| N | without correction: dev. | with Miller-Madow: dev. |
|---|---|---|
| 128 | 3.68% | **0.34%** |
| 512 | 0.64% | **0.28%** |
| 2048 | 0.11% | **0.12%** |
| 8192 | 0.02% | **0.03%** |

Gate A4 (dev. < 5% on 4 values of N): PASS on all four with
Miller-Madow; without correction N=128 passes marginally and N=50/32
fail (8.6% / 13.8%).

**Why the denominator is log(min(N,B))** (fix for Defect 1): at most
min(N,B) bins can be non-empty (proof in D6), so the attainable bound is
log(min(N,B)), not log(B). With N=10, B=32 the maximum observed entropy is
log(10)=2.3026, as documented in Defect 1 and re-verified.

**Clamp at 1 (declared pragmatic choice).** The Miller-Madow correction is
a correction *in expectation*; on a sample exactly uniform over the bins it
can push H_MM above log(min(N,B)) (e.g. N=320 uniform: H_plugin =
log 32, correction +0.048). The clamp `min(1, ·)` keeps the [0,1] scale
required by A6 at the cost of a small distortion at the ceiling. It is a
practical convenience, not a theoretical necessity: documented as such
(project constraint on choices without theoretical foundation).

**Reliability criterion (Defect 5 / Q2).**

```
reliable  ⟺  (B - 1) / (2N)  ≤  0.05 · log(min(N, B))
```

that is: the sample is reliable when the *worst-case* absolute value of the
first-order bias (which occurs with K=B) does not exceed 5% of the
normalization scale. For B=32: N ≥ 90. The criterion uses B−1 (not K̂−1) so
that it can be evaluated a priori, independently of the data. Evidence from
exp2 (on N(0,1)): at N=32 and N=50 the flag trips and the deviation without
correction is indeed 13.8% / 8.6%; from N=128 upward the flag is green and
the corrected deviation is ≤0.34%.

**Scope of the criterion (from internal adversarial verification
(AI-assisted), cells confirmed at 400 seeds): a NECESSARY, NOT sufficient
condition.** The criterion models only the bin-count bias; two real
mechanisms are invisible to it and produce `reliable=True` with error
beyond 5%:

| distribution | B | N | flag | mean dev. H_norm |
|---|---|---|---|---|
| lognormal σ=2 | 32 | 96–128 | green | **+10.3 … +11.8%** |
| lognormal σ=2 | 32 | 200 | green | **+7.96%** |
| zero-inflated 90% ("dead" ReLU + Exp(1)) | 32 | 128 | green | **−7.43%** |
| zero-inflated 85% (half-normal) | 32 | 128 | green | −6.77% |
| lognormal σ=2 | 8 | 96 / 200 / 512 | green | **+30.2% / +16.8% / +5.1%** |

Mechanisms: (a) extreme sparsity → in the occupied bins Np_k ≪ 1 and the
real bias far exceeds the correction (K̂−1)/(2N) → residual
underestimation; (b) heavy tails → the sample quantile Q(0.99)
systematically underestimates the population one at N~100 → compressed
range → overestimation. Safe empirical floors from the probes (400 seeds):
at B=32 all the adversarial cases tried fall below 5% from **N ≥ 256**; at
**B=8 the criterion is mis-calibrated** for heavy tails up to N≈512 (the
log 8 scale is small relative to the edge noise). Conservative side (red
flag with the estimate correct within 5%, e.g. B=128 at N=128–261):
systematic but in the safe direction — accepted.

Honest semantics of the flag, reflected in the docstrings: `False` =
"certified unreliable"; `True` = "**no a priori alarm signal** from the
count-bias model", NOT a certification for arbitrary distributions (no
data-independent criterion can be one: the deviation depends on tail/atoms
unknown a priori). Mitigations documented for the user:
N ≥ 256, fixed `value_range`, or the bootstrap CI (D4) as a data-driven
verification. The zero-inflated 90% case at N=128 is frozen as a regression
test of the limitation (`tests/test_reliability_limitations.py`).
*Open point reported to the user (not decided autonomously, rule 05):
raising the default threshold is a trade-off with no single answer.*

**Clamp blindness quantified (from internal adversarial verification
(AI-assisted)):** near the ceiling, two realizable samples at N=96/B=32
with plug-in entropies 4.15% apart both produce exactly H_norm = 1.0000;
the width of the blind zone is ≈ (B−1)/(2N)/log B, that is exactly 5% at
the minimum reliability threshold N=90. It remains a declared practical
convenience; those who need to distinguish values at the ceiling should use
the unnormalized `entropy` (nats).

Every batch-wise result carries `reliable` and a `reason` when
False (`"N < N_min(B)"`, `"degenerate"`, `"N = 1"`). It distinguishes "low
entropy because the behavior is concentrated" from "unreliable estimate"
(Defect 5).

---

## D3 — Strategies for negative values (Q3 / Defect 3 / A5 / C2)

**Anchor reproduced (exp3):** on the documented vector
`[5, -5, 0.01×6]`: shift→1.9064, abs→0.7362, relu→0.0856 — identical to
the documented analysis of the defect, to 4 decimal places: the validated
implementation is the same one that produced the evidence of Defect 3.

**`shift` is REMOVED from the library.** Two reasons: (a) no theoretical
rationale (already noted in Defect 3); (b) a pathology discovered in exp3:
on a *constant* row (e.g. uniform activation, the maximum-entropy case!)
the shift zeroes it out completely → zero sum → undefined entropy.
A strategy that turns the maximum-entropy case into an undefined case does
not measure what it claims to measure.

**Activation → strategy table (deliverable A5/C2):**

| Wrapped module | Strategy | Theoretical rationale | Evidence exp3 |
|---|---|---|---|
| ReLU, ReLU6, Softplus | `relu` (no-op) | output ≥ 0 by construction | negative mass 0% |
| Sigmoid | `relu` (no-op) | output in (0,1) | negative mass 0% |
| Softmax | none: the output IS already a distribution; entropy computed directly | p sums to 1 by construction | — |
| GELU, SiLU/Swish | `relu` | BOUNDED negative range (GELU ≥ −0.170, SiLU ≥ −0.278, verified) vs unbounded positive range: the sign encodes "almost off", not a direction | scale-dependent negative mass: GELU 14.7%→2.6%, SiLU 24.3%→6.9% (σ=1→3); decreases with scale (verified) |
| LeakyReLU | `relu` | negatives are signal attenuated by a factor α (default 0.01): treating them as "off" loses ~α-proportional mass | case covered by the same GELU/SiLU logic |
| Tanh, Softsign, Hardtanh | `abs` | sign = direction, absolute value = intensity; negative mass ~50%: `relu` would destroy half of the signal | tanh: 50.2% negative mass |
| Identity, Linear, unmapped modules | **no default: explicit choice mandatory** (error with list of options and semantics) | negative mass ~50% and sign semantics unknown; the strategies are NOT interchangeable (anchor: 0.0856 vs 0.7362 differ by 8.6×) | identity: 50.2% |

**Documented caveat for GELU/SiLU:** with inputs concentrated near zero
(σ≈1) the `relu` strategy discards 15–24% of the absolute mass. The caveat
goes in the table's docstring, not silenced: for inputs at a larger
scale the fraction drops quickly (2.6%/6.9% at σ=3).

---

## D4 — Uncertainty of the estimate (Q5 / C4, IN SCOPE by user decision): full-pipeline bootstrap, normal interval

**[REOPENED AND RE-DECIDED after internal adversarial verification
(AI-assisted) — replaces the closed-form choice of the first version.]**
The verification showed that exp4 had validated the closed-form on a
problem strictly easier than the real pipeline: categorical data with FIXED
bins. In the real pipeline the quantile edges are re-estimated from the
same sample, and that variance is invisible to the Basharin formula.

**Adopted formula** (batch-wise entropy only; see below for why):

```
for r = 1..R (default R = 200):
    x*_r  = resample with replacement of the N RAW values
    quantile edges RE-estimated on x*_r → winsorized histogram → H_MM,r
sd_boot = sample standard deviation of { H_MM,r }
CI 95%  = H_MM ± 1.96 · sd_boot     [normal interval centered on the
                                     point estimate, NOT percentile]
```

**Evidence (exp4b).** Coverage of the nominal 95% CI on the FULL pipeline
(continuous data → per-sample edges → winsorization → H_MM),
reference = analytical binned-clipped entropy at the population quantiles,
200 replications, R = 200:

| dist | N | closed-form | boot-perc | boot-basic | **boot-normal** | s_claim/s_act |
|---|---|---|---|---|---|---|
| normal | 200 | 78.0% | 91.0% | 47.5% | **94.5%** | 0.688 |
| normal | 1000 | 80.0% | 97.0% | 76.0% | **94.5%** | 0.697 |
| normal | 5000 | 88.0% | 97.5% | 90.5% | **96.0%** | 0.729 |
| uniform | 200 | 99.5% | 4.0% | 31.5% | **100.0%** | 1.311 |
| uniform | 1000 | 100.0% | 17.0% | 33.0% | **100.0%** | 1.367 |
| uniform | 5000 | 96.5% | 40.5% | 47.0% | **97.0%** | 1.083 |
| exponential | 200 | 64.5% | 98.0% | 69.0% | **91.0%** | 0.481 |
| exponential | 1000 | 62.5% | 93.0% | 79.0% | **90.5%** | 0.457 |
| exponential | 5000 | 69.0% | 95.0% | 89.5% | **95.5%** | 0.499 |

Gate (≥85% on ALL cells: 3 distributions × N ∈ {200, 1000, 5000}):
**only boot-normal is eligible** (min 90.5%). Rejections:

- **closed-form (Basharin), min 62.5%**: the ratio declared-stderr /
  real-sd ranges between 0.46 (exponential) and 1.37 (uniform) and does NOT
  converge to 1 as N grows → the deficit is structural (variance from
  re-estimating the edges), not small-sample. Note: on exp4 (fixed bins)
  the same formula covered 87.5–99.5% — the difference IS the edge
  variance.
- **boot percentile, min 4.0%**: collapses at the entropy ceiling
  (uniform), same mechanism documented in exp4.
- **boot basic, min 31.5%**.
- **calibrated inflation of the stderr: rejected without testing** — the
  required factor depends on distribution and N in both directions
  (0.46–1.37), so no principled constant exists.

Why boot-normal works where the percentile fails: sd_boot estimates well
the *scale* of the variability (including that of the edges) even when
the *location* of the replicates is biased downward near the ceiling;
the normal interval recenters on the point estimate H_MM, the percentile
does not.

**API consequence (documented, not silent).** The CI requires the RAW
values (each replicate must re-estimate the edges), so it cannot be derived
from the counts alone, and it costs R full histograms. Therefore:
`BatchEntropy` does NOT carry a per-batch stderr (the Basharin one is
removed: a standard error under-calibrated by 30–50% is worse than no
number); the uncertainty is obtained on demand with
`confidence_interval(activations,
...)`. The hooks do not retain the activations (memory), so the CI is
an explicit analysis on the activation the user passes, not an
automatic by-product of tracking.

**Declared domain of validity (refined by a second internal adversarial
verification (AI-assisted) of the chosen method, 8 distributions ×
N ∈ {100,128,200,1000,5000}, up to 2000 replications on the critical
cells):**

- The coverage is to be read as "**≥ 85%, conservative**", not exact 95%:
  measured 86–100% on the validated grid, with one-sided miss patterns
  (sparse data: misses upward, bias/sd down to −0.48; lognormal: downward,
  +0.29). On uniform it over-covers (sd_boot/sd_real ≈ 1.7).
- Validity floor **N ≥ 200** — the `reliable=True` flag at N ≥ 90 does NOT
  transfer to the CI: on sparse/skewed data at N=100–128 the coverage
  drops to 80.8–83.3%.
- **Extreme sparsity (>90% of the mass in one bin): floor N ≈ 400** — at
  95% zeros and N=200 the coverage is 83.3% ± 0.6 (violates the gate), it
  recovers from N≈350–400. Detectable at runtime from the histogram itself:
  `confidence_interval` emits a warning when p_max > 0.9 and N < 400.
- The interval is **truncated to [0, log(min(N,B))]**: on nearly uniform
  data the raw upper limit exceeded the theoretical maximum in ~100% of
  the runs; truncation to a bound that contains the estimand never reduces
  the coverage (it keeps hi ≥ H_MM for the Miller-Madow overshoot at the
  ceiling).
- Bootstrap replicates can be degenerate on real sparse data
  (0.7% at 95% zeros, N=200): the library uses the convention H=0 (same
  as D5), not an error.
- Student's t (df=3), lognormal σ=1.5, bimodal: PASS (87.5–96%); the
  closed-form on the same cells collapses to 37–58% — confirms the
  rejection.

**Documented blind spot (unchanged from exp4):** on a degenerate sample
all replicates are identical → sd_boot = 0 → CI of zero width,
spurious "certainty". The reliability flag CANNOT be replaced by the
uncertainty: they are complementary.

**For the per-example entropy there is no sampling stderr**: for each
example it is a deterministic function of the activation vector, not an
estimate from N samples. Uncertainty applies only to aggregates over the
batch (e.g. SEM of the mean = std/√N). The API must reflect this
distinction, not offer a fictitious per-row stderr.

---

## D5 — Edge-case policies (A7, C3)

| Case | Behavior | Rationale |
|---|---|---|
| N = 0 (empty tensor) | immediate `ValueError`, for BOTH metrics | the entropy of an empty sample is not defined; no value or flag would represent it honestly. *(Row added from internal adversarial review (AI-assisted): it was missing from the first version; the scripts in validation/ behaved inconsistently on this case — the library does not.)* |
| NaN or ±Inf in the input | immediate `ValueError` | the entropy of data containing NaN/Inf is not defined; any silent handling would mask a bug in the observed network |
| N = 1 (batch-wise) | H = 0, `reliable=False`, reason `"N=1"` | a single sample is degenerate by construction |
| all values identical | H = 0, `reliable=False`, reason `"degenerate"` | zero quantile range; mathematically H=0 is correct but indistinguishable from true concentration without a flag (Defect 5) |
| all-zero row (per-example) | `NaN` + per-row `valid` mask | undefined distribution (0/0); NaN is honest, 0 would lie (verified exp3/exp5) |
| 4D+ tensor (conv, C3 — user decision: deferred) | core functions: `ValueError`; hooks/layers: the layer is SKIPPED with an explicit `warnings.warn` ("conv not validated in v1, layer skipped") | never emit unvalidated numbers (Q4 open); the runtime warning satisfies C3; a tracker on a CNN remains useful for the FC layers |
| 1D input to batch-wise | treated as (N, 1) | well-defined one-dimensional case |

*Note (2026-07-13, emerged while writing the numpy mirror, see D7): `_validate_input` of the torch core checks `isfinite` on the original dtype and then downcasts to float32 — a float64 that is finite but > ~3.4e38 passes the check and silently becomes inf in the computation. Open gap on the torch core (no test exercises it); the numpy mirror rejects this input. To be decided whether to align the torch core (not decided autonomously, rule 05).*
**→ CLOSED by D11 (2026-08-03):** user decision, torch core aligned with the numpy mirror (cast first, check after). Gate `validation/exp10_dtype_validation.py`, regression in `test_core_histogram.py` and `test_core_np_equivalence.py`. The defect was worse than noted here: it did not only produce an internal `inf` but an observable result `H=0.0` with `reliable=True`.

---

## D6 — Proofs of the bounds (A1 / A2)

### A1 — batch-wise: 0 ≤ H ≤ log(min(N, B))

Let `c_1..c_B` be the counts (`Σc_k = N`), `K = #{k : c_k > 0}`,
`p_k = c_k/N`.

1. **H ≥ 0**: every term −p_k·log p_k ≥ 0 because 0 < p_k ≤ 1.
   **H = 0 ⟺ K = 1**: if K=1 the only p is 1 and the term is 0; if K≥2
   there exists p_k ∈ (0,1) with a strictly positive contribution.
2. **H ≤ log K** (Gibbs' inequality against the uniform distribution
   over the K occupied bins): log K − H = D_KL(p ‖ u_K) ≥ 0, with
   equality ⟺ p uniform over the K occupied bins.
3. **K ≤ min(N, B)**: K ≤ B by construction; K ≤ N because each of the N
   samples falls in exactly one bin, so the occupied bins are at
   most N.

Hence 0 ≤ H ≤ log(min(N,B)). **Attainability of the maximum** (verified
in exp5, without clipping):
- N ≤ B: each sample in a distinct bin → p uniform over N bins →
  H = log N exactly (verified: N=8, B=32 → H = log 8 to 1e-5).
- N > B: K = B and N/B samples per bin are needed, hence B | N (verified:
  N=320, B=32, 10 per bin → H = log 32 to 1e-5).
- Under default winsorization the maximum drops to 99.92% of log B
  **for continuous data with uniform density** (corrected formula, scoping
  and discrete exception in D1: with mass on B discrete levels log B
  remains exactly attainable even with clipping active).

### A2 — per-example: 0 ≤ H ≤ log(M)

Let `a ∈ R^M` be the post-strategy row (hence a_i ≥ 0), `S = Σa_i`.

- **S = 0**: the distribution does not exist (0/0) → declared NaN (D5).
- **S > 0**: p_i = a_i/S is a distribution over M symbols; same Gibbs
  arguments: 0 ≤ H ≤ log M; H = 0 ⟺ p one-hot ⟺ a single a_i > 0;
  H = log M ⟺ p uniform ⟺ all a_i equal and > 0.

Verified in exp5: one-hot → 0 (‖·‖ < 1e-7), uniform → log 16 (1e-5),
zero row → NaN, inequality on 64 random rows for both surviving
strategies (`relu`, `abs`).

---

## D7 — Reference numpy core (E1 / E2): numpy mirror of the torch core, gated equivalence

**Adopted formula.** No new formula: D1–D6 unchanged. `entropylens/core_np.py` replicates the same operations in numpy float32 (sums, quantiles and bin index in float64 since D12); the only operations not bitwise-guaranteed are the quantiles (`np.quantile` vs `torch.quantile`, linear interpolation in both) and the summation order in the reductions.

**Evidence (exp6, torch 2.8.0+cpu / numpy 2.3.3 / Python 3.13).**

| Gate | Outcome |
|------|-------|
| G1 counts bitwise + identical degenerate mask (6 distributions × N ∈ {1,8,128,1000} × B ∈ {8,32} × 5 seeds) | PASS |
| G1b value_range branch (fixed range): counts bitwise and \|ΔH\| ≤ 1e-5 | PASS |
| G2 max \|ΔH\| | 4.77e-07 (gate 1e-5 nats) |
| G2 max \|Δnormalized\| | 1.79e-07 (gate 5e-6) |
| G3 reliable/reason identical | PASS |
| G4 per-example max \|ΔH\| (relu and abs), identical NaN positions | 7.15e-07 (gate 1e-6) |
| G5 CI: mean relative difference of the half-width | 0.060 (gate 10%); per-column pooled coverage 0.938, min column 0.90 on gauss N=400 (gate ≥0.85) |

**API consequence (documented, not silent).** The numpy CI uses `rng: np.random.Generator` in place of `generator: torch.Generator`: the two CIs are NOT reproducible from one another (different RNGs), the equivalence is statistical, not bitwise. `core_np` does not import torch/tensorflow/jax (AST test E2, analogue of B4). **Documented divergence on input validation:** `core_np` validates finiteness AFTER the cast to float32, so a float64 input that overflows to inf in float32 is rejected (ValueError), whereas the torch core checks `isfinite` on the original dtype and then silently computes on inf after the downcast — the numpy mirror is intentionally stricter. **→ SUPERSEDED by D11 (2026-08-03):** the torch core now casts first and checks after, the two cores reject the same input.

---

## D8 — Capture semantics per backend (E3 / E5): TF = inference-mode auxiliary model; JAX = capture_intermediates

**TF choice: auxiliary model, not hooks and not wrappers.** Keras has no forward hooks. `keras.Model(model.inputs[0], {name: layer.output})` shares the layer objects with the user model (exp7 G4: identity on d1/d2/d3) and does NOT touch the user model. Documented cost: one additional forward per observation, in `training=False` — with BatchNorm/Dropout the observed activations differ from the training step (exp7 G5 — negative anchor: the expected outcome of the gate is the DIFFERENCE, not a defect). Subclassed models (without `model.inputs`) are not supported in v1: explicit error, no silent fallback.

**JAX/Flax choice: capture_intermediates in the same forward.** The intermediates come out of the same pass (no second forward), in a mutable collection outside the grad path (exp8 G3). The captured activations of the `Dense` layers are PRE-activation (flax activations are functions, not modules — exp8 G4): NO strategy table is possible and the per-example entropy requires an explicit `strategy_map` per path (consistent with D3: no silent default). Without a strategy: batch-wise only, with a one-time warning per path at the first observation (same convention as hooks/C2: automatic degradation is never silent; the warning states that the per-example entropy requires an entry in strategy_map). Structural note (exp8 + probing): the `__call__` marker recurs at EVERY nesting level of the modules, not only at the root — the path walker must filter it recursively; the leaves are 1-tuples of arrays (unwrap `[0]`).

**Validated versions (do not extrapolate to the floor declared by the packages).** Gates measured on TF 2.21.0 / Keras 3.14 and JAX 0.10.0 / Flax 0.12.7 (CPU). The pair flax 0.8.0 + recent jax is BROKEN at import (`from jax.experimental import maps` removed): the floors in the pyproject extras must cite verified pairs, not theoretical minimums.

---

## D9 — Scope of the invariance guarantees per backend (E3 / E5)

| Backend | Guarantee | Evidence |
|---------|----------|----------|
| PyTorch | B1/B2/B3 bitwise via hooks (unchanged) | existing tests |
| TensorFlow | user model NEVER touched ⇒ outputs and gradients invariant by construction; frozen bitwise (baseline before construction of the aux, observe between 3 SGD steps) with `enable_op_determinism()` | exp7 G2/G3 PASS (TF 2.21.0) |
| JAX/Flax | outputs and gradients bitwise with/without capture, without jit AND under `jax.jit` (max \|d\| = 0.00e+00) | exp8 G1–G3 PASS (JAX 0.10.0 / Flax 0.12.7) |

**Scope of the jit claim (scoping, honesty).** Gate G2 compares two INDEPENDENTLY jit-compiled closures (the real usage pattern: user forward vs instrumented forward), on CPU, stable over 5 consecutive reruns: it is empirical evidence on these versions, NOT a guarantee of the JAX language. A bump of jaxlib/XLA can change fusion/order of the reductions: rerun exp8 at every update of the jax/flax pins.

**Common caveat (documented).** The TF observation is inference-mode (D8): it is not a weakness of the invariance (the user's step is not touched) but a measurement semantics that differs from the training step with stochastic layers or with batch statistics.

---

## D10 — Per-example denominator (A2 / defect M1 from internal design review (AI-assisted)): division by `s`, without clamp

**Defect.** The implementation normalized each row with
`a / s.clamp(min=1e-38)` (numpy: `np.maximum(s, 1e-38)`). When the
post-strategy sum of the row drops below 1e-38 the row is divided by 1e-38
instead of by its own sum: `p` no longer sums to 1 and H collapses toward 0
while `valid` stays `True`.

**Adopted formula.**

```
s = Σ_i a_i         (a_i ≥ 0 by construction, post-strategy)
valid ⟺ s > 0
p_i = a_i / s       [no clamp on the denominator]
```

*Since D12 (2026-11-02): sum and division in float64, for the overflow
at the opposite end of the scale.*

`s` is a sum of non-negative terms, so `s ≥ max(a)`: the test `s > 0` is
exact and there is no row with non-zero elements and zero sum. The clamp
was therefore a defense against an impossible case that created a real one.

**Evidence (exp9, gate G1 declared before the sweep).** Uniform row of
M=8 — the **maximum**-entropy case — as the scale of the elements varies,
error `|H − log M|`:

| element scale | C0 clamp (current) | C1 without clamp | C2 rescale/max | C3 float64 |
|---|---|---|---|---|
| 1e-38 | 2.33e-07 | 2.33e-07 | 2.33e-07 | 5.71e-09 |
| 1e-39 | **2.37e-01** | 2.33e-07 | 2.33e-07 | 5.71e-09 |
| 1e-40 | **1.71e+00** | 2.33e-07 | 2.33e-07 | 5.71e-09 |
| 1e-44 | **2.08e+00** | 2.33e-07 | 2.33e-07 | 5.71e-09 |

At 1e-40 the defective code reads H = 0.368412 (true log 8 = 2.079442) with
`valid=True`: a uniform row is reported as nearly one-hot, that is
exactly the false signal of hyper-specialization of Defect 2.
Gates passed by C1/C2/C3: G1 (sweep above), G2 (zero row → NaN + invalid,
D5 convention intact), G3 (D3 anchors unchanged: abs 0.7362, relu 0.0856),
G4 (one-hot → 0), G5 (mixed-scale battery, no NaN/Inf in valid rows).
G6 requires that C0 fail G1 — verified, the gate catches the defect and
does not pass vacuously.

**Choice: C1 (no clamp).** The three candidates are indistinguishable on
the gates; C1 wins on minimality — zero added operations, whereas C2 adds
one reduction per row and C3 doubles the memory of the computation. C1
**restores the reference formula already validated**:
`validation/common.py:147` divided by `s` without clamp. The clamp was
introduced during implementation (iteration 3) and no existing gate probed
subnormal rows — which is why exp9 exists.

**Internal consistency (motive for the decision).** `shift` was removed in
D3 on the rationale that «a strategy that turns the maximum-entropy case
into an undefined case does not measure what it claims to measure». The
clamp turned the same maximum-entropy case into H≈0.37 declared valid: same
class of defect, same standard, same conclusion.

**Scope.** The threshold is the scale of `s`, not that of the individual
elements: with large M a row can exceed 1e-38 in sum while having much
smaller elements. A regime attainable with collapsing networks,
post-softmax activations in the tails, or very small weights. Frozen in
`tests/test_core_perexample.py::test_underflow_uniform_row` (6 scales × 2
strategies, of which 1e-30 and 1e-38 as controls) and in the numpy twin in
`test_core_np_equivalence.py` (4 scales × 2 strategies).

---

## D11 — Input validation order (A7 / E1 / defect M5 from internal design review (AI-assisted)): cast first, check after

**Defect.** `_validate_input` of the torch core checked `torch.isfinite`
on the ORIGINAL dtype and downcast afterwards:

```
if not x.dtype.is_floating_point: x = x.float()
if not torch.isfinite(x).all(): raise ValueError(...)
return x.float()                      # <- l'overflow avviene QUI
```

A float64 that is finite but above the float32 maximum (~3.4028e38) passes
the check and becomes `inf` in the returned tensor. **It raises no error**:
`batch_wise_entropy` reports H = 0.0 with `reliable = True` — that is,
"maximally concentrated neuron, no a priori alarm". Same class of
false signal as Defect 2 and D10.

The numpy mirror already cast BEFORE checking (D7) and rejected the same
input: the two cores had different input contracts. The gap had been
noted in D5 (note of 2026-07-13) as an open point, not decided autonomously
(rule 05).

**Adopted formula (user decision, 2026-08-03 — option 1).**

```
x = x.float()                          # cast per primo
if not isfinite(x).all(): ValueError   # un solo controllo, dopo il cast
return x
```

A single post-cast check catches BOTH causes: an original NaN/Inf
stays NaN/Inf after the cast, so a second pass is not needed.
The error message also explicitly names the overflow case.

**Alternatives considered and discarded** (recorded because they are part
of the decision):
- *Keep the torch core permissive and document the divergence*:
  discarded — it would leave the library returning `H=0, reliable=True` on
  overflowed input, exactly the failure mode just corrected in D10,
  and A7 would stay green on top of a known hole.
- *Check before AND after the cast, with distinct messages* ("contains
  NaN/Inf" vs "exceeds the float32 range"): discarded by the user — the
  only advantage is diagnostic and it costs a second `isfinite` reduction
  on every call.

**Evidence (exp10).** Gates declared before the sweep:

| Gate | Content | Outcome |
|---|---|---|
| G1 | overflowing float64 → ValueError on 3 entry points × 2 cores (6 cases) | PASS |
| G2 | no regression: NaN, ±Inf, empty, 4D, 1D still rejected | PASS |
| G3 | ordinary float64 still ACCEPTED and bitwise equal to the equivalent float32 (including 1e37, large but representable) | PASS |
| G4 | float32 path: D3 anchors (0.7362 / 0.0856) and D10 subnormal uniform row (1e-40 → log 8) unchanged. At the D11 commit (4a6b29f) also bitwise equal to pre-D11; D12 then deliberately changed the float32 intermediates, a change gated by exp11 G4 | PASS |
| G5 | the pre-fix order (frozen in exp10 as `c0_validate`) MUST fail G1 — re-run at every run: it accepts the input and returns inf after the cast (pre-fix the library gave H=0.0, reliable=True) | PASS |
| G6 | the two cores agree on accept/reject over the whole battery (7 cases) | PASS |

**Methodological note (bug in the gate, not in the library).** The first
version of G3 compared the per-example entropies with `torch.equal`, which
is False in the presence of NaN by definition: with the `relu` strategy, 7
rows out of 256 were entirely negative → zero sum → legitimate NaN (D5).
The gate reported a non-existent float64/float32 divergence. Corrected to a
NaN-aware comparison (identical masks + finite values bitwise equal).
Recorded because it is the second case in which a gate produced a false
signal: the gates themselves must be read with the same distrust as the
code.

**Correction (2026-11-03, internal multi-agent code review, AI-assisted).**
In the first version G5 was an unconditional `check(..., True)`: after the
fix it printed "already fixed" and could not fail. G4 declared a bitwise
identity with the pre-fix that it verified only to tolerance, and its
"D10 regression" row used scale 0.25, where the pre-D10 code also gives
log 8. Both corrected (G5 with a frozen baseline like exp9/exp11, G4
reformulated and row at 1e-40). Third case of a gate that signaled
something false. The drift of the numpy mirror was also corrected: the
error message now names the overflow case like the torch one, and the cast
is in `np.errstate(over="ignore")`. Previously, with
warnings-as-errors, `core_np` raised `RuntimeWarning` instead of
`ValueError`; it is now frozen in
`test_overflow_rejection_same_error_in_both_cores`.

**API consequence (declared, not silent).** Observable behavior
change: float64 inputs with values beyond the float32 range, previously
accepted (and silently wrong), now raise `ValueError`. It is a
correction, not a breakage: those values produced meaningless numbers.
An activation that overflows float32 signals a problem in the observed
network — surfacing it is the purpose of the tool.

---

## D12 — Overflow in the intermediates on finite float32 input (A2 / A7 / E1, follow-up to D10/D11): sums, quantiles and bin index in float64

**Defect.** D11 rejects at the cast the values outside the float32 range.
Values *inside* the range still overflow further on, in three places, in
both cores:

- *per-example*: `s = a.sum(1)` → `inf` for a large row; `p = a / inf = 0`
  → H = 0 with `valid=True`. Uniform row of 8 elements at 1e38: H = −0.0
  (true log 8 = 2.0794).
- *batch-wise, bin index*: `hi − lo` and `xc − lo` → `inf`;
  `(xc − lo) / width` = NaN; the NaN index ends up in bin 0 for every
  sample → H = 0 with `reliable=True`. Columns of integers × 2^118 (finite,
  max ~3.3e38): H = 0.0 (true 3.4617 / 3.4633 / 3.4667).
- *batch-wise, quantile interpolation*: `a + w·(b − a)` with
  `b − a = inf` when the two interpolated order statistics straddle
  zero near the maximum → `lo = +inf`, column marked
  degenerate. Honest flag, wrong answer: H = 0, `reliable=False`
  (true 0.0844, `reliable=True`).

Same class of false signal as Defect 2, D10 and D11. Found in
code review of the D10/D11 branch (2026-10-29): exp10 covered only the
cast, no gate probed finite values near the float32 maximum.

**Adopted formula (user decision, 2026-11-02 — option 1 of 3).**

```
per-esempio:  s = Σ_i float64(a_i);   p_i = float32(float64(a_i) / s)
batch-wise:   x64 = float64(x);  lo, hi = quantile(x64);  width = hi − lo
              idx = min(floor((clip(x64, lo, hi) − lo) / width · B), B − 1)
```

The rest (log, Miller-Madow, normalization) stays in float32. Float64
holds the difference of any two float32 values and the sum of up to ~1e270
of them: no finite float32 input can overflow these intermediates.
Side effect: the subnormal sums of D10 become normal numbers.

*Perimeter.* The proposed option named "sums and bin-index arithmetic".
The exp11 sweep showed that the quantile interpolation also overflows
(G3): same cause, same remedy, included in option 1 and
declared here rather than left unsaid.

**Alternatives considered and discarded** (recorded because they are part
of the decision):
- *Rescale per row / per column by the maximum, in float32*: passes the
  same gates with more code (one more reduction per row and per column,
  plus the `value_range` branch to be handled separately).
- *Reject in validation every input whose intermediates could
  overflow* (e.g. `max|x| > finfo(float32).max / M`): a second restriction
  of the input contract after D11, with a conservative bound that
  would reject perfectly computable inputs.

**Evidence (exp11).** Candidates C0 (current float32 arithmetic) and C1
(float64). Gates declared before the sweep:

| Gate | Content | Outcome |
|---|---|---|
| G1 | per-example, uniform rows 1e37 / 1e38 / 3e38 / ±3e38 × 2 strategies: \|H − log M_eff\| < 1e-5, `valid` | C1 PASS; C0 H = −0.0 from 1e38 upward |
| G2 | integers × 2^k (k = 60, 100, 118): H and `reliable` bitwise equal to k = 0 | C1 PASS; C0 H = 0.0 at k = 118 |
| G3 | quantile straddling zero: 2 × (−1.75), 126 × (+1.75), × 2^127 | C1 PASS (H = 0.084391); C0 H = 0, `reliable=False` |
| G4 | no regression on ordinary input (C1 vs C0): per-example max \|dH\| = 9.54e-07; batch-wise only edge flips (5/4160 columns, max 1 sample moved); D3 anchors and D10 subnormal row unchanged | PASS |
| G5 | C0 MUST fail G1, G2, G3 | PASS (it fails all three) |
| G6 | library, both cores via the public API: G1–G3, counts bitwise between the cores, \|dH\| ≤ 1e-6, `confidence_interval` with non-zero width on the G2 input | PASS (pre-fix: 9 FAIL) |

**Note on the edge flips (G4).** In float64 a sample that falls within the
float32 rounding of a bin edge can change bin: 5 columns out of
4160 in the battery, never more than one sample per column. It is not a
regression — in float32 that assignment was itself a rounding
artifact — and it is the reason why G4 is a constraint on the moves and
not a bitwise equality with the pre-fix behavior.

**Methodological note (circularity of E1; internal design review
(AI-assisted), round 1, row 18).** Before the fix the G6 check "the two
cores agree" PASSED: both cores were broken in the same way. Second
concrete case (after D10) in which the equivalence torch ≡ numpy says
nothing about correctness. The exp11 gates are anchored to exact references
(log M, invariance under powers of two), not to the twin core.

**Resynchronization of exp6.** The numpy candidate embedded in
`validation/exp6_np_equivalence.py` (convention: copied VERBATIM into
`core_np.py`) had not been updated with D10: it still contained the
`1e-38` floor and stayed green only because its battery has no subnormal
rows. With D12 the torch core moved to float64 and the float32 candidate
diverged through edge flips (G1 counts, max \|dH\| 1.74e-04).
Candidate resynchronized to `core_np.py` (D10 + D12; identity verified via
AST on `_quantile_hist_counts`, `_miller_madow`, `_validate_input`): exp6
green again with the same numbers as before (max \|dH\| 4.77e-07, pooled
coverage 0.938).

**Cost.** Transient float64 copies of the activations in the intermediates
(~2× memory in those steps), marked `ponytail:` in `core.py`. Time,
measured in code review (CPU, 128×512 and 256×4096, best of 7):
`batch_wise_entropy` ×1.20–1.21 (torch) / ×1.25–1.29 (numpy),
`per_example_entropy` ×1.27–1.28 / ×1.14–1.15. The cost falls on the
observation path, that is at every traced forward and for every layer. The
suite has inputs too small to show it. Platform constraint: the MPS backend
of PyTorch (Apple GPU) does not support float64, so
`per_example_entropy` on MPS tensors now raises an error. It is not
verified on hardware; GPU support is in any case already declared
unverified in the README. These two costs were missing from the first
version of this entry: they were reported to the user, who on 2026-11-03
confirmed option 1.

**Scope.** Activations finite but close to 3.4e38: diverging networks.
Rare, but it is precisely the case in which the tool must raise an alarm
and not a "concentrated neuron". Frozen in
`tests/test_core_perexample.py::test_overflow_uniform_row` (3 rows × 2
strategies), `tests/test_core_batchwise.py::test_overflow_power_of_two_scale_invariance`
(k = 60 and 100 as controls, 118) and `::test_overflow_quantile_straddle`.
Numpy twins in `test_core_np_equivalence.py`:
`test_per_example_overflow_equivalence` (3e38 row × 2 strategies) and
`test_batch_wise_overflow_equivalence` (G2 and G3 inputs; counts bitwise
between the cores and equal to the unscaled input, so a revert in both
cores is also caught). The exp6 numpy candidate is now bound to
`core_np.py` by gate G0 (AST identity), which closes the drift described
above.

---

## Map of decisions → checklist items

| Decision | Items closed (downstream of implementation + tests) |
|---|---|
| D1 | A3, C1 |
| D2 | A4, A6 (part), Defect 5 |
| D3 | A5, C2 |
| D4 | C4 |
| D5 | A7, C3 |
| D6 | A1, A2 |
| D7 | E1, E2 |
| D8 | E3 (part), E5 (part) |
| D9 | E3 (part), E5 (part) |
| D10 | A2 (regression), E1 (numpy mirror aligned) |
| D11 | A7 (known gap closed), E1 (input contract unified) |
| D12 | A2 (regression), A7 (finite input at the limits of float32), E1 (cores aligned; exp6 candidate resynchronized) |

B1–B4, D1–D3 (checklist) remain entirely the responsibility of the
implementation and of the test suite. Items E4, E6, E7 (Keras table, backend-agnostic report/viz, packaging/CI) do not derive from D7–D9 and likewise remain the responsibility of the implementation.
