# EXPERIMENT_NOTES — D1: training experiment with the corrected estimator

Script: `experiments/train_experiment.py` (deterministic, fixed seeds).
Setup: MLP 20→64→32→2 (ReLU), synthetic binary classification (two
Gaussian blobs ±1, N=2000), Adam lr=1e-3, 30 full-batch epochs. At each
epoch:

- **H_new**: normalized batch-wise entropy of the first ReLU layer, the
  library's corrected estimator (quantile-winsorized + Miller-Madow +
  log(min(N,B)), MATH_DECISIONS.md D1/D2), via `EntropyTracker`.
- **H_old**: the draft's defective estimator (min-max binning, Defect 2),
  local copy, on the SAME activations of the same run.

## Result (full run in the script's log)

| epoch | loss | H_new (norm) | H_old (nats) |
|---|---|---|---|
| 0 | 0.6830 | 0.6408 | 2.0441 |
| 10 | 0.4258 | 0.6413 | 2.0529 |
| 20 | 0.2156 | 0.6477 | 2.0762 |
| 29 | 0.0929 | 0.6543 | 2.1071 |

Trend: **both estimators read a slight, monotonic increase** in the
batch-wise entropy during training — H_new +2.1% relative
(0.6408→0.6543), H_old +3.1% (2.0441→2.1071 nats).

## Reading (without over-claiming)

1. **The correction of Defect 2 does NOT overturn the empirical reading on
   this run**: the trend "entropy rises during training" observed with the
   defective code is also reproduced with the corrected estimator.
2. **Why they coincide here**: the two estimators diverge only in the
   presence of outliers or heavy tails (exp1: one outlier in 1001 → 99.7%
   collapse for min-max, 0.18% for quantile-winsorized). The activations
   of this network on clean Gaussian data do not produce extreme outliers,
   so min-max is not "triggered" in its failure mode. The correction
   protects the cases where the data are NOT this well-behaved — it does
   not change the easy cases.
3. **Comparison with the draft's observation (~2.0→~2.3 nats)**: the
   direction coincides; the amplitude does not (here +0.06 nats).
   Different architecture, task and duration: no conclusion about the
   amplitude is transferable.
4. **This is ONE synthetic run**, not a general law: no claim of the kind
   "entropy always increases during training" is supported (explicit
   project constraint on non-generalization).

Item D1 of the checklist: closed by this comparison (same run, two
estimators, difference in reading documented).
