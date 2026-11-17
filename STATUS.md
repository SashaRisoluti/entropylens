# STATUS — Status registry of the project's exit checklist

Reference run for iterations 1–3: `pytest tests/ -q` →
**52 passed** (single run, D3) + `validation/exp1..exp5 + exp4b` all
exit 0, 2026-07-10, torch 2.8.0+cpu, Python 3.13 (later full
runs are recorded below, in order). Cycle: iteration 1 = formula validation
(exp1–exp5, MATH_DECISIONS.md); iteration 2 = internal multi-agent
adversarial verification (AI-assisted) and amendments (D4 re-decided →
exp4b, D6/D1 formula corrected, D5 N=0 row, D2 re-scoping of the flag);
iteration 3 = TDD implementation (Tasks 0–11 of the plan) incorporating the
fixes from the second internal adversarial verification (AI-assisted) (CI
sparsity warning, CI truncation, regression tests for the limitations).

Full run of 2026-07-13 — `pytest tests/ -q` → **142 passed**, 9 validation
scripts exit 0 (exp1–exp5, exp4b, exp6–exp8), torch 2.8.0+cpu / TF
2.21.0 / JAX 0.10.0 / Flax 0.12.7 / Python 3.13, iteration 4 (TF/JAX
backends, D7–D9).

Full run of 2026-08-03 (before D11) — `pytest tests/ -q` → **163 passed**,
10 validation scripts exit 0 (exp1–exp9), same versions, iteration 5
(D10 correction of the per-example denominator, from internal design
review (AI-assisted)). No previously PASS item has regressed. *(Entry
restored on 2026-11-03: the D11 commit had overwritten it instead of adding
the next one.)*

Full run of 2026-08-03 (after D11) — `pytest tests/ -q` → **168 passed**, 11 validation
scripts exit 0 (exp1–exp10), same versions, iteration 5 (internal design
review (AI-assisted): D10 per-example denominator, D11 input validation
order, outlier-magnitude sweep in exp1, CI wiring of all validation
scripts). No previously PASS item has regressed.

Full run of 2026-11-02 — `pytest tests/ -q` → **181 passed**, 12 validation
scripts exit 0 (exp1–exp11), torch 2.8.0+cpu / numpy 2.5.3 / TF 2.21.0
/ JAX 0.10.0 / Flax 0.12.7 / Python 3.13.7, iteration 5 (follow-up from code
review: D12 overflow in the intermediates on finite float32 input, exp6
numpy candidate resynchronized). No previously PASS item has
regressed.

Full run of 2026-11-03 — `pytest tests/ -q` → **183 passed**, 12 validation
scripts exit 0 (exp1–exp11), same versions as on 2026-11-02, after the
corrections from the internal multi-agent code review (AI-assisted;
Brooks-lint, 4 reviewers). The corrections:
- vacuous gates of exp10 (G5, G4) made effective;
- new gate exp6 G0 (candidate ≡ `core_np.py`, AST identity);
- NaN-safe accumulators in exp9/exp11;
- D11 contract of the numpy mirror aligned (same `ValueError` even with
  warnings-as-errors);
- numpy twin for the G3 case of exp11;
- CI loop over `exp*.py`.

No previously PASS item has regressed.

**Process limitation (from internal design review (AI-assisted),
unresolved):** no checklist item covers *device placement*. `core.py`
allocates several tensors without `device=`, so GPU support is neither
verified nor gated — see the "What's supported today" table in the README.
A new item is needed (proposal: B6, an AST check analogous to B4/B5) plus a
verification on CUDA hardware, absent from this environment and from CI.

| Item | Status | Iteration  | Notes |
|------|-------|------------|------|
| A1   | PASS  | 3          | D6 proof + `test_core_batchwise.py::test_a1_*` (inequality in 4 regimes, attainment N<B and B\|N, degenerate, N=1) |
| A2   | PASS  | 3, 5       | D6 proof + `test_core_perexample.py::test_a2_*` (one-hot→0, uniform→log M, zero row→NaN+valid, inequality on 64 rows). It. 5: D10 implementation defect corrected (denominator with clamp 1e-38 → uniform row read as H≈0.37 with `valid=True`); gate `validation/exp9_underflow.py`, regression `test_underflow_uniform_row` (6 scales × 2 strategies) + numpy twin. D12: row sum that overflowed float32 on finite elements (uniform row at 1e38 → H=0 with `valid=True`) now in float64; gate `validation/exp11_overflow.py`, regression `test_overflow_uniform_row` (3 rows × 2 strategies) + numpy twin |
| A3   | PASS  | 3          | <10% threshold approved by the user; `test_core_batchwise.py::test_a3_outlier_gate` measures ~0.2% (gate exp1: 0.182%) |
| A4   | PASS  | 3          | `test_core_batchwise.py::test_a4_n_invariance_of_normalized_entropy`: 4 values of N, dev <5% vs analytical reference (exp2: ≤0.34%). Scope beyond N(0,1): see D2 counterexamples + `test_reliability_limitations.py` |
| A5   | PASS  | 3          | table with rationales in `strategies.py` (D3); `test_strategies.py` (mappings, non-empty rationales, explicit error for unmapped modules, no silent subclass) |
| A6   | PASS  | 3          | report normalized-by-default, nats only with an explicit bound: `test_report.py` |
| A7   | PASS  | 3, 5       | D5 policies (including N=0 → ValueError from internal adversarial review (AI-assisted)): `test_core_histogram.py` (NaN/Inf, 4D, empty), `test_core_batchwise.py` (N=1, degenerate, 1D), `test_core_perexample.py` (zero row, 1D rejected). It. 5: closed the known gap from the D5 note of 2026-07-13 (float64 > float32 max accepted → H=0.0 with `reliable=True`); D11 aligns the torch core with the numpy mirror, gate `validation/exp10_dtype_validation.py` (G1-G6), regression in `test_core_histogram.py` (3 entry points). D12: finite input near the float32 max overflowed in the intermediates (row sum, `hi − lo`, quantile interpolation → H=0 with green flag); intermediates in float64, gate exp11 G1-G6, regressions in `test_core_batchwise.py::test_overflow_*` and `test_core_perexample.py` |
| B1   | PASS  | 3          | `test_hooks.py::test_b1_output_bitwise_invariant` — torch.equal (exactly 0.0), also after detach |
| B2   | PASS  | 3          | `test_layers.py::test_b2_output_bitwise_identical` + bitwise gradients |
| B3   | PASS  | 3          | `test_hooks.py::test_b3_*` (bitwise gradients with/without tracker, entropy tensors detached) + no_grad in all core functions |
| B4   | PASS  | 3          | `test_b4_core_independence.py` — AST check: core.py does not import/use torch.nn; functions on pure tensors |
| C1   | PASS  | 3          | quantile-winsorized implemented (D1); counts bitwise-equal to the histc reference of exp1 (`test_core_histogram.py`) |
| C2   | PASS  | 3          | A5 table + `UnmappedActivationError` with explicit guidance; hooks: one-time warning and batch-wise-only tracking for unmapped layers |
| C3   | PASS  | 3          | user decision: conv deferred. Core: ValueError on 4D; hooks/layers: layer skipped with explicit warning (`test_hooks.py::test_c3_*`, `test_layers.py::test_conv_*`) |
| C4   | PASS  | 3          | user decision: IN scope. D4 re-decided after internal adversarial verification (AI-assisted): full-pipeline bootstrap with normal interval (exp4b: 90.5–100%; closed-form rejected, min 62.5%). Second verification: coverage "≥85% conservative", floor N≥200 (N≈400 with >90% sparsity → runtime warning), CI truncated to [0, log(min(N,B))]. `test_uncertainty.py` |
| D1   | PASS  | 3          | `experiments/train_experiment.py` + `EXPERIMENT_NOTES.md`: same run, corrected vs defective estimator — upward trend confirmed by both (+2.1% vs +3.1%); they diverge only with outliers (exp1) |
| D2   | PASS  | 3          | every A/B item has an automated test named in this table; reproducible suite in `tests/` |
| D3   | PASS  | 3, 5       | 52 tests in ONE single run (it. 3) + 6 scripts exit 0; re-verified it. 5: 183 tests + 12 scripts (exp1-exp11) in a single session, see the 2026-11-03 full-run block above |
| E1   | PASS  | 4, 5       | numpy core equivalent to the torch core (D7): `test_core_np_equivalence.py` (counts bitwise, \|dH\|<=1e-5, value_range branch included); full sweep in `validation/exp6_np_equivalence.py` (max \|dH\| 4.77e-07, pooled CI coverage 0.938). It. 5: input contract now unified between the two cores (D11, exp10 G6) — previously they diverged on overflowing float64; `test_dtype_contract_matches_torch_policy`. D12: both cores in float64 in the intermediates, counts still bitwise (exp11 G6, `test_batch_wise_overflow_equivalence`); exp6 numpy candidate resynchronized to `core_np.py` (it was stuck at pre-D10). Pre-fix, G6 "the cores agree" passed with both broken: E1 does not certify correctness (cf. D12) |
| E2   | PASS  | 4          | `test_b5_np_core_independence.py`: AST check — `core_np.py` and `record.py` do not import torch/tensorflow/jax/flax/keras (analogous to B4) |
| E3   | PASS  | 4          | `test_tf_tracker.py` (12 tests): user model never touched (shared aux model, D8); invariance frozen bitwise in `validation/exp7_tf_capture.py` G2/G3 with op-determinism; nested submodels and subclassed models explicitly excluded |
| E4   | PASS  | 4          | `test_strategies_tf.py` (7 tests): Keras table by class + activation identity; linear Dense and LeakyReLU with default slope (0.3 > 0.1) -> explicit error (D3); hard_tanh -> abs |
| E5   | PASS  | 4          | `test_jax_tracker.py` (10 tests): capture_intermediates same forward, bitwise also under jit (`validation/exp8_jax_capture.py` G1-G3, max \|d\| = 0); explicit strategy_map with one-time warning per path (D8 amended); FrozenDict and multi-sow handled |
| E6   | PASS  | 4          | `test_report_np_records.py` (5 tests): report/viz on numpy records without torch; aggregation semantically invariant but NOT bitwise (numpy summation order, <=1 ulp float32; reliable_fraction accumulated in float64; ddof=1 pinned by test) |
| E7   | PASS  | 4          | pyproject 0.2.0: numpy as base dependency, extras `tf`/`jax` with VALIDATED floors (jax>=0.10, flax>=0.12 — the flax 0.8 pair is broken at import); CI job `backends` (tf-cpu, jax cpu, exp6-8 exit 0) |

## Documented limitations (not regressions, frozen in tests)

- `reliable` flag: a necessary, NOT sufficient condition (D2): on sparse
  data (90% zeros) or heavy-tailed data (lognormal σ=2) it can be green with
  a 7–14% deviation for N∈[90, ~200] at B=32; B=8 mis-calibrated for heavy
  tails. Frozen in `test_reliability_limitations.py`. Open point
  for the user: raising the default threshold (trade-off).
- Bootstrap CI: "≥85% conservative", not exact 95%; floor N≥200
  (N≈400 under extreme sparsity, with runtime warning).
- Heavy tails (Cauchy): H_norm 4–5× more variable from seed to seed than the
  fixed range (D1) — mitigation: `value_range`.
- Winsorization: attainable maximum 99.92% of log B on continuous uniform
  density (asymmetric formula corrected in D1/exp1).
