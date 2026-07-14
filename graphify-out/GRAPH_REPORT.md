# Graph Report - c:\Users\sasha.risoluti\OneDrive - Interpublic\Documents\personale\entropylens  (2026-07-14)

## Corpus Check
- 24 files · ~30,208 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 416 nodes · 676 edges · 24 communities (20 shown, 4 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 27 edges (avg confidence: 0.71)
- Token cost: 9,200 input · 123,470 output

## Community Hubs (Navigation)
- EntropyTracker Hooks
- Math Decisions, Docs & CI
- Validation Estimator Library
- Torch Core Estimators
- JAX/Flax Capture
- Numpy Core Mirror
- Strategy Tables (torch + Keras)
- TF Auxiliary Tracker
- Compat & Report/Viz Modules
- Entropy Report API
- Batch-wise Entropy Tests
- Uncertainty CI Tests
- Exp6 Numpy Equivalence
- Exp4b Pipeline Bootstrap
- Histogram Machinery Tests
- Per-example Entropy Tests
- Dashboard Image Concepts
- Layer Wrapper Tests
- Reliability Limitation Tests
- Exp4 Uncertainty Experiment
- Exp8 Flax Capture PoC
- Exp7 TF Capture PoC
- Package Root

## God Nodes (most connected - your core abstractions)
1. `EntropyTracker` - 30 edges
2. `TFEntropyTracker` - 22 edges
3. `EntropyReport` - 20 edges
4. `JaxEntropyTracker` - 18 edges
5. `capture()` - 12 edges
6. `LayerRecord` - 11 edges
7. `_x()` - 11 edges
8. `Validation Experiments (exp1–exp8, frozen evidence)` - 11 edges
9. `entropylens` - 10 edges
10. `_mlp()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `Honest Limitations` --semantically_similar_to--> `Documented Limitations (frozen in tests)`  [INFERRED] [semantically similar]
  README.md → STATUS.md
- `main()` --calls--> `EntropyTracker`  [EXTRACTED]
  experiments/viz_demo.py → entropylens/hooks.py
- `test_b3_gradients_bitwise_invariant()` --calls--> `EntropyTracker`  [EXTRACTED]
  tests/test_hooks.py → entropylens/hooks.py
- `test_c3_conv_skipped_with_warning()` --calls--> `EntropyTracker`  [EXTRACTED]
  tests/test_hooks.py → entropylens/hooks.py
- `test_unreliable_layers_marked()` --calls--> `EntropyTracker`  [EXTRACTED]
  tests/test_report.py → entropylens/hooks.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **TF/JAX backend extension (iteration 4: D7–D9, E1–E7, CI backends job)** — math_decisions_d7_numpy_core_equivalence, math_decisions_d8_capture_semantics, math_decisions_d9_backend_invariance_guarantees, readme_numpy_core, readme_tf_entropytracker, readme_jax_entropytracker, _github_workflows_tests_backends_job [EXTRACTED 1.00]
- **E1–E7 backend exit checklist (all PASS, full run 2026-07-13)** — status_e1_numpy_core_equivalence, status_e2_core_np_independence, status_e3_tf_tracker, status_e4_keras_strategy_table, status_e5_jax_tracker, status_e6_np_records_report, status_e7_packaging_ci, status_checklist_registry [EXTRACTED 1.00]
- **Validation-first + adversarial verification cycle** — math_decisions_validation_experiments, math_decisions_adversarial_verification, math_decisions_d1_quantile_winsorized_binning, math_decisions_d2_reliability_criterion, math_decisions_d4_bootstrap_confidence_interval [EXTRACTED 1.00]
- **Reliability masking applied consistently across summary plot, layer '3' heatmap, and top-5 ranking (unreliable neurons greyed/excluded)** — experiments_entropy_dashboard_summary_line_plot, experiments_entropy_dashboard_layer3_heatmap, experiments_entropy_dashboard_top5_neurons, experiments_entropy_dashboard_reliability_masking [EXTRACTED 1.00]

## Communities (24 total, 4 thin omitted)

### Community 0 - "EntropyTracker Hooks"
Cohesion: 0.08
Nodes (32): EntropyTracker, Non-invasive tracking via forward hooks (mode 1: observe an existing model with, Attach entropy tracking to an existing model without modifying it.      tracke, entropylens: entropy-based statistical interpretability for PyTorch.  Measures, main(), old_minmax_entropy(), D1 (checklist): repeat the training experiment of 04_ARCHITECTURE_DRAFT with the, The defective old-draft estimator (Difetto 2), local copy for     comparison onl (+24 more)

### Community 1 - "Math Decisions, Docs & CI"
Cohesion: 0.09
Nodes (41): CI Backends Job (TF/JAX gates), CI Workflow (tests), CI Test Job (torch matrix), D1 Training Experiment (corrected vs defective estimator, same run), H_new: corrected batch-wise estimator (quantile-winsorized + Miller-Madow + log(min(N,B))), H_old: defective draft estimator (min-max binning, Difetto 2), Adversarial Verification (multi-agent review), D1 — Quantile-Winsorized Binning (+33 more)

### Community 2 - "Validation Estimator Library"
Cohesion: 0.08
Nodes (25): Tensor, basharin_stderr_from_counts(), entropy_from_counts(), hist_entropy_fixed(), hist_entropy_minmax(), hist_entropy_quantile(), knn_entropy_kl(), miller_madow_from_counts() (+17 more)

### Community 3 - "Torch Core Estimators"
Cohesion: 0.09
Nodes (22): batch_wise_entropy(), BatchEntropy, confidence_interval(), _miller_madow(), per_example_entropy(), PerExampleEntropy, _quantile_hist_counts(), Pure math core: tensor-in / tensor-out.  MUST NOT reference torch.nn (checklist (+14 more)

### Community 4 - "JAX/Flax Capture"
Cohesion: 0.15
Nodes (23): capture(), _flatten(), JaxEntropyTracker, JAX/Flax capture (checklist E5): entropy from flax capture_intermediates.  Des, One forward pass with intermediates: returns (outputs, intermediates)., {'Dense_0': {'__call__': (arr,)}} -> {'Dense_0': arr}; root skipped., Entropy tracker for flax intermediates.      Feed it the intermediates from ca, Record entropy for every 2D captured array; appends one record         per path (+15 more)

### Community 5 - "Numpy Core Mirror"
Cohesion: 0.12
Nodes (17): batch_wise_entropy(), BatchEntropy, confidence_interval(), _miller_madow(), per_example_entropy(), PerExampleEntropy, _quantile_hist_counts(), Backend-agnostic numpy mirror of entropylens.core (checklist E1/E2).  Formulas (+9 more)

### Community 6 - "Strategy Tables (torch + Keras)"
Cohesion: 0.09
Nodes (10): Activation -> negative-handling strategy table (checklist A5 / C2).  This table, Strategy for an activation module, by exact type (A5 table).      Raises Unmappe, strategy_for(), _ensure_tables(), Keras activation -> negative-handling strategy table (checklist E4).  Mirror o, Strategy for a Keras layer, by exact class or activation identity.      Raises, strategy_for_keras(), UnmappedActivationError (+2 more)

### Community 7 - "TF Auxiliary Tracker"
Cohesion: 0.19
Nodes (19): TensorFlow/Keras tracker (checklist E3): observation by auxiliary model.  Desi, One inference-mode observation pass; appends one record per layer., Entropy tracker for functional/Sequential Keras models.      Unlike the torch, TFEntropyTracker, _mlp(), E3: TF tracker — observation by auxiliary model, user model untouched., test_conv_4d_skipped_with_warning(), test_e3_user_model_output_invariant() (+11 more)

### Community 8 - "Compat & Report/Viz Modules"
Cohesion: 0.15
Nodes (20): Duck-typed array conversion so report/viz never import a framework., torch.Tensor (any device), np.ndarray, or array-like -> np.ndarray., to_numpy(), Aggregation and presentation (checklist A6 / Difetto 4).  Default view is ALWA, dashboard(), layer_trend_data(), neuron_matrix(), plot_layer_trend() (+12 more)

### Community 9 - "Entropy Report API"
Cohesion: 0.17
Nodes (10): EntropyReport, (layer_name, batch_wise_normalized_mean) in registration order:         normali, Raw nats, each value paired with the bound it is relative to.         Values wi, _np_records(), E6: report and viz consume records produced by core_np (no torch inside the rec, test_depth_trend_and_raw_on_np_records(), test_std_and_sem_pin_ddof1(), test_summary_on_np_records() (+2 more)

### Community 10 - "Batch-wise Entropy Tests"
Cohesion: 0.14
Nodes (7): _analytic_normalized_href(), A1 (bounds), A3 (outlier gate <10%), A4 (N-invariance <5%), plus the exp1/exp2 a, Difetto 2 regression: 1 outlier on 1000 gaussians moves H by <10%     (measured, Binned-clipped N(0,1) reference from validation/exp2., Difetto 1 regression: same distribution, 4 batch sizes, normalized     entropy w, test_a3_outlier_gate(), test_a4_n_invariance_of_normalized_entropy()

### Community 11 - "Uncertainty CI Tests"
Cohesion: 0.17
Nodes (10): _analytic_h(), C4: full-pipeline bootstrap CI (MATH_DECISIONS.md D4 as amended, validation/exp4, Scaled-down exp4b anchor: gaussian N=400, 40 reps, coverage of the     analytic, Adversarial finding: on near-uniform data the raw normal interval     exceeded l, Adversarial finding: at >90% single-bin mass the coverage floor     rises to N~4, Documented blind spot (D4): degenerate sample -> every bootstrap     replicate i, test_ci_coverage_full_pipeline_smoke(), test_ci_sparsity_warning_below_validated_floor() (+2 more)

### Community 12 - "Exp6 Numpy Equivalence"
Cohesion: 0.28
Nodes (10): BatchEntropy, _miller_madow(), np_batch_wise_entropy(), np_confidence_interval(), np_per_example_entropy(), PerExampleEntropy, _quantile_hist_counts(), EXP6 (E1): equivalenza numerica del core numpy candidato vs entropylens.core. (+2 more)

### Community 13 - "Exp4b Pipeline Bootstrap"
Cohesion: 0.18
Nodes (5): hist_rows(), mm_rows(), EXP4b (Q5 / C4 — D4 REOPENED): uncertainty on the FULL pipeline.  Adversarial ve, Quantile-winsorized histogram per ROW. m: (R, N) -> counts (R, B).      Same for, Vectorized Miller-Madow entropy per row. counts: (r, B) -> (r,).

### Community 14 - "Histogram Machinery Tests"
Cohesion: 0.25
Nodes (4): _histc_reference(), Internal histogram machinery: per-column quantile-winsorized counts must agree w, Scalar reference: identical to validation/common.py quantile_hist_counts., test_counts_match_histc_reference_per_column()

### Community 16 - "Dashboard Image Concepts"
Cohesion: 0.50
Nodes (8): Entropy Dashboard (training visualization), Entropy evolution over training steps/epochs, Layer '1' neuron-by-step normalized entropy heatmap (~64 neurons), Layer '3' neuron-by-step normalized entropy heatmap (~32 neurons, grey unreliable bands), Normalized entropy metric [0, 1], Reliability masking (grey = unreliable, 'x' markers = partly unreliable), Per-layer mean entropy line plot (layers '1' and '3' over 30 steps), Top-5 neurons at final step listing (layer '3' neuron 29: 0.8723 highest; 2 unreliable excluded)

### Community 18 - "Reliability Limitation Tests"
Cohesion: 0.29
Nodes (7): _analytic_zeroinf_exp_ref(), Frozen documentation of the reliability flag's KNOWN limitations (adversarial re, Population binned-clipped normalized entropy of the mixture     pi*delta_0 + (1-, zero-inflated 90% + Exp(1) at N=128, B=32: reliable=True but the     mean normal, N=50 < 90: flag red even though the corrected estimate is decent     on gaussian, test_flag_conservative_side_is_safe_direction(), test_flag_green_while_biased_on_sparse_data()

### Community 19 - "Exp4 Uncertainty Experiment"
Cohesion: 0.29
Nodes (5): bootstrap_cis(), mm_rows(), EXP4 (Q5 / C4, user decision: IN SCOPE v1): uncertainty method choice.  *** SUPE, Vectorized Miller-Madow entropy per row. counts: (r, K) -> (r,)., Returns (percentile CI, basic CI). The basic/reversed CI     [2*H_hat - q97.5, 2

### Community 21 - "Exp7 TF Capture PoC"
Cohesion: 0.50
Nodes (3): make_mlp(), EXP7 (E3): cattura TF via modello ausiliario — invarianza per costruzione.  Pr, train()

## Knowledge Gaps
- **7 isolated node(s):** `H_old: defective draft estimator (min-max binning, Difetto 2)`, `entropylens`, `CI Test Job (torch matrix)`, `E1 — Numpy Core Equivalence (PASS, iter. 4)`, `E2 — core_np Framework Independence (PASS, iter. 4)` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EntropyReport` connect `Entropy Report API` to `EntropyTracker Hooks`, `Compat & Report/Viz Modules`, `JAX/Flax Capture`, `TF Auxiliary Tracker`?**
  _High betweenness centrality (0.107) - this node is a cross-community bridge._
- **Why does `EntropyTracker` connect `EntropyTracker Hooks` to `Entropy Report API`, `JAX/Flax Capture`?**
  _High betweenness centrality (0.083) - this node is a cross-community bridge._
- **Why does `LayerRecord` connect `JAX/Flax Capture` to `EntropyTracker Hooks`, `Entropy Report API`, `TF Auxiliary Tracker`?**
  _High betweenness centrality (0.074) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `EntropyTracker` (e.g. with `LayerRecord` and `EntropyReport`) actually correct?**
  _`EntropyTracker` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `TFEntropyTracker` (e.g. with `LayerRecord` and `EntropyReport`) actually correct?**
  _`TFEntropyTracker` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `EntropyReport` (e.g. with `EntropyTracker` and `JaxEntropyTracker`) actually correct?**
  _`EntropyReport` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `JaxEntropyTracker` (e.g. with `LayerRecord` and `EntropyReport`) actually correct?**
  _`JaxEntropyTracker` has 3 INFERRED edges - model-reasoned connections that need verification._