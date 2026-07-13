# Graph Report - .  (2026-07-13)

## Corpus Check
- Corpus is ~21,095 words - fits in a single context window. You may not need a graph.

## Summary
- 267 nodes · 390 edges · 21 communities (17 shown, 4 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 25 edges (avg confidence: 0.8)
- Token cost: 0 input · 119,341 output

## Community Hubs (Navigation)
- EntropyTracker Hooks
- Validation Estimator Library
- Core Entropy Estimators
- Math Decisions & Docs
- Viz Dashboards
- Activation Layer Wrappers
- Batch-wise Entropy Tests
- Uncertainty CI Tests
- Entropy Report
- Exp4b Pipeline Bootstrap
- Histogram Machinery Tests
- Per-example Entropy Tests
- Viz Tests
- Dashboard Image Concepts
- Reliability Limitation Tests
- Exp4 Uncertainty Experiment
- Strategy Table Tests
- Core Independence Tests
- Exp2 Normalization Experiment
- Package Root

## God Nodes (most connected - your core abstractions)
1. `EntropyTracker` - 29 edges
2. `EntropyReport` - 11 edges
3. `batch_wise_entropy()` - 8 edges
4. `STATUS: project exit-checklist registry (A1-D3, all PASS)` - 8 edges
5. `confidence_interval()` - 7 edges
6. `_records()` - 7 edges
7. `dashboard()` - 7 edges
8. `_tracked()` - 7 edges
9. `entropy_from_counts()` - 7 edges
10. `D1: Quantile-winsorized binning (robust histogram estimator)` - 7 edges

## Surprising Connections (you probably didn't know these)
- `H_old: defective draft estimator (min-max binning, Difetto 2)` --semantically_similar_to--> `Difetto 2: min-max binning defect (outlier collapse)`  [INFERRED] [semantically similar]
  EXPERIMENT_NOTES.md → MATH_DECISIONS.md
- `Honest limitations (documented, tested, not hidden)` --semantically_similar_to--> `Documented limitations frozen as regression tests`  [INFERRED] [semantically similar]
  README.md → STATUS.md
- `test_empty_tracker_raises()` --calls--> `EntropyTracker`  [EXTRACTED]
  tests/test_viz.py → entropylens/hooks.py
- `CI Tests Workflow (pytest, Python 3.11-3.13, CPU torch)` --conceptually_related_to--> `STATUS: project exit-checklist registry (A1-D3, all PASS)`  [INFERRED]
  .github/workflows/tests.yml → STATUS.md
- `batch_wise_entropy: per-neuron entropy across the batch` --implements--> `D1: Quantile-winsorized binning (robust histogram estimator)`  [INFERRED]
  README.md → MATH_DECISIONS.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Validated formula specification D1-D6 for entropylens core** — math_decisions_d1_quantile_winsorized_binning, math_decisions_d2_miller_madow_normalization, math_decisions_d2_reliability_criterion, math_decisions_d3_negative_value_strategies, math_decisions_d4_bootstrap_confidence_interval, math_decisions_d5_edge_case_policies, math_decisions_d6_bound_proofs [EXTRACTED 1.00]
- **Adversarial verification revision cycle (decisions amended on evidence)** — math_decisions_adversarial_verification, math_decisions_d4_bootstrap_confidence_interval, math_decisions_d2_reliability_criterion, math_decisions_d1_quantile_winsorized_binning, math_decisions_d5_edge_case_policies [EXTRACTED 1.00]
- **Honest uncertainty and limitation reporting across docs** — readme_honest_limitations, status_documented_limitations, math_decisions_d2_reliability_criterion, math_decisions_d4_bootstrap_confidence_interval [INFERRED 0.85]
- **Reliability masking applied consistently across summary plot, layer '3' heatmap, and top-5 ranking (unreliable neurons greyed/excluded)** — experiments_entropy_dashboard_summary_line_plot, experiments_entropy_dashboard_layer3_heatmap, experiments_entropy_dashboard_top5_neurons, experiments_entropy_dashboard_reliability_masking [EXTRACTED 1.00]

## Communities (21 total, 4 thin omitted)

### Community 0 - "EntropyTracker Hooks"
Cohesion: 0.10
Nodes (24): EntropyTracker, LayerRecord, Non-invasive tracking via forward hooks (mode 1: observe an existing model witho, Attach entropy tracking to an existing model without modifying it.      tracker, main(), old_minmax_entropy(), D1 (checklist): repeat the training experiment of 04_ARCHITECTURE_DRAFT with the, The defective old-draft estimator (Difetto 2), local copy for     comparison onl (+16 more)

### Community 1 - "Validation Estimator Library"
Cohesion: 0.10
Nodes (22): Tensor, basharin_stderr_from_counts(), entropy_from_counts(), hist_entropy_fixed(), hist_entropy_minmax(), hist_entropy_quantile(), knn_entropy_kl(), miller_madow_from_counts() (+14 more)

### Community 2 - "Core Entropy Estimators"
Cohesion: 0.14
Nodes (24): batch_wise_entropy(), BatchEntropy, confidence_interval(), _miller_madow(), per_example_entropy(), PerExampleEntropy, _quantile_hist_counts(), Pure math core: tensor-in / tensor-out.  MUST NOT reference torch.nn (checklist (+16 more)

### Community 3 - "Math Decisions & Docs"
Cohesion: 0.14
Nodes (24): CI Tests Workflow (pytest, Python 3.11-3.13, CPU torch), D1 Training Experiment (corrected vs defective estimator, same run), H_new: corrected batch-wise estimator (quantile-winsorized + Miller-Madow + log(min(N,B))), H_old: defective draft estimator (min-max binning, Difetto 2), Multi-agent adversarial verification of MATH_DECISIONS, D1: Quantile-winsorized binning (robust histogram estimator), D2: Miller-Madow correction + log(min(N,B)) normalization, Reliability flag criterion (B-1)/(2N) <= 0.05*log(min(N,B)) (+16 more)

### Community 4 - "Viz Dashboards"
Cohesion: 0.21
Nodes (16): dashboard(), layer_trend_data(), neuron_matrix(), plot_layer_trend(), plot_neuron_heatmap(), _plt(), Post-hoc training visualization (optional extra: matplotlib).  Feed an EntropyTr, Neurons x steps heatmap of normalized entropy for one layer;     unreliable cell (+8 more)

### Community 5 - "Activation Layer Wrappers"
Cohesion: 0.13
Nodes (3): EntropyActivation, Drop-in activation wrappers (mode 2: tracking built into the architecture).  Eac, B2: drop-in layers leave output and gradients bitwise unchanged. Each subclass c

### Community 6 - "Batch-wise Entropy Tests"
Cohesion: 0.14
Nodes (7): _analytic_normalized_href(), A1 (bounds), A3 (outlier gate <10%), A4 (N-invariance <5%), plus the exp1/exp2 a, Difetto 2 regression: 1 outlier on 1000 gaussians moves H by <10%     (measured, Binned-clipped N(0,1) reference from validation/exp2., Difetto 1 regression: same distribution, 4 batch sizes, normalized     entropy w, test_a3_outlier_gate(), test_a4_n_invariance_of_normalized_entropy()

### Community 7 - "Uncertainty CI Tests"
Cohesion: 0.17
Nodes (10): _analytic_h(), C4: full-pipeline bootstrap CI (MATH_DECISIONS.md D4 as amended, validation/exp4, Scaled-down exp4b anchor: gaussian N=400, 40 reps, coverage of the     analytic, Adversarial finding: on near-uniform data the raw normal interval     exceeded l, Adversarial finding: at >90% single-bin mass the coverage floor     rises to N~4, Documented blind spot (D4): degenerate sample -> every bootstrap     replicate i, test_ci_coverage_full_pipeline_smoke(), test_ci_sparsity_warning_below_validated_floor() (+2 more)

### Community 8 - "Entropy Report"
Cohesion: 0.24
Nodes (4): EntropyReport, Aggregation and presentation (checklist A6 / Difetto 4).  Default view is ALWAYS, (layer_name, batch_wise_normalized_mean) in registration order:         normaliz, Raw nats, each value paired with the bound it is relative to.         Values wit

### Community 9 - "Exp4b Pipeline Bootstrap"
Cohesion: 0.18
Nodes (5): hist_rows(), mm_rows(), EXP4b (Q5 / C4 — D4 REOPENED): uncertainty on the FULL pipeline.  Adversarial ve, Quantile-winsorized histogram per ROW. m: (R, N) -> counts (R, B).      Same for, Vectorized Miller-Madow entropy per row. counts: (r, B) -> (r,).

### Community 10 - "Histogram Machinery Tests"
Cohesion: 0.25
Nodes (4): _histc_reference(), Internal histogram machinery: per-column quantile-winsorized counts must agree w, Scalar reference: identical to validation/common.py quantile_hist_counts., test_counts_match_histc_reference_per_column()

### Community 12 - "Viz Tests"
Cohesion: 0.36
Nodes (8): viz: post-hoc training visualization. Honesty rules under test: cross-layer view, test_dashboard_saves_png(), test_empty_tracker_raises(), test_layer_trend_data_shapes_and_scale(), test_neuron_matrix_and_reliability_mask(), test_plot_functions_smoke(), test_top_neurons_ranked_reliable_only(), _tracked()

### Community 13 - "Dashboard Image Concepts"
Cohesion: 0.50
Nodes (8): Entropy Dashboard (training visualization), Entropy evolution over training steps/epochs, Layer '1' neuron-by-step normalized entropy heatmap (~64 neurons), Layer '3' neuron-by-step normalized entropy heatmap (~32 neurons, grey unreliable bands), Normalized entropy metric [0, 1], Reliability masking (grey = unreliable, 'x' markers = partly unreliable), Per-layer mean entropy line plot (layers '1' and '3' over 30 steps), Top-5 neurons at final step listing (layer '3' neuron 29: 0.8723 highest; 2 unreliable excluded)

### Community 14 - "Reliability Limitation Tests"
Cohesion: 0.29
Nodes (7): _analytic_zeroinf_exp_ref(), Frozen documentation of the reliability flag's KNOWN limitations (adversarial re, Population binned-clipped normalized entropy of the mixture     pi*delta_0 + (1-, zero-inflated 90% + Exp(1) at N=128, B=32: reliable=True but the     mean normal, N=50 < 90: flag red even though the corrected estimate is decent     on gaussian, test_flag_conservative_side_is_safe_direction(), test_flag_green_while_biased_on_sparse_data()

### Community 15 - "Exp4 Uncertainty Experiment"
Cohesion: 0.29
Nodes (5): bootstrap_cis(), mm_rows(), EXP4 (Q5 / C4, user decision: IN SCOPE v1): uncertainty method choice.  *** SUPE, Vectorized Miller-Madow entropy per row. counts: (r, K) -> (r,)., Returns (percentile CI, basic CI). The basic/reversed CI     [2*H_hat - q97.5, 2

### Community 18 - "Exp2 Normalization Experiment"
Cohesion: 0.40
Nodes (3): EXP2 (Q2 / Difetto 1 / A4): cross-batch-size normalization.  Same underlying dis, Bias-dominance reliability criterion., reliable()

## Knowledge Gaps
- **4 isolated node(s):** `entropylens`, `per_example_entropy: per-example entropy across neurons (mandatory strategy)`, `B1/B2/B3 bitwise-invariance guarantees (torch.equal, exact)`, `entropylens.viz: post-hoc matplotlib dashboards (layer trend, neuron heatmap, top-k)`
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EntropyTracker` connect `EntropyTracker Hooks` to `Entropy Report`, `Core Entropy Estimators`, `Viz Tests`?**
  _High betweenness centrality (0.145) - this node is a cross-community bridge._
- **Why does `hist_entropy_fixed()` connect `Validation Estimator Library` to `Core Entropy Estimators`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Why does `knn_entropy_kl()` connect `Validation Estimator Library` to `Core Entropy Estimators`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `EntropyReport` (e.g. with `EntropyTracker` and `LayerRecord`) actually correct?**
  _`EntropyReport` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `ValueError` (e.g. with `confidence_interval()` and `per_example_entropy()`) actually correct?**
  _`ValueError` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `entropylens`, `per_example_entropy: per-example entropy across neurons (mandatory strategy)`, `B1/B2/B3 bitwise-invariance guarantees (torch.equal, exact)` to the rest of the system?**
  _4 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `EntropyTracker Hooks` be split into smaller, more focused modules?**
  _Cohesion score 0.09523809523809523 - nodes in this community are weakly interconnected._