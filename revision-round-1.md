# Revision Tracking — Round 1

## Paper Information

| Field | Value |
|-------|-------|
| Manuscript | entropylens (repo-as-methods-paper bundle: README, MATH_DECISIONS, 02/03, EXPERIMENT_NOTES, validation/, STATUS) |
| Revision Round | 1 |
| Date | 2026-08-03 |
| Previous Decision | Major Revision |
| Original commit | `51e63ea` (D10 fix, P0 items) |
| This round's commit | pending — see Summary of Changes |

Round 1 covers everything resolvable by documentation, validation-script,
and non-API code changes without new experiments or public-API changes.
Per the skill's anti-pattern 7 ("scope creep during revision — new content
requires explicit user approval") and this project's own rule (05_ITERATION_LOOP:
threshold/trade-off decisions are not made autonomously), items requiring a
new experiment, a public API change, or a paper-vs-library framing decision
are tracked as `UNRESOLVED — pending approval`, not silently implemented.

---

## Revision Tracking Table

| # | Issue | Reviewer | Type | Resolution | Location | Status |
|---|---|---|---|---|---|---|
| 1 | Per-example underflow: uniform row read H≈0.37 with `valid=True` at subnormal row sums | R1 (M1) | Major | Removed the `s.clamp(min=1e-38)` floor in both cores; `s` is a sum of non-negatives so `s>0` was already exact. New D10 decision entry, frozen regression (6 scales × 2 strategies, both cores) | `entropylens/core.py:217`, `core_np.py:146`, `MATH_DECISIONS.md` D10 | RESOLVED (prior commit `51e63ea`) |
| 2 | `reliable` is a per-batch scalar broadcast to per-neuron shape; structurally cannot fire on its own documented counterexamples (sparse/heavy-tailed, which are per-neuron) | R1 (M2), DA | Major | Not implemented — adding a data-dependent per-neuron signal (from `p_max`, `K̂`, already computed) changes `BatchEntropy`'s public contract | `entropylens/core.py:157-166` | **UNRESOLVED — pending approval** (API change) |
| 3 | `level=0.95` parameter implies a coverage guarantee the validation explicitly disclaims ("≥85%, conservative") | R1 (M3) | Major | Not implemented — renaming/restricting a public parameter is a breaking change | `entropylens/core.py:229` | **UNRESOLVED — pending approval** (API change) |
| 4 | Normalizer `log(min(N,B))` and attainable maximum `H*=0.9992·log B` (continuous case, D1) disagree; interaction undocumented | R1 (M4) | Minor | Not implemented this round — folds into item 2/3 API discussion | `MATH_DECISIONS.md` D1/D2 | DELIBERATE_LIMITATION (tracked, not actioned) |
| 5 | `_validate_input` checks `isfinite` before the float32 downcast (torch) vs. after (numpy) — a finite float64 > ~3.4e38 silently becomes `inf` in the torch core only, and A7 is marked PASS over this self-disclosed gap | R1 (M5) | Major | Already surfaced to user as BLOCKED in the prior round's plan (`fix-review-defects.md`); three options presented, recommendation given (align torch to numpy). No response received yet | `core.py:39-41`, `MATH_DECISIONS.md:343` | **UNRESOLVED — pending user decision** |
| 6 | No literature review or citations anywhere in the bundle | R2 (D1) | Major (if paper) | Not actioned — this is downstream of an unresolved fork: is this bundle a paper (needs citations, related work) or a library (doesn't)? See Editorial note below | — | **UNRESOLVED — pending scope decision** |
| 7 | Miller-Madow kept despite being weakest exactly in the regime (Np_k≪1) the document's own D2 counterexamples isolate; NSB/Chao-Shen/Grassberger never evaluated | R2 (D2) | Major | Not actioned — requires implementing and gating a second estimator family, a multi-day validation cycle of its own scale to D1-D6 | — | **UNRESOLVED — pending approval** (new validation cycle) |
| 8 | No baseline comparison (fraction-nonzero, variance, Gini) showing the estimator earns its complexity | R2 (D3) | Major | Not actioned — requires a real-model experiment, same fork as DA's strongest counter-argument (row 12) | — | **UNRESOLVED — pending approval** (new experiment) |
| 9 | No positioning against the information-plane / estimator-artifact literature (Tishby, Saxe et al.) | R2 (D4) | Major | Not actioned — same fork as row 6 | — | **UNRESOLVED — pending scope decision** |
| 10 | Normative spec (MATH_DECISIONS.md) is in Italian; README explicitly flags this as a barrier | R2 (D5) | Minor | Not actioned — full translation is ~450 lines of formula-dense text; substantial effort, own approval needed given size/cost | `MATH_DECISIONS.md` (whole file) | **UNRESOLVED — pending approval** (scale) |
| 11 | GPU/CUDA support unverified; `core.py` has 7 device-unsafe allocation sites; README silent on the question | R3 (P1) | Major | Documented truthfully (new "What's supported today" table states "not verified" + names the 7 sites' consequence). Code fix NOT applied — that's an API/behavior-surface change requiring a CUDA machine to verify, tracked separately | `README.md` "What's supported today" | PARTIAL — honesty gap closed, code gap open, **pending approval + hardware** |
| 12 | Cross-backend numbers not shown comparable when BatchNorm/Dropout make TF's separate inference-mode forward diverge from the training pass | R3 (P2) | Major | Documented explicitly in the new capability table (last row) — was implied by D8 but never stated as an open measurement gap. Experiment itself not run | `README.md` "What's supported today" | PARTIAL — documented, **UNRESOLVED experiment pending approval** |
| 13 | Validated applicability surface (backend × layer × activation) scattered across five documents | R3 (P3) | Minor | Added single intersection table | `README.md` new "What's supported today" section | RESOLVED |
| 14 | No decision rule for what a practitioner should *do* with a normalized-entropy reading | R3 (P4) | Minor | Not actioned — this is a product-design question (what pathologies to name, what thresholds to suggest), not a documentation gap; needs your input on what guidance you're willing to stand behind | — | **UNRESOLVED — pending user input** |
| 15 | `viz_demo.py` reproducibility of the flagship dashboard image is asserted, not gated in CI | R3 (P5) | Minor | Not actioned this round (folds into CI-wiring work below, but running matplotlib+full training in CI is a scope call: adds ~significant runtime for a smoke test) | `experiments/viz_demo.py` | DELIBERATE_LIMITATION (flagged, not actioned) |
| 16 | `shift`-removal principle ("a strategy that turns the max-entropy case into an undefined one doesn't measure what it claims") condemns the pre-fix clamp under its own logic | DA | Critical | Same root cause as row 1; D10 entry cross-references D3's rationale explicitly | `MATH_DECISIONS.md` D10 | RESOLVED (prior commit `51e63ea`) |
| 17 | `reliable=True` was redefined into unfalsifiability after counterexamples were found | DA | Critical | Same issue as row 2 — unresolved pending API-change approval | `core.py:44-60` | **UNRESOLVED — pending approval** |
| 18 | E1 (numpy≡torch) is circular: agreement isn't correctness if both share a spec bug (as row 1 proved) | DA | Major | Not actioned — re-anchoring `exp6` to analytic references (like exp1/exp2/exp5) rather than to the torch core is itself a validation-first cycle of real size (5 gates, 369-line script), not a quick patch | `validation/exp6_np_equivalence.py` | **UNRESOLVED — pending approval** (new validation cycle) |
| 19 | Confirmation bias: gate thresholds set by the same process that measured against them (D4's 85% floor introduced after 95% failed) | DA | Major | Acknowledged as a structural property of the process, not a specific document to edit. No action taken; flagging is the response | — | DELIBERATE_LIMITATION |
| 20 | "Bitwise identical" (non-perturbation) sits next to accuracy language and will be read as measurement precision | DA | Major | Not actioned — a wording audit across README's 8 uses; low effort but touches user-facing claims, held for this round's batch approval | `README.md` (8 occurrences) | **UNRESOLVED — pending approval** (small, listed for next batch) |
| 21 | The end-to-end evidence (EXPERIMENT_NOTES) shows the corrected and defective estimators agree on the training-run reading — the flagship contribution changes no conclusion in its only real-model test | DA | Major (strongest counter-argument) | Not actioned — this is exactly the missing experiment in row 8; a model/task where they diverge materially is the single highest-value addition per the original Phase-2 decision, and the biggest scope decision in this whole revision | `EXPERIMENT_NOTES.md` | **UNRESOLVED — pending approval** (new experiment, highest priority if pursued) |
| 22 | Outlier anchor (1000.0) is the extreme of its own range, not representative | DA | Minor | Added a 5σ/20σ/100σ/1000σ magnitude sweep to `exp1`; min-max degrades continuously (7.4%→99.7%), quantile-winsorized/fixed-range stay flat across three orders of magnitude — new evidence, not just the extreme point. Recorded in `MATH_DECISIONS.md` D1 | `validation/exp1_binning_outliers.py`, `MATH_DECISIONS.md` D1 | RESOLVED |
| 23 | (mechanical, self-identified, not from panel) README claimed `torch` as the "only runtime dependency"; `pyproject.toml` lists `numpy>=1.24` as a base dep since iteration 4 — an `import entropylens` following the README's install line would ImportError | — | Editorial | Fixed install instructions | `README.md` Install | RESOLVED |
| 24 | (mechanical) Version floors (`torch>=2.0`, `numpy>=1.24`, `tensorflow>=2.16`) are theoretical minimums, violating this project's own D8 rule ("floors must cite verified pairs, not theoretical minimums") already applied to jax/flax | — | Editorial | Documented the gap (README now states only the tested versions are exercised); did NOT change `pyproject.toml` floors — raising them is a real compatibility claim requiring testing against the low end, or accepting the known-untested risk, either way your call | `README.md` Install | PARTIAL — documented, **floor change pending approval** |
| 25 | (mechanical) `validation/` scripts other than exp6-8 were not wired into CI despite README claiming they "must stay green... part of the final gate" | — | Editorial | Wired all 7 remaining scripts (exp1-5, exp4b, exp9) into the `test` job (torch/numpy only, no new dependency); `backends` job unchanged | `.github/workflows/tests.yml` | RESOLVED |
| 26 | (mechanical) STATUS.md D3 row still said "52 test... 6 script" after the header block above it was updated to 163/10 in the prior commit | — | Editorial | Fixed the stale row | `STATUS.md` A2/D3 rows | RESOLVED |

---

## Per-row Commitment Ledger

```yaml
- concern_id: R1-M1
  commitment_extracted:
    - commitment_text: "Fix the per-example row-sum underflow"
      commitment_type: add_analysis
      required_evidence_type: prose_edit
      fulfillment_status: fulfilled

- concern_id: R1-M2
  commitment_extracted:
    - commitment_text: "Give the reliability flag a per-neuron, data-dependent component"
      commitment_type: other
      required_evidence_type: other
      fulfillment_status: not-fulfilled
      unfulfilled_rationale: "requires a BatchEntropy field addition (public API change); awaiting explicit approval, see Editorial Note"

- concern_id: R1-M3
  commitment_extracted:
    - commitment_text: "Reconcile confidence_interval's level parameter with measured coverage"
      commitment_type: other
      required_evidence_type: other
      fulfillment_status: not-fulfilled
      unfulfilled_rationale: "breaking parameter change; awaiting explicit approval"

- concern_id: R1-M5
  commitment_extracted:
    - commitment_text: "Resolve the float64-to-inf validation asymmetry between torch and numpy cores"
      commitment_type: other
      required_evidence_type: other
      fulfillment_status: not-fulfilled
      unfulfilled_rationale: "three options presented to user in a prior turn (fix-review-defects.md); no decision received yet"

- concern_id: R2-D1
  commitment_extracted:
    - commitment_text: "Add a literature review and citations"
      commitment_type: add_citation
      required_evidence_type: new_section
      fulfillment_status: not-fulfilled
      unfulfilled_rationale: "deferred to future work pending the paper-vs-library scope decision (Editorial Note)"

- concern_id: R2-D2
  commitment_extracted:
    - commitment_text: "Evaluate NSB/Chao-Shen against Miller-Madow on the D2 adversarial battery"
      commitment_type: add_experiment
      required_evidence_type: new_section
      fulfillment_status: not-fulfilled
      unfulfilled_rationale: "rejected for this round, reasons: full validation cycle at D1-D6 scale; recommend as a scoped follow-up if pursued"

- concern_id: R2-D3
  commitment_extracted:
    - commitment_text: "Baseline comparison against fraction-nonzero / variance / Gini"
      commitment_type: add_experiment
      required_evidence_type: new_section
      fulfillment_status: not-fulfilled
      unfulfilled_rationale: "deferred to future work, tied to the same real-model experiment as DA's strongest counter-argument (concern DA-1)"

- concern_id: R2-D4
  commitment_extracted:
    - commitment_text: "Position against the information-plane / estimator-artifact literature"
      commitment_type: add_citation
      required_evidence_type: discussion_paragraph
      fulfillment_status: not-fulfilled
      unfulfilled_rationale: "deferred pending the same scope decision as R2-D1"

- concern_id: R2-D5
  commitment_extracted:
    - commitment_text: "Translate MATH_DECISIONS.md to English or promote an English normative spec"
      commitment_type: other
      required_evidence_type: other
      fulfillment_status: not-fulfilled
      unfulfilled_rationale: "deferred to future work: ~450 lines, high cost, needs explicit go-ahead"

- concern_id: R3-P1
  commitment_extracted:
    - commitment_text: "Verify or fix GPU/CUDA support"
      commitment_type: add_clarification
      required_evidence_type: prose_edit
      fulfillment_status: partial
      unfulfilled_rationale: "honesty gap closed (README states 'not verified' + names the 7 unsafe sites); the code fix itself (thread device= through core.py, add a B6 AST gate, verify on CUDA hardware) is scoped and ready but not yet approved to execute"
    - commitment_text: "Fix the 7 device-unsafe allocation sites in core.py"
      commitment_type: other
      required_evidence_type: other
      fulfillment_status: not-fulfilled
      unfulfilled_rationale: "deferred to future work pending approval; no CUDA hardware in this environment to verify against"

- concern_id: R3-P2
  commitment_extracted:
    - commitment_text: "Measure cross-backend numerical comparability with BatchNorm active"
      commitment_type: add_experiment
      required_evidence_type: new_section
      fulfillment_status: not-fulfilled
      unfulfilled_rationale: "deferred to future work, new experiment"

- concern_id: R3-P3
  commitment_extracted:
    - commitment_text: "Add a single 'what's supported today' intersection table"
      commitment_type: add_section
      required_evidence_type: new_section
      fulfillment_status: fulfilled

- concern_id: R3-P4
  commitment_extracted:
    - commitment_text: "State a decision rule for what a practitioner should do with a reading"
      commitment_type: other
      required_evidence_type: other
      fulfillment_status: not-fulfilled
      unfulfilled_rationale: "product-design question requiring user input on what guidance to commit to; not a documentation gap"

- concern_id: DA-1
  commitment_extracted:
    - commitment_text: "Produce a real-model experiment where the corrected and defective estimators disagree materially"
      commitment_type: add_experiment
      required_evidence_type: new_section
      fulfillment_status: not-fulfilled
      unfulfilled_rationale: "deferred to future work: highest-value item in the original roadmap, but a genuinely new experiment requiring scope approval, not implementable as a documentation-only revision"

- concern_id: DA-2
  commitment_extracted:
    - commitment_text: "Give the reliability flag a data-dependent component (unfalsifiability)"
      commitment_type: other
      required_evidence_type: other
      fulfillment_status: not-fulfilled
      unfulfilled_rationale: "duplicate of R1-M2; same rationale"

- concern_id: DA-3
  commitment_extracted:
    - commitment_text: "Re-anchor exp6 to analytic references instead of the torch core"
      commitment_type: other
      required_evidence_type: other
      fulfillment_status: not-fulfilled
      unfulfilled_rationale: "deferred to future work: a validation-first cycle at D1-D6 scale, not a quick patch"

- concern_id: DA-4
  commitment_extracted:
    - commitment_text: "Sweep outlier magnitude instead of reporting only the 1000x anchor"
      commitment_type: add_analysis
      required_evidence_type: new_table
      fulfillment_status: fulfilled

- concern_id: DA-5
  commitment_extracted:
    - commitment_text: "Audit 'bitwise identical' wording for conflation with measurement accuracy"
      commitment_type: prose_edit
      required_evidence_type: prose_edit
      fulfillment_status: not-fulfilled
      unfulfilled_rationale: "batched with other small wording items, held pending approval alongside the API-touching decisions in this round"
```

---

## Editorial Note — the fork underneath rows 6, 9, 21

Rows 6, 8, 9, and 21 all resolve to the same open question: **is this
bundle being positioned as a paper (needs literature, needs a baseline,
needs the disagreement experiment to justify the estimator's complexity)
or as a library (documentation completeness and code correctness are the
bar, and "we chose a well-validated correction and documented its limits"
is sufficient)?** The Phase-2 editorial decision flagged this fork and
arbitrated it as: literature is CRITICAL if paper, MAJOR if library. This
round does not resolve it because it changes the shape of everything
downstream (which of P0-P3 even apply). Recommend deciding this before a
round 2.

## Summary of Changes

- Resolved this round: 7 (rows 1 restated, 13, 22, 23, 25, 26, plus the
  D10 fix already committed)
- Partial (documented, code/experiment pending): 3 (rows 11, 12, 24)
- Deliberate limitation (acknowledged, not actioned): 3 (rows 4, 15, 19)
- Unresolved, pending approval or decision: 13 (rows 2, 3, 5, 6, 7, 8, 9,
  10, 14, 17, 18, 20, 21)
- Files touched: `README.md`, `STATUS.md`, `MATH_DECISIONS.md`,
  `.github/workflows/tests.yml`, `validation/exp1_binning_outliers.py`
- No public API changed. No new dependency. No new experiment run.
- Full regression: `pytest tests/ -q` → 163 passed; all 10 validation
  scripts exit 0, run in the same session as this document.

## Revision Completeness Checklist

- [x] Every reviewer/DA comment from the Phase-2 decision has a row
- [x] Every RESOLVED item specifies the exact file/location changed
- [x] Every DELIBERATE_LIMITATION item states why it's a boundary, not an oversight
- [x] Every UNRESOLVED item states what's blocking it (approval, hardware, or user decision) — none are silently dropped
- [x] No scope creep: no new experiments run, no API changed, without it being logged here first
- [x] Full suite + all validation scripts green in one session after the changes
- [ ] Round 2 requires: (a) your decision on the float64/inf gap (row 5), (b) your decision on the paper-vs-library fork (Editorial Note), (c) your approval on which UNRESOLVED items to actually take on, since several are multi-day validation cycles in their own right
