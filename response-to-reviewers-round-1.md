# Response to Reviewers — Round 1

> **Snapshot note (added 2026-11-03).** This is the round-1 response as of
> 2026-08-03 (`9131651`). Since then:
> - M5 (the float64/inf validation gap, "no decision received yet" below) was
>   **resolved** by D11 (`4a6b29f`, user decision).
> - The E1-circularity concern is **partly addressed** by D12 /
>   `validation/exp11`, which gate on exact references instead of the twin
>   core.
> - Where this file says the numpy underflow twin covers "both cores" at 6
>   scales, the numpy twin covers 4 scales.
>
> Current state: `MATH_DECISIONS.md` D10–D12, `STATUS.md`.

**Manuscript:** entropylens (repo bundle reviewed as a methods/tools paper)
**Previous decision:** Major Revision
**Revision date:** 2026-08-03
**Full tracking table:** [`revision-round-1.md`](revision-round-1.md)

---

Thank you for the panel's review. This round addresses everything
resolvable through documentation, validation-script, and non-API code
changes. It does **not** attempt the items that require a new experiment,
a public API change, or a scope decision only the author can make — those
are itemized below with what specifically blocks them, not silently
dropped. This is deliberate: several of the "quick fixes" implied by the
original roadmap (item 6, item 9) are, on inspection, validation cycles of
the same scale as the project's own D1-D6 process, and committing to them
without sign-off would be the scope creep this project's own iteration
discipline exists to prevent.

---

## Response to R1 (Methodology)

**M1 — per-example underflow.** Fixed. `s.clamp(min=1e-38)` removed from
both cores; `s` is a sum of non-negative terms so `s>0` was already exact,
and the clamp was defending against a case that cannot occur while
creating one that could. A uniform row at element scale 1e-40 now reads
H=2.079442 (true log 8) instead of H=0.368. New decision entry D10 in
MATH_DECISIONS.md, with the exp9 evidence table and an explicit
cross-reference to D3's rationale for removing `shift` — the same failure
class, the same standard. Frozen as a regression (6 element scales × 2
strategies, both cores). *(Committed prior round, `51e63ea`; restated here
for completeness of the record.)*

**M2 — reliability flag is a per-batch scalar wearing a per-neuron
shape.** Not fixed this round. You are right that it structurally cannot
fire on D2's own sparse/heavy-tailed counterexamples, which are
per-neuron phenomena. The fix (a second, data-dependent field from
`p_max`/`K̂`, both already computed) is a `BatchEntropy` API addition. I'm
not making that call for you — it's flagged pending approval.

**M3 — `level=0.95` implies coverage the validation disclaims.** Same
answer as M2: real fix, requires changing a public parameter's contract,
awaiting your go-ahead.

**M4 — normalizer vs. attainable-maximum interaction undocumented.**
Acknowledged, tracked, folded into the M2/M3 discussion since all three
touch the same surface.

**M5 — float64→inf validation asymmetry.** This was already surfaced to
you last round with three options and a recommendation (align the torch
core to the numpy core's stricter behavior). No decision received yet —
still open, not re-litigated here.

## Response to R2 (Domain)

**D1/D4 — no literature, no positioning against the information-plane
work.** Not actioned. Both resolve to one question: is this a paper or a
library? The original editorial decision arbitrated this as CRITICAL if
paper, MAJOR if library, and I'm not deciding it for you — see the
Editorial Note in the tracking table. Until that's settled, writing a
literature section risks writing the wrong one.

**D2 — Miller-Madow unevaluated against NSB/Chao-Shen in its own weakest
regime.** Not actioned. This is a new validation cycle at the scale of
D1-D6, not a documentation fix. Flagging its size rather than attempting a
shortcut version of it.

**D3 — no baseline comparison.** Not actioned; same real-model experiment
as the Devil's Advocate's strongest counter-argument (see below) — one
experiment, if approved, closes both.

**D5 — Italian spec is a barrier.** Agreed, not actioned. ~450 lines of
formula-dense text; wanted your sign-off on the cost before starting,
given the size.

## Response to R3 (Perspective)

**P1 — GPU/CUDA unverified.** Partially addressed: the honesty gap is
closed — README's new "What's supported today" table states plainly that
GPU is not verified and names the mechanism (7 device-unsafe allocation
sites in `core.py`, listed with line numbers in the prior turn's analysis).
The code fix itself is scoped (thread `device=` through the 7 sites, add
an AST gate analogous to B4/B5 so the regression class can't recur) but
not executed — it needs your approval and, to actually confirm on
hardware, a CUDA machine I don't have here.

**P2 — cross-backend numbers not shown comparable under BatchNorm.**
Documented as an explicit open measurement gap in the new table (it was
implied by D8 but never stated as unmeasured). Experiment not run —
pending approval.

**P3 — applicability surface scattered across five documents.** Done. One
table, backend × layer × activation × training-semantics, in the README.

**P4 — no decision rule for what to do with a reading.** Not actioned —
this is a product-design question (which pathologies to name, what
thresholds to commit to), not something I should decide unilaterally and
write into the docs as if it were fact.

**P5 — dashboard reproducibility asserted, not CI-gated.** Flagged, not
actioned — running full matplotlib + training in CI is a real runtime
cost/scope trade-off, held for your input rather than silently added.

## Response to Devil's Advocate

**Strongest counter-argument (the corrected and defective estimators
agree on the only real-model evidence in the bundle).** Not actioned —
this is the single highest-value item in the entire roadmap and I'm
treating it that way by *not* attempting a rushed version of it. A model
and task where the two estimators diverge materially is a real experiment
design decision, not a documentation task.

**CRITICAL: `shift`-removal principle condemns the clamp.** Resolved by
M1's fix — same commit, cross-referenced in D10 explicitly so the
inconsistency you named is now recorded as closed, not just fixed.

**CRITICAL: unfalsifiable reliability flag.** Same issue as R1's M2 —
tracked once, not duplicated as separate work.

**Circular validation (E1 numpy≡torch, not vs. analytic references).**
Not actioned. Re-anchoring `exp6` properly is itself a 5-gate,
369-line validation rewrite at D1-D6 scale — flagging its true size
rather than patching around it.

**Cherry-picked outlier magnitude.** Resolved. `exp1` now sweeps
5σ/20σ/100σ/1000σ instead of reporting only the extreme point. Result is
informative beyond just closing the finding: min-max degrades
*continuously* with magnitude (7.4%→99.7%), while quantile-winsorized and
fixed-range stay flat within measurement noise across three orders of
magnitude — the robustness doesn't depend on how extreme the outlier is,
which is a stronger and more honest claim than the single-point anchor
supported. Recorded in MATH_DECISIONS.md D1.

**"Bitwise identical" reads as measurement precision, not
non-perturbation.** Not actioned this round — a real finding, batched with
the other small wording/API items pending a combined approval pass rather
than edited piecemeal.

## Editorial / self-identified (not from the panel)

Three mechanical accuracy gaps found while re-reading README/STATUS after
the M1 fix: `numpy` missing from the stated install command despite being
a base dependency since iteration 4; 7 of 10 validation scripts never
wired into CI despite the README's "must stay green" claim; a stale row
in STATUS.md left over from before this round's test-count update. All
three fixed — none had a design trade-off, so none were held for
approval.

Also documented, not fixed: the base-dependency version floors
(`torch>=2.0`, `numpy>=1.24`, `tensorflow>=2.16`) are theoretical
minimums, which is exactly the practice D8 rejected for jax/flax after
the flax 0.8 import break. README now states plainly that only the tested
versions are exercised. Raising the floors to match, or testing the low
end, is a real claim about compatibility — your call, not mine to assert.

---

## Summary of Changes

- Resolved: 7 items (including the prior round's D10 fix, restated for
  the record)
- Partially addressed (documentation done, code/experiment pending
  approval): 3 items
- Deliberate limitations (acknowledged, not actioned, with rationale): 3
  items
- Unresolved pending your approval or decision: 13 items
- Files changed this round: `README.md`, `STATUS.md`,
  `MATH_DECISIONS.md`, `.github/workflows/tests.yml`,
  `validation/exp1_binning_outliers.py`
- No public API changed, no new dependency, no new experiment run
- `pytest tests/ -q`: 163 passed. All 10 validation scripts exit 0 in the
  same session.

We believe this round closes every gap that didn't require your input,
and has been explicit about the ones that do rather than guessing at
them. Full point-by-point mapping in `revision-round-1.md`.
