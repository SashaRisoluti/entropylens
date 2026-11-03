# Fix P0 review defects

> **Snapshot note (added 2026-11-03).** This is the working plan from before
> D10; its line references and script counts are from that time.
> - The "BLOCKED" float64/inf item below was **resolved** by D11 (`4a6b29f`,
>   user decision).
> - A follow-up overflow defect was fixed in D12.
> - The GPU/device item is still open.
>
> Current state: `MATH_DECISIONS.md`, `STATUS.md`.

## Goal
Close the three P0 defects from the peer-review panel: the confirmed per-example
underflow bug, the unverified GPU/device question, and the self-declared
float64→inf validation gap. Nothing else — P1/P2/P3 items stay untouched.

## Scope note
This is a bug-fix cycle inside the `05_ITERATION_LOOP.md` loop: propose →
verify numerically → implement → regression → update STATUS. Steps 1–2 exist
because this project does not implement before a validation script passes a
declared gate.

---

## Tasks

- [x] **1. `validation/exp9_underflow.py`** — declare the gate in the docstring
  first: a uniform row at element scale 1e-30 … 1e-44 must read `log M` within
  1e-5, `valid=True`; a true zero row must read NaN, `valid=False`; the D3
  anchors (`abs`→0.7362, `relu`→0.0856) must not move by more than 5e-4.
  Sweep the current code and at least two candidate fixes (row-max rescale;
  float64 reduction). → **Verify:** script exits 0, current code FAILs the
  sweep, chosen candidate PASSes.

- [x] **2. Fix `per_example_entropy`** in [core.py:210-225](entropylens/core.py#L210-L225)
  with whichever candidate exp9 selected. Do not change the `valid = s > 0`
  semantics or the NaN convention (D5). → **Verify:** `python validation/exp9_underflow.py`
  exits 0.

- [x] **3. Mirror the fix in [core_np.py](entropylens/core_np.py)** — the D7
  equivalence gate breaks if only one core moves. → **Verify:**
  `python validation/exp6_np_equivalence.py` exits 0 (G4 per-example |ΔH| ≤ 1e-6).

- [x] **4. Regression tests** — add `test_underflow_uniform_row` to
  [tests/test_core_perexample.py](tests/test_core_perexample.py) (parametrized
  over the element scales, both strategies) and the numpy twin in
  `test_core_np_equivalence.py`. Freeze the failing case as a permanent
  regression, same convention as `test_reliability_limitations.py`.
  → **Verify:** the new test fails on `git stash` of task 2, passes after.

- [x] **5. Full regression in one run** (loop step 5, D3 discipline):
  `pytest tests/ -q` + all 9 validation scripts in the same session.
  → **Verify:** 142+N passed, 9 scripts exit 0, no previously-PASS item regressed.

- [x] **6. Record the decision** — add `D10` to
  [MATH_DECISIONS.md](MATH_DECISIONS.md) with the formula, the exp9 evidence
  table, and the internal-consistency note (this is the same failure class that
  justified removing `shift` in D3). Update the A2 row in
  [STATUS.md](STATUS.md#L22) to cite exp9. → **Verify:** D10 present, A2 row
  cites the new script, D3's `shift` rationale cross-referenced.

- [ ] **7. GPU probe** — `python -c "import torch; from entropylens import batch_wise_entropy, per_example_entropy; x=torch.randn(512,64,device='cuda'); print(batch_wise_entropy(x).normalized[:3]); print(per_example_entropy(x,'abs').entropy[:3])"`
  on a CUDA machine. Not runnable in this environment — no CUDA.
  → **Verify:** either it prints, or it raises a device-mismatch `RuntimeError`
  and the result decides task 8.

- [ ] **8. Act on the probe result** — if it raised: thread `device=`/`dtype=`
  through [core.py:75-88](entropylens/core.py#L75-L88) and
  [core.py:281](entropylens/core.py#L281), add one CUDA-gated test
  (`@pytest.mark.skipif(not torch.cuda.is_available())`). If it passed: add the
  same test anyway and state GPU support in the README. → **Verify:** the test
  exists and is green on a CUDA box; README says which it is.

## Done When
- [ ] exp9 exits 0 and its regression test is in the suite
- [ ] `pytest tests/ -q` + 9 validation scripts green in one consecutive run
- [ ] D10 written, STATUS A2 updated
- [ ] GPU support is a stated fact in the README, not an open question

---

## BLOCKED — needs your decision before it can move

**float64 → inf validation gap** (review finding M5; already self-flagged at
[MATH_DECISIONS.md:343](MATH_DECISIONS.md#L343)). `_validate_input` checks
`torch.isfinite` on the original dtype, then downcasts
([core.py:39-41](entropylens/core.py#L39-L41)), so a finite float64 above
~3.4e38 becomes `inf` silently. `core_np` already rejects it (D7).

Three options, real trade-off, no correct answer — `05_ITERATION_LOOP.md`
says the loop stops here rather than choosing for you:

| Option | Effect |
|---|---|
| Align torch to numpy: validate after the cast | Consistent across cores; **behavior change** — inputs that ran before now raise `ValueError` |
| Keep torch permissive, document the divergence | No breakage; two cores with different contracts, and A7 stays green over a known hole |
| Validate before *and* after the cast, warn on the gap | Strictest; adds a second `isfinite` pass on every call |

Recommendation: option 1. The divergence exists because the numpy mirror is
right and the torch core is wrong; a `ValueError` on an activation that
overflows float32 is the honest outcome, and any model producing one has a
bug worth surfacing. Confirm and it moves into the task list as task 2b.

## Notes
- Candidate fix already probed (row-max rescale, entropy is invariant to
  positive row scaling): uniform rows read `log 8` correctly from 1e-30 down to
  1e-44, zero rows still NaN, both D3 anchors unchanged to 4 decimals. exp9
  still has to choose between candidates on a declared gate — that probe is
  evidence the fix direction works, not the validation itself.
- Tasks 1–6 are the critical path and are sequential. Task 7 is independent and
  can run in parallel on any CUDA machine.
- Out of scope for this plan: reliability-flag redesign (M2), `level` semantics
  (M3), normalizer/`H*` interaction (M4), and everything in review P1–P3.
