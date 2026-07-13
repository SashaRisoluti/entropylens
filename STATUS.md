# STATUS — Registro di stato della checklist di uscita del progetto

Ultima esecuzione completa: `pytest tests/ -q` → **52 passed** (run unico,
D3) + `validation/exp1..exp5 + exp4b` tutti exit 0, 2026-07-10, torch
2.8.0+cpu, Python 3.13. Ciclo: iterazione 1 = validazione formule
(exp1–exp5, MATH_DECISIONS.md); iterazione 2 = verifica avversaria
multi-agente e amendment (D4 rideciso → exp4b, D6/D1 formula corretta, D5
riga N=0, D2 ri-scoping del flag); iterazione 3 = implementazione TDD
(Task 0–11 del piano) con incorporazione dei fix della seconda verifica
avversaria (warning sparsità CI, troncamento CI, test di regressione delle
limitazioni).

| Item | Stato | Iterazione | Note |
|------|-------|------------|------|
| A1   | PASS  | 3          | dimostrazione D6 + `test_core_batchwise.py::test_a1_*` (disuguaglianza 4 regimi, attainment N<B e B\|N, degenere, N=1) |
| A2   | PASS  | 3          | dimostrazione D6 + `test_core_perexample.py::test_a2_*` (one-hot→0, uniforme→log M, riga zero→NaN+valid, disuguaglianza 64 righe) |
| A3   | PASS  | 3          | soglia <10% approvata dall'utente; `test_core_batchwise.py::test_a3_outlier_gate` misura ~0.2% (gate exp1: 0.182%) |
| A4   | PASS  | 3          | `test_core_batchwise.py::test_a4_n_invariance_of_normalized_entropy`: 4 N, dev <5% vs riferimento analitico (exp2: ≤0.34%). Portata oltre N(0,1): vedi D2 counterexamples + `test_reliability_limitations.py` |
| A5   | PASS  | 3          | tabella con motivazioni in `strategies.py` (D3); `test_strategies.py` (mappature, motivazioni non vuote, errore esplicito per non mappati, no subclass silenzioso) |
| A6   | PASS  | 3          | report normalizzato-di-default, nats solo con bound esplicito: `test_report.py` |
| A7   | PASS  | 3          | politiche D5 (inclusa N=0 → ValueError da revisione avversaria): `test_core_histogram.py` (NaN/Inf, 4D, vuoto), `test_core_batchwise.py` (N=1, degenere, 1D), `test_core_perexample.py` (riga zero, 1D rifiutato) |
| B1   | PASS  | 3          | `test_hooks.py::test_b1_output_bitwise_invariant` — torch.equal (0.0 esatto), anche dopo detach |
| B2   | PASS  | 3          | `test_layers.py::test_b2_output_bitwise_identical` + gradienti bitwise |
| B3   | PASS  | 3          | `test_hooks.py::test_b3_*` (gradienti bitwise con/senza tracker, tensori entropia detached) + no_grad in tutte le funzioni core |
| B4   | PASS  | 3          | `test_b4_core_independence.py` — check AST: core.py non importa/usa torch.nn; funzioni su tensori puri |
| C1   | PASS  | 3          | quantile-winsorized implementato (D1); conteggi bitwise-uguali al riferimento histc di exp1 (`test_core_histogram.py`) |
| C2   | PASS  | 3          | tabella A5 + `UnmappedActivationError` con guida esplicita; hooks: warning una-tantum e tracking solo batch-wise per layer non mappati |
| C3   | PASS  | 3          | decisione utente: conv rimandato. Core: ValueError su 4D; hooks/layers: layer saltato con warning esplicito (`test_hooks.py::test_c3_*`, `test_layers.py::test_conv_*`) |
| C4   | PASS  | 3          | decisione utente: IN scope. D4 rideciso dopo verifica avversaria: bootstrap full-pipeline a intervallo normale (exp4b: 90.5–100%; closed-form rigettata, min 62.5%). Seconda verifica: coperture "≥85% conservativa", piano N≥200 (N≈400 con >90% sparsità → warning runtime), CI troncato a [0, log(min(N,B))]. `test_uncertainty.py` |
| D1   | PASS  | 3          | `experiments/train_experiment.py` + `EXPERIMENT_NOTES.md`: stesso run, stimatore corretto vs difettoso — tendenza in salita confermata da entrambi (+2.1% vs +3.1%); divergono solo con outlier (exp1) |
| D2   | PASS  | 3          | ogni item A/B ha test automatizzato nominato in questa tabella; suite riproducibile in `tests/` |
| D3   | PASS  | 3          | 52 test passati in UN'unica esecuzione (`pytest tests/ -q`) + 6 script di validazione exit 0 nella stessa sessione |
| E1   | TODO  | 4          | core numpy equivalente al core torch (D7) — gate congelati in `validation/exp6_np_equivalence.py`, implementazione `core_np.py` in corso |
| E2   | TODO  | 4          | indipendenza di core_np da torch/tf/jax (D7, analogo B4) — test AST previsto |
| E3   | TODO  | 4          | tracker TensorFlow via modello ausiliario (D8/D9) — gate congelati in `validation/exp7_tf_capture.py` |
| E4   | TODO  | 4          | tabella strategie Keras (D3 esteso) — implementazione prevista |
| E5   | TODO  | 4          | cattura JAX/Flax via capture_intermediates (D8/D9) — gate congelati in `validation/exp8_jax_capture.py` |
| E6   | TODO  | 4          | report/viz backend-agnostici su record numpy — implementazione prevista |
| E7   | TODO  | 4          | packaging extras tf/jax + job CI backends — implementazione prevista |

## Limitazioni documentate (non regressioni, congelate in test)

- Flag `reliable`: condizione necessaria, NON sufficiente (D2): su dati
  sparsi (90% zeri) o a coda pesante (lognormale σ=2) può essere verde con
  deviazione 7–14% per N∈[90, ~200] a B=32; B=8 mis-calibrato per code
  pesanti. Congelato in `test_reliability_limitations.py`. Punto aperto
  per l'utente: alzare la soglia di default (trade-off).
- CI bootstrap: "≥85% conservativa", non 95% esatto; piano N≥200
  (N≈400 sotto sparsità estrema, con warning runtime).
- Code pesanti (Cauchy): H_norm 4–5× più variabile da seed a seed del
  range fisso (D1) — mitigazione: `value_range`.
- Winsorizzazione: massimo raggiungibile 99.92% di log B su densità
  uniforme continua (formula asimmetrica corretta in D1/exp1).
