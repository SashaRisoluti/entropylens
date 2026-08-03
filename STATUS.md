# STATUS — Registro di stato della checklist di uscita del progetto

Esecuzione di riferimento delle iterazioni 1–3: `pytest tests/ -q` →
**52 passed** (run unico, D3) + `validation/exp1..exp5 + exp4b` tutti
exit 0, 2026-07-10, torch 2.8.0+cpu, Python 3.13 (l'ultima esecuzione
completa è quella dell'iterazione 4, sotto). Ciclo: iterazione 1 = validazione formule
(exp1–exp5, MATH_DECISIONS.md); iterazione 2 = verifica avversaria
multi-agente e amendment (D4 rideciso → exp4b, D6/D1 formula corretta, D5
riga N=0, D2 ri-scoping del flag); iterazione 3 = implementazione TDD
(Task 0–11 del piano) con incorporazione dei fix della seconda verifica
avversaria (warning sparsità CI, troncamento CI, test di regressione delle
limitazioni).

Full run di 2026-07-13 — `pytest tests/ -q` → **142 passed**, 9 script di
validazione exit 0 (exp1–exp5, exp4b, exp6–exp8), torch 2.8.0+cpu / TF
2.21.0 / JAX 0.10.0 / Flax 0.12.7 / Python 3.13, iterazione 4 (backend
TF/JAX, D7–D9).

Full run di 2026-08-03 — `pytest tests/ -q` → **168 passed**, 11 script di
validazione exit 0 (exp1–exp10), stesse versioni, iterazione 5 (revisione
paritaria simulata: D10 denominatore della per-esempio, D11 ordine di
validazione dell'input, sweep magnitudine outlier in exp1, wiring CI di
tutti gli script di validazione). Nessun item precedentemente PASS è
regredito.

**Limitazione di processo (da revisione paritaria, non risolta):** nessun
item della checklist copre il *device placement*. `core.py` alloca vari
tensori senza `device=`, quindi il supporto GPU non è verificato né
gated — vedi la tabella "What's supported today" nel README. Serve un item
nuovo (proposta: B6, check AST analogo a B4/B5) più una verifica su
hardware CUDA, assente da questo ambiente e dalla CI.

| Item | Stato | Iterazione | Note |
|------|-------|------------|------|
| A1   | PASS  | 3          | dimostrazione D6 + `test_core_batchwise.py::test_a1_*` (disuguaglianza 4 regimi, attainment N<B e B\|N, degenere, N=1) |
| A2   | PASS  | 3, 5       | dimostrazione D6 + `test_core_perexample.py::test_a2_*` (one-hot→0, uniforme→log M, riga zero→NaN+valid, disuguaglianza 64 righe). It. 5: difetto di implementazione D10 corretto (denominatore con clamp 1e-38 → riga uniforme letta H≈0.37 con `valid=True`); gate `validation/exp9_underflow.py`, regressione `test_underflow_uniform_row` (6 scale × 2 strategie) + gemello numpy |
| A3   | PASS  | 3          | soglia <10% approvata dall'utente; `test_core_batchwise.py::test_a3_outlier_gate` misura ~0.2% (gate exp1: 0.182%) |
| A4   | PASS  | 3          | `test_core_batchwise.py::test_a4_n_invariance_of_normalized_entropy`: 4 N, dev <5% vs riferimento analitico (exp2: ≤0.34%). Portata oltre N(0,1): vedi D2 counterexamples + `test_reliability_limitations.py` |
| A5   | PASS  | 3          | tabella con motivazioni in `strategies.py` (D3); `test_strategies.py` (mappature, motivazioni non vuote, errore esplicito per non mappati, no subclass silenzioso) |
| A6   | PASS  | 3          | report normalizzato-di-default, nats solo con bound esplicito: `test_report.py` |
| A7   | PASS  | 3, 5       | politiche D5 (inclusa N=0 → ValueError da revisione avversaria): `test_core_histogram.py` (NaN/Inf, 4D, vuoto), `test_core_batchwise.py` (N=1, degenere, 1D), `test_core_perexample.py` (riga zero, 1D rifiutato). It. 5: chiuso il gap noto della nota D5 2026-07-13 (float64 > float32 max accettato → H=0.0 con `reliable=True`); D11 allinea il core torch al mirror numpy, gate `validation/exp10_dtype_validation.py` (G1-G6), regressione in `test_core_histogram.py` (3 entry point) |
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
| D3   | PASS  | 3, 5       | 52 test in UN'unica esecuzione (it. 3) + 6 script exit 0; ri-verificato it. 5: 163 test + 10 script (exp1-exp9) in un'unica sessione, vedi blocco full-run 2026-08-03 sopra |
| E1   | PASS  | 4, 5       | core numpy equivalente al core torch (D7): `test_core_np_equivalence.py` (counts bitwise, \|dH\|<=1e-5, ramo value_range incluso); sweep completo in `validation/exp6_np_equivalence.py` (max \|dH\| 4.77e-07, copertura CI pooled 0.938). It. 5: contratto d'ingresso ora unificato tra i due core (D11, exp10 G6) — prima divergevano sul float64 traboccante; `test_dtype_contract_matches_torch_policy` |
| E2   | PASS  | 4          | `test_b5_np_core_independence.py`: check AST — `core_np.py` e `record.py` non importano torch/tensorflow/jax/flax/keras (analogo B4) |
| E3   | PASS  | 4          | `test_tf_tracker.py` (12 test): modello utente mai toccato (aux model condiviso, D8); invarianza congelata bitwise in `validation/exp7_tf_capture.py` G2/G3 con op-determinism; submodel annidati e modelli subclassed esclusi esplicitamente |
| E4   | PASS  | 4          | `test_strategies_tf.py` (7 test): tabella Keras per classe + identita' di attivazione; Dense lineare e LeakyReLU con slope di default (0.3 > 0.1) -> errore esplicito (D3); hard_tanh -> abs |
| E5   | PASS  | 4          | `test_jax_tracker.py` (10 test): capture_intermediates stesso forward, bitwise anche sotto jit (`validation/exp8_jax_capture.py` G1-G3, max \|d\| = 0); strategy_map esplicita con warning una-tantum per path (D8 emendata); FrozenDict e multi-sow gestiti |
| E6   | PASS  | 4          | `test_report_np_records.py` (5 test): report/viz su record numpy senza torch; aggregazione semanticamente invariante ma NON bitwise (ordine di somma numpy, <=1 ulp float32; reliable_fraction accumulata in float64; ddof=1 pinnato da test) |
| E7   | PASS  | 4          | pyproject 0.2.0: numpy dipendenza base, extras `tf`/`jax` con floor VALIDATI (jax>=0.10, flax>=0.12 — coppia flax 0.8 rotta all'import); job CI `backends` (tf-cpu, jax cpu, exp6-8 exit 0) |

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
