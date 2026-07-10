# EXPERIMENT_NOTES — D1: esperimento di training con lo stimatore corretto

Script: `experiments/train_experiment.py` (deterministico, seed fissi).
Setup: MLP 20→64→32→2 (ReLU), classificazione binaria sintetica (due blob
gaussiani ±1, N=2000), Adam lr=1e-3, 30 epoche full-batch. Ad ogni epoca:

- **H_new**: entropia batch-wise normalizzata del primo layer ReLU,
  stimatore corretto della libreria (quantile-winsorized + Miller-Madow +
  log(min(N,B)), MATH_DECISIONS.md D1/D2), via `EntropyTracker`.
- **H_old**: stimatore difettoso della bozza (min-max binning, Difetto 2),
  copia locale, sulle STESSE attivazioni dello stesso run.

## Risultato (run completo nel log dello script)

| epoca | loss | H_new (norm) | H_old (nats) |
|---|---|---|---|
| 0 | 0.6830 | 0.6408 | 2.0441 |
| 10 | 0.4258 | 0.6413 | 2.0529 |
| 20 | 0.2156 | 0.6477 | 2.0762 |
| 29 | 0.0929 | 0.6543 | 2.1071 |

Trend: **entrambi gli stimatori leggono un aumento lieve e monotono**
dell'entropia batch-wise durante il training — H_new +2.1% relativo
(0.6408→0.6543), H_old +3.1% (2.0441→2.1071 nats).

## Lettura (senza over-claiming)

1. **La correzione del Difetto 2 NON ribalta la lettura empirica su questo
   run**: la tendenza "l'entropia sale durante il training" osservata con
   il codice difettoso si riproduce anche con lo stimatore corretto.
2. **Perché qui coincidono**: i due stimatori divergono solo in presenza di
   outlier o code pesanti (exp1: un outlier su 1001 → collasso del 99.7%
   del min-max, 0.18% del quantile-winsorized). Le attivazioni di questa
   rete su dati gaussiani puliti non producono outlier estremi, quindi il
   min-max non viene "attivato" nel suo modo di rottura. La correzione
   protegge i casi in cui i dati NON sono così docili — non cambia i casi
   facili.
3. **Confronto con l'osservazione della bozza (~2.0→~2.3 nats)**: la
   direzione coincide; l'ampiezza no (qui +0.06 nats). Architettura, task
   e durata diversi: nessuna conclusione sull'ampiezza è trasferibile.
4. **Questo è UN run sintetico**, non una legge generale: nessuna
   affermazione del tipo "l'entropia aumenta sempre durante il training"
   è supportata (vincolo esplicito di 04_ARCHITECTURE_DRAFT.md e
   01_PROJECT_GOAL.md sulla non-generalizzazione).

Item D1 della checklist: chiuso da questo confronto (stesso run, due
stimatori, differenza di lettura documentata).
