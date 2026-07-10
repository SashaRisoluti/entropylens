# MATH_DECISIONS — formule validate e decisioni chiuse

Prodotto dagli esperimenti in `validation/` (exp1–exp5 + exp4b, tutti PASS,
eseguiti 2026-07-10 su torch 2.8.0+cpu, Python 3.13). Ogni decisione qui
sotto cita l'evidenza numerica che l'ha determinata. Questo file è la
specifica delle formule per `entropylens/core.py`; la bozza in
`04_ARCHITECTURE_DRAFT.md` resta valida solo per la struttura dei moduli.

**Storia delle revisioni.** Prima stesura sottoposta a verifica avversaria
multi-agente (2026-07-10). Esiti incorporati: D4 **riaperta e ridecisa**
(la closed-form falliva il proprio gate sulla pipeline reale → exp4b,
bootstrap full-pipeline); D6/D1 formula del massimo winsorizzato corretta
per quantili asimmetrici (regressa in exp1); D5 aggiunta riga N=0; D1
aggiunti scoping del tetto H* e caveat code pesanti.

Tutte le entropie sono in **nats** (logaritmo naturale).

---

## D1 — Binning robusto (Q1 / Difetto 2 / A3 / C1): quantile-winsorized binning

**Formula adottata.** Dato un campione `x_1..x_N`, con `B` bin e quantili
`q_lo = 0.01`, `q_hi = 0.99` di default:

```
lo = Quantile(x, q_lo),  hi = Quantile(x, q_hi)
se hi <= lo:  H = 0, reliable = False, reason = "degenerate"
altrimenti:   bin uniformi su [lo, hi]; i valori fuori range sono
              clampati nel primo/ultimo bin (winsorizzazione)
H_plugin = - Σ_k p_k log p_k,   p_k = c_k / N
```

**Evidenza (exp1).** Con 1000 valori N(0,1) + 1 outlier (1000.0), 10 seed:

| stimatore | variazione relativa media | esito vs gate A3 (<10%) |
|---|---|---|
| min-max (bozza vecchia) | **99.735%** | riproduce il Difetto 2 |
| quantile-winsorized | **0.182%** (max 0.624%) | PASS |
| range fisso [-4,4] | 0.176% | PASS |
| k-NN Kozachenko-Leonenko | 0.932% | PASS |

**Scelta: quantile-winsorized come default.** Range fisso resta come opzione
esplicita (`range=(lo, hi)`) per chi conosce a priori il range
dell'attivazione: è marginalmente più robusto ma sposta sull'utente una
scelta che, se sbagliata, taglia fuori comportamento reale della rete.
k-NN è **rigettato come default** nonostante superi il gate, per tre ragioni
strutturali, non di robustezza: (a) stima l'entropia *differenziale*, non
limitata e potenzialmente negativa, quindi incompatibile con la
normalizzazione [0,1] richiesta da A6/Difetto 4; (b) diverge a −∞ su valori
duplicati (caso reale con attivazioni ReLU piene di zeri); (c) costo O(N²).

**Costo noto della winsorizzazione (documentato, non nascosto).** Su dati
continui a **densità uniforme** il bound `log B` non è raggiungibile
esattamente sotto clipping: il massimo raggiungibile è

```
H* = -( p_lo·log p_lo + p_hi·log p_hi + (B-2)·p_m·log p_m )
     p_m  = (q_hi - q_lo)/B
     p_lo = p_m + q_lo            [massa clampata nel primo bin]
     p_hi = p_m + (1 - q_hi)      [massa clampata nell'ultimo bin]
```

che per i default (B=32, q=0.01/0.99) vale 3.46298 = **99.92% di log(32)**
(verificato exp1 contro il valore analitico, inclusi i casi asimmetrici:
q=(0.05,0.99) → 3.43717, q=(0.0,0.95) → 3.43835). Distorsione ritenuta
trascurabile rispetto al collasso del 99.7% che risolve.

*Correzione da revisione avversaria:* la prima stesura usava
`p_e = p_m + q_lo` per ENTRAMBI i bin di bordo — corretta solo per quantili
simmetrici (giusta ai default per coincidenza); con q=(0.05,0.99) la massa
sommava a 1.04 e "H*" superava log 32. La formula corretta sopra è ora
regressa in exp1 (mass-check Σp = 1 + confronto empirico a 1e-4).

**Portata del tetto H\* (scoping, da revisione avversaria):** vale per dati
continui, non in generale. Con massa concentrata su B livelli discreti i
quantili campionari cadono sui livelli estremi, nulla viene clampato e
H = log B esattamente sotto i default (verificato exp1: 32 livelli × 100,
q=0.01/0.99 → H = log 32 a 1e-5). La frase corretta è: "il clipping riduce
il massimo raggiungibile solo quando esiste massa oltre i quantili
q_lo/q_hi".

**Caveat riproducibilità su code pesanti (da revisione avversaria,
riprodotto in proprio):** con dati a coda pesante (Cauchy) i bordi quantile
ri-stimati per batch rendono H_norm più variabile da seed a seed di un
riferimento a range fisso: std 0.036 vs 0.008 a N=2000 e 0.017 vs 0.003 a
N=10000 (4–5×, 30 seed; non svanisce al crescere di N). Nessun gate violato
(robustezza agli outlier e N-invarianza in media restano), ma è un costo di
riproducibilità documentato: per distribuzioni a coda pesante nota è
preferibile l'opzione a range fisso `value_range=(lo, hi)`.

---

## D2 — Normalizzazione cross-layer (Q2 / Difetti 1+4 / A4 / A6): Miller-Madow + log(min(N,B))

**Formula adottata.**

```
K = numero di bin non vuoti
H_MM   = H_plugin + (K - 1) / (2N)          [correzione Miller-Madow]
H_norm = min(1, H_MM / log(min(N, B)))
```

**Evidenza (exp2).** N(0,1), B=32, 20 seed, riferimento analitico
H_true = 3.29296 (quantili di popolazione, code clampate):

| N | senza correzione: dev. | con Miller-Madow: dev. |
|---|---|---|
| 128 | 3.68% | **0.34%** |
| 512 | 0.64% | **0.28%** |
| 2048 | 0.11% | **0.12%** |
| 8192 | 0.02% | **0.03%** |

Gate A4 (dev. < 5% su 4 valori di N): PASS su tutti e quattro con
Miller-Madow; senza correzione N=128 passa marginalmente e N=50/32
falliscono (8.6% / 13.8%).

**Perché il denominatore è log(min(N,B))** (fix Difetto 1): al massimo
min(N,B) bin possono essere non vuoti (dimostrazione in D6), quindi il bound
raggiungibile è log(min(N,B)), non log(B). Con N=10, B=32 l'entropia massima
osservata è log(10)=2.3026, come documentato nel Difetto 1 e riverificato.

**Clamp a 1 (scelta pragmatica dichiarata).** La correzione Miller-Madow è
una correzione *in aspettazione*; su un campione esattamente uniforme sui
bin può spingere H_MM sopra log(min(N,B)) (es. N=320 uniforme: H_plugin =
log 32, correzione +0.048). Il clamp `min(1, ·)` mantiene la scala [0,1]
richiesta da A6 al costo di una piccola distorsione al soffitto. È una
comodità pratica, non una necessità teorica: documentata come tale
(vincolo di 01_PROJECT_GOAL.md sulle scelte senza fondamento teorico).

**Criterio di affidabilità (Difetto 5 / Q2).**

```
reliable  ⟺  (B - 1) / (2N)  ≤  0.05 · log(min(N, B))
```

cioè: il campione è affidabile quando il modulo *worst-case* del bias del
primo ordine (che si ha con K=B) non supera il 5% della scala di
normalizzazione. Per B=32: N ≥ 90. Il criterio usa B−1 (non K̂−1) per essere
valutabile a priori, indipendente dai dati. Evidenza exp2: a N=32 e N=50 il
flag scatta e la deviazione senza correzione è effettivamente 13.8% / 8.6%;
da N=128 in su il flag è verde e la deviazione corretta è ≤0.34%.

Ogni risultato batch-wise porta con sé `reliable` e una `reason` quando
False (`"N < N_min(B)"`, `"degenerate"`, `"N = 1"`). Distingue "entropia
bassa perché il comportamento è concentrato" da "stima non affidabile"
(Difetto 5).

---

## D3 — Strategie per valori negativi (Q3 / Difetto 3 / A5 / C2)

**Anchor riprodotto (exp3):** sul vettore documentato
`[5, -5, 0.01×6]`: shift→1.9064, abs→0.7362, relu→0.0856 — identici a
`02_MATH_FOUNDATIONS.md` a 4 decimali: l'implementazione validata è la
stessa che ha prodotto l'evidenza del Difetto 3.

**`shift` è RIMOSSO dalla libreria.** Due ragioni: (a) nessuna motivazione
teorica (già notato nel Difetto 3); (b) patologia scoperta in exp3: su una
riga *costante* (es. attivazione uniforme, il caso a entropia massima!)
lo shift la annulla completamente → somma zero → entropia indefinita.
Una strategia che trasforma il caso di massima entropia in un caso
indefinito non misura ciò che dichiara di misurare.

**Tabella attivazione → strategia (deliverable A5/C2):**

| Modulo avvolto | Strategia | Motivazione teorica | Evidenza exp3 |
|---|---|---|---|
| ReLU, ReLU6, Softplus | `relu` (no-op) | output ≥ 0 per costruzione | massa negativa 0% |
| Sigmoid | `relu` (no-op) | output in (0,1) | massa negativa 0% |
| Softmax | nessuna: l'output È già una distribuzione; entropia calcolata direttamente | p somma a 1 per costruzione | — |
| GELU, SiLU/Swish | `relu` | range negativo LIMITATO (GELU ≥ −0.170, SiLU ≥ −0.278, verificati) vs range positivo illimitato: il segno codifica "quasi spento", non una direzione | massa negativa scale-dependent: GELU 14.7%→2.6%, SiLU 24.3%→6.9% (σ=1→3); decresce con la scala (verificato) |
| LeakyReLU | `relu` | i negativi sono segnale attenuato di fattore α (default 0.01): trattarli come "spento" perde massa ~α-proporzionale | caso coperto dalla stessa logica GELU/SiLU |
| Tanh, Softsign, Hardtanh | `abs` | segno = direzione, modulo = intensità; massa negativa ~50%: `relu` distruggerebbe metà del segnale | tanh: 50.2% massa negativa |
| Identity, Linear, moduli non mappati | **nessun default: scelta esplicita obbligatoria** (errore con elenco opzioni e semantica) | massa negativa ~50% e semantica del segno ignota; le strategie NON sono intercambiabili (anchor: 0.0856 vs 0.7362 differiscono di 8.6×) | identity: 50.2% |

**Caveat documentato per GELU/SiLU:** con input concentrati vicino a zero
(σ≈1) la strategia `relu` scarta il 15–24% della massa assoluta. Il caveat
va nella docstring della tabella, non silenziato: per input a scala
maggiore la frazione scende rapidamente (2.6%/6.9% a σ=3).

---

## D4 — Incertezza della stima (Q5 / C4, IN SCOPE per decisione utente): bootstrap full-pipeline, intervallo normale

**[RIAPERTA E RIDECISA dopo verifica avversaria — sostituisce la scelta
closed-form della prima stesura.]** La verifica ha mostrato che exp4 aveva
validato la closed-form su un problema strettamente più facile della
pipeline reale: dati categorici con bin FISSI. Nella pipeline vera i bordi
quantile sono ri-stimati dallo stesso campione, e quella varianza è
invisibile alla formula di Basharin.

**Formula adottata** (solo per entropia batch-wise; vedi sotto perché):

```
per r = 1..R (default R = 200):
    x*_r  = ricampionamento con reimmissione degli N valori GREZZI
    bordi quantile RI-stimati su x*_r → istogramma winsorizzato → H_MM,r
sd_boot = deviazione standard campionaria dei { H_MM,r }
CI 95%  = H_MM ± 1.96 · sd_boot     [intervallo normale centrato sulla
                                     stima puntuale, NON percentile]
```

**Evidenza (exp4b).** Copertura del CI al 95% nominale sulla pipeline
COMPLETA (dati continui → bordi per-campione → winsorizzazione → H_MM),
riferimento = entropia binned-clipped analitica ai quantili di popolazione,
200 replicazioni, R = 200:

| dist | N | closed-form | boot-perc | boot-basic | **boot-normal** | s_claim/s_act |
|---|---|---|---|---|---|---|
| normal | 200 | 78.0% | 91.0% | 47.5% | **94.5%** | 0.688 |
| normal | 1000 | 80.0% | 97.0% | 76.0% | **94.5%** | 0.697 |
| normal | 5000 | 88.0% | 97.5% | 90.5% | **96.0%** | 0.729 |
| uniform | 200 | 99.5% | 4.0% | 31.5% | **100.0%** | 1.311 |
| uniform | 1000 | 100.0% | 17.0% | 33.0% | **100.0%** | 1.367 |
| uniform | 5000 | 96.5% | 40.5% | 47.0% | **97.0%** | 1.083 |
| exponential | 200 | 64.5% | 98.0% | 69.0% | **91.0%** | 0.481 |
| exponential | 1000 | 62.5% | 93.0% | 79.0% | **90.5%** | 0.457 |
| exponential | 5000 | 69.0% | 95.0% | 89.5% | **95.5%** | 0.499 |

Gate (≥85% su TUTTE le celle: 3 distribuzioni × N ∈ {200, 1000, 5000}):
**solo boot-normal è eleggibile** (min 90.5%). Rigetti:

- **closed-form (Basharin), min 62.5%**: il rapporto stderr-dichiarato /
  sd-reale oscilla tra 0.46 (exponential) e 1.37 (uniform) e NON converge
  a 1 al crescere di N → il deficit è strutturale (varianza di ri-stima
  dei bordi), non small-sample. Nota: su exp4 (bin fissi) la stessa
  formula copriva 87.5–99.5% — la differenza È la varianza dei bordi.
- **boot percentile, min 4.0%**: collassa al plafond di entropia
  (uniforme), stesso meccanismo documentato in exp4.
- **boot basic, min 31.5%**.
- **inflazione calibrata dello stderr: rigettata senza test** — il fattore
  necessario dipende da distribuzione e N in entrambe le direzioni
  (0.46–1.37), quindi nessuna costante principiata esiste.

Perché boot-normal funziona dove il percentile fallisce: sd_boot stima
bene la *scala* della variabilità (inclusa quella dei bordi) anche quando
la *posizione* dei replicati è distorta verso il basso vicino al plafond;
l'intervallo normale ricentra sulla stima puntuale H_MM, il percentile no.

**Conseguenza API (documentata, non silenziosa).** Il CI richiede i valori
GREZZI (ogni replicato deve ri-stimare i bordi), quindi non è derivabile
dai soli conteggi, e costa R istogrammi completi. Perciò:
`BatchEntropy` NON trasporta uno stderr per-batch (il Basharin è rimosso:
un errore standard sotto-calibrato del 30–50% è peggio di nessun numero);
l'incertezza si ottiene on demand con `confidence_interval(activations,
...)`. Gli hook non conservano le attivazioni (memoria), quindi il CI è
un'analisi esplicita sull'attivazione che l'utente passa, non un
sottoprodotto automatico del tracking.

**Dominio di validità dichiarato:** copertura verificata per N ≥ 200
(exp4b); sotto N_min(B)=90 il flag `reliable` scatta comunque. **Punto
cieco documentato (invariato da exp4):** su un campione degenere tutti i
replicati sono identici → sd_boot = 0 → CI di ampiezza nulla, "certezza"
spuria. Il flag di affidabilità NON è sostituibile dall'incertezza: sono
complementari.

**Per l'entropia per-esempio non esiste stderr campionario**: per ogni
esempio è una funzione deterministica del vettore di attivazione, non una
stima da N campioni. L'incertezza si applica solo ad aggregati sul batch
(es. SEM della media = std/√N). L'API deve riflettere questa distinzione,
non offrire uno stderr fittizio per riga.

---

## D5 — Politiche sui casi limite (A7, C3)

| Caso | Comportamento | Motivazione |
|---|---|---|
| N = 0 (tensore vuoto) | `ValueError` immediato, per ENTRAMBE le metriche | l'entropia di un campione vuoto non è definita; nessun valore o flag la rappresenterebbe onestamente. *(Riga aggiunta da revisione avversaria: mancava dalla prima stesura; gli script in validation/ si comportavano in modo incoerente su questo caso — la libreria no.)* |
| NaN o ±Inf nell'input | `ValueError` immediato | l'entropia di dati contenenti NaN/Inf non è definita; ogni gestione silenziosa maschererebbe un bug della rete osservata |
| N = 1 (batch-wise) | H = 0, `reliable=False`, reason `"N=1"` | un campione singolo è degenere per costruzione |
| tutti i valori identici | H = 0, `reliable=False`, reason `"degenerate"` | range quantile nullo; matematicamente H=0 è corretto ma indistinguibile da vera concentrazione senza flag (Difetto 5) |
| riga tutta zero (per-esempio) | `NaN` + maschera `valid` per riga | distribuzione indefinita (0/0); NaN è onesto, 0 mentirebbe (verificato exp3/exp5) |
| tensore 4D+ (conv, C3 — decisione utente: rimandato) | funzioni core: `ValueError`; hooks/layers: il layer viene SALTATO con `warnings.warn` esplicito ("conv non validato in v1, layer ignorato") | mai emettere numeri non validati (Q4 aperto); il warning a runtime soddisfa C3; un tracker su una CNN resta utile per i layer FC |
| input 1D a batch-wise | trattato come (N, 1) | caso monodimensionale ben definito |

---

## D6 — Dimostrazioni dei bound (A1 / A2)

### A1 — batch-wise: 0 ≤ H ≤ log(min(N, B))

Siano `c_1..c_B` i conteggi (`Σc_k = N`), `K = #{k : c_k > 0}`,
`p_k = c_k/N`.

1. **H ≥ 0**: ogni termine −p_k·log p_k ≥ 0 perché 0 < p_k ≤ 1.
   **H = 0 ⟺ K = 1**: se K=1 l'unico p è 1 e il termine è 0; se K≥2
   esiste p_k ∈ (0,1) con contributo strettamente positivo.
2. **H ≤ log K** (disuguaglianza di Gibbs contro la distribuzione uniforme
   sui K bin occupati): log K − H = D_KL(p ‖ u_K) ≥ 0, con uguaglianza ⟺
   p uniforme sui K bin occupati.
3. **K ≤ min(N, B)**: K ≤ B per costruzione; K ≤ N perché ognuno degli N
   campioni cade in esattamente un bin, quindi i bin occupati sono al
   più N.

Quindi 0 ≤ H ≤ log(min(N,B)). **Raggiungibilità del massimo** (verificata
exp5, senza clipping):
- N ≤ B: ogni campione in un bin distinto → p uniforme su N bin →
  H = log N esattamente (verificato: N=8, B=32 → H = log 8 a 1e-5).
- N > B: serve K = B e N/B campioni per bin, quindi B | N (verificato:
  N=320, B=32, 10 per bin → H = log 32 a 1e-5).
- Sotto winsorizzazione di default il massimo scende al 99.92% di log B
  **per dati continui a densità uniforme** (formula corretta, scoping ed
  eccezione discreta in D1: con massa su B livelli discreti log B resta
  raggiungibile esattamente anche con clipping attivo).

### A2 — per-esempio: 0 ≤ H ≤ log(M)

Sia `a ∈ R^M` la riga post-strategia (quindi a_i ≥ 0), `S = Σa_i`.

- **S = 0**: la distribuzione non esiste (0/0) → NaN dichiarato (D5).
- **S > 0**: p_i = a_i/S è una distribuzione su M simboli; stessi argomenti
  di Gibbs: 0 ≤ H ≤ log M; H = 0 ⟺ p one-hot ⟺ un solo a_i > 0;
  H = log M ⟺ p uniforme ⟺ tutti gli a_i uguali e > 0.

Verificati exp5: one-hot → 0 (‖·‖ < 1e-7), uniforme → log 16 (1e-5),
riga zero → NaN, disuguaglianza su 64 righe casuali per entrambe le
strategie superstiti (`relu`, `abs`).

---

## Mappa decisioni → item della checklist

| Decisione | Item chiusi (a valle dell'implementazione + test) |
|---|---|
| D1 | A3, C1 |
| D2 | A4, A6 (parte), Difetto 5 |
| D3 | A5, C2 |
| D4 | C4 |
| D5 | A7, C3 |
| D6 | A1, A2 |

B1–B4, D1–D3 (checklist) restano interamente a carico dell'implementazione
e della suite di test.
