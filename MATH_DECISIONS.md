# MATH_DECISIONS — formule validate e decisioni chiuse

Prodotto dagli esperimenti in `validation/` (exp1–exp5 + exp4b, tutti PASS,
eseguiti 2026-07-10 su torch 2.8.0+cpu, Python 3.13). Ogni decisione qui
sotto cita l'evidenza numerica che l'ha determinata. Questo file è la
specifica delle formule per `entropylens/core.py`; della bozza
architetturale originale resta valida solo la struttura dei moduli.

**Storia delle revisioni.** Prima stesura sottoposta a verifica avversaria
multi-agente (2026-07-10). Esiti incorporati: D4 **riaperta e ridecisa**
(la closed-form falliva il proprio gate sulla pipeline reale → exp4b,
bootstrap full-pipeline); D6/D1 formula del massimo winsorizzato corretta
per quantili asimmetrici (regressa in exp1); D5 aggiunta riga N=0; D1
aggiunti scoping del tetto H* e caveat code pesanti.
2026-07-13: aggiunte D7–D9 (backend TF/JAX) sulla base di exp6–exp8, prima dell'implementazione (stessa disciplina validation-first di D1–D6).
2026-08-03: aggiunta D10 (denominatore della per-esempio) da revisione paritaria simulata: difetto di implementazione, non di formula — il clamp `1e-38` collassava a H≈0.37 una riga uniforme con `valid=True`. Gate in exp9, regressione congelata.
2026-08-03: aggiunta D11 (ordine di validazione dell'input), chiude il punto aperto lasciato dalla nota D5 del 2026-07-13: decisione utente, opzione "cast prima, controllo dopo". Gate in exp10.
2026-11-02: aggiunta D12 (overflow negli intermedi su input float32 finito) da code review del branch D10/D11: decisione utente, opzione "float64 negli intermedi". Gate in exp11; candidato numpy di exp6 risincronizzato (era fermo a prima di D10).

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

**Sweep sulla magnitudine dell'outlier (da revisione paritaria simulata —
finding DA "cherry-picking": 1000.0 è il caso massimamente favorevole al
quantile-winsorized e massimamente sfavorevole al min-max).** Stesso
protocollo, outlier a 5/20/100/1000 deviazioni standard:

| outlier | min-max | quantile-winsorized | range fisso | k-NN |
|---|---|---|---|---|
| 5σ | 7.396% | 0.182% | 0.176% | 0.496% |
| 20σ | 41.116% | 0.182% | 0.176% | 0.644% |
| 100σ | 76.970% | 0.182% | 0.176% | 0.767% |
| 1000σ | 99.735% | 0.182% | 0.176% | 0.932% |

Il min-max degrada **continuamente** con la magnitudine (7% a 20% a 77% a
99.7%): 1000.0 non è un caso patologico isolato, è il limite di una curva
che già a 5σ (un outlier realistico, non estremo) supera il gate del 10%.
Quantile-winsorized e range fisso sono **piatti** su tre ordini di
grandezza — la robustezza non dipende dalla scala dell'outlier, coerente
con il meccanismo (il clamp ignora il valore oltre il quantile, qualunque
esso sia). Il k-NN cresce lievemente con la magnitudine ma resta sotto
gate. Sweep completo in `validation/exp1_binning_outliers.py` (informativo,
non gated: il gate A3 resta fissato all'anchor documentato per restare
confrontabile nel tempo).

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
(vincolo di progetto sulle scelte senza fondamento teorico).

**Criterio di affidabilità (Difetto 5 / Q2).**

```
reliable  ⟺  (B - 1) / (2N)  ≤  0.05 · log(min(N, B))
```

cioè: il campione è affidabile quando il modulo *worst-case* del bias del
primo ordine (che si ha con K=B) non supera il 5% della scala di
normalizzazione. Per B=32: N ≥ 90. Il criterio usa B−1 (non K̂−1) per essere
valutabile a priori, indipendente dai dati. Evidenza exp2 (su N(0,1)): a
N=32 e N=50 il flag scatta e la deviazione senza correzione è effettivamente
13.8% / 8.6%; da N=128 in su il flag è verde e la deviazione corretta è
≤0.34%.

**Portata del criterio (da verifica avversaria, celle confermate a 400
seed): condizione NECESSARIA, NON sufficiente.** Il criterio modella solo
il bias da conteggio dei bin; due meccanismi reali gli sono invisibili e
producono `reliable=True` con errore oltre il 5%:

| distribuzione | B | N | flag | dev. media H_norm |
|---|---|---|---|---|
| lognormale σ=2 | 32 | 96–128 | verde | **+10.3 … +11.8%** |
| lognormale σ=2 | 32 | 200 | verde | **+7.96%** |
| zero-inflated 90% (ReLU "morta" + Exp(1)) | 32 | 128 | verde | **−7.43%** |
| zero-inflated 85% (half-normal) | 32 | 128 | verde | −6.77% |
| lognormale σ=2 | 8 | 96 / 200 / 512 | verde | **+30.2% / +16.8% / +5.1%** |

Meccanismi: (a) sparsità estrema → nei bin occupati Np_k ≪ 1 e il bias reale
supera di molto la correzione (K̂−1)/(2N) → sotto-stima residua; (b) code
pesanti → il quantile campionario Q(0.99) sotto-stima sistematicamente
quello di popolazione a N~100 → range compresso → sovra-stima. Piani
empirici sicuri dai probe (400 seed): a B=32 tutti i casi avversari provati
scendono sotto il 5% da **N ≥ 256**; a **B=8 il criterio è mis-calibrato**
per code pesanti fino a N≈512 (la scala log 8 è piccola rispetto al rumore
dei bordi). Lato conservativo (flag rosso con stima corretta entro il 5%,
es. B=128 a N=128–261): sistematico ma nella direzione sicura — accettato.

Semantica onesta del flag, riflessa nelle docstring: `False` = "certificato
inaffidabile"; `True` = "**nessun segnale d'allarme a priori** dal modello
di bias sui conteggi", NON una certificazione per distribuzioni arbitrarie
(nessun criterio indipendente dai dati può esserlo: la deviazione dipende
da coda/atomi ignoti a priori). Mitigazioni documentate per l'utente:
N ≥ 256, `value_range` fisso, o il CI bootstrap (D4) come verifica
data-driven. Il caso zero-inflated 90% a N=128 è congelato come test di
regressione della limitazione (`tests/test_reliability_limitations.py`).
*Punto aperto segnalato all'utente (non deciso in autonomia, regola 05):
alzare la soglia di default è un trade-off senza risposta univoca.*

**Cecità del clamp quantificata (da verifica avversaria):** vicino al
soffitto, due campioni realizzabili a N=96/B=32 con entropie plug-in
distanti il 4.15% producono entrambi H_norm = 1.0000 esatto; l'ampiezza
della zona cieca è ≈ (B−1)/(2N)/log B, cioè esattamente il 5% alla soglia
minima di affidabilità N=90. Resta una comodità pratica dichiarata; chi
deve distinguere valori al soffitto usi `entropy` (nats) non normalizzata.

Ogni risultato batch-wise porta con sé `reliable` e una `reason` quando
False (`"N < N_min(B)"`, `"degenerate"`, `"N = 1"`). Distingue "entropia
bassa perché il comportamento è concentrato" da "stima non affidabile"
(Difetto 5).

---

## D3 — Strategie per valori negativi (Q3 / Difetto 3 / A5 / C2)

**Anchor riprodotto (exp3):** sul vettore documentato
`[5, -5, 0.01×6]`: shift→1.9064, abs→0.7362, relu→0.0856 — identici a
all'analisi documentata del difetto, a 4 decimali: l'implementazione validata è la
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

**Dominio di validità dichiarato (raffinato da seconda verifica avversaria
del metodo scelto, 8 distribuzioni × N ∈ {100,128,200,1000,5000}, fino a
2000 replicazioni sulle celle critiche):**

- La copertura va letta come "**≥ 85%, conservativa**", non 95% esatto:
  misurata 86–100% sulla griglia validata, con pattern di miss unilaterali
  (dati sparsi: miss verso l'alto, bias/sd fino a −0.48; lognormale: verso
  il basso, +0.29). Su uniforme sovra-copre (sd_boot/sd_reale ≈ 1.7).
- Piano di validità **N ≥ 200** — il flag `reliable=True` a N ≥ 90 NON si
  trasferisce al CI: su dati sparsi/asimmetrici a N=100–128 la copertura
  scende a 80.8–83.3%.
- **Sparsità estrema (>90% della massa in un bin): piano N ≈ 400** — a
  95% di zeri e N=200 la copertura è 83.3% ± 0.6 (viola il gate), recupera
  da N≈350–400. Rilevabile a runtime dall'istogramma stesso:
  `confidence_interval` emette un warning quando p_max > 0.9 e N < 400.
- L'intervallo è **troncato a [0, log(min(N,B))]**: su dati quasi uniformi
  il limite superiore grezzo superava il massimo teorico nel ~100% delle
  run; il troncamento a un bound che contiene l'estimando non riduce mai la
  copertura (mantiene hi ≥ H_MM per l'overshoot Miller-Madow al soffitto).
- I replicati bootstrap possono essere degeneri su dati sparsi reali
  (0.7% a 95% di zeri, N=200): la libreria usa la convenzione H=0 (stessa
  di D5), non un errore.
- t di Student (df=3), lognormale σ=1.5, bimodale: PASS (87.5–96%); la
  closed-form sulle stesse celle collassa a 37–58% — conferma il rigetto.

**Punto cieco documentato (invariato da exp4):** su un campione degenere
tutti i replicati sono identici → sd_boot = 0 → CI di ampiezza nulla,
"certezza" spuria. Il flag di affidabilità NON è sostituibile
dall'incertezza: sono complementari.

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

*Nota (2026-07-13, emersa scrivendo il mirror numpy, vedi D7): `_validate_input` del core torch controlla `isfinite` sul dtype originale e poi esegue il downcast a float32 — un float64 finito ma > ~3.4e38 supera il check e diventa silenziosamente inf nel calcolo. Gap aperto sul core torch (nessun test lo esercita); il mirror numpy rifiuta questo input. Da decidere se allineare il core torch (non deciso in autonomia, regola 05).*
**→ CHIUSO da D11 (2026-08-03):** decisione utente, core torch allineato al mirror numpy (cast prima, controllo dopo). Gate `validation/exp10_dtype_validation.py`, regressione in `test_core_histogram.py` e `test_core_np_equivalence.py`. Il difetto era peggiore di quanto annotato qui: non produceva solo `inf` interno ma un risultato osservabile `H=0.0` con `reliable=True`.

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

## D7 — Core numpy di riferimento (E1 / E2): mirror numpy del core torch, equivalenza gated

**Formula adottata.** Nessuna nuova formula: D1–D6 invariate. `entropylens/core_np.py` replica le stesse operazioni in numpy float32; le uniche operazioni non bitwise-garantite sono i quantili (`np.quantile` vs `torch.quantile`, interpolazione lineare in entrambi) e l'ordine di somma nelle riduzioni.

**Evidenza (exp6, torch 2.8.0+cpu / numpy 2.3.3 / Python 3.13).**

| Gate | Esito |
|------|-------|
| G1 counts bitwise + maschera degenerate identica (6 distribuzioni × N ∈ {1,8,128,1000} × B ∈ {8,32} × 5 seed) | PASS |
| G1b ramo value_range (range fisso): counts bitwise e \|ΔH\| ≤ 1e-5 | PASS |
| G2 max \|ΔH\| | 4.77e-07 (gate 1e-5 nats) |
| G2 max \|Δnormalized\| | 1.79e-07 (gate 5e-6) |
| G3 reliable/reason identici | PASS |
| G4 per-example max \|ΔH\| (relu e abs), posizioni NaN identiche | 7.15e-07 (gate 1e-6) |
| G5 CI: scarto relativo medio semi-ampiezza | 0.060 (gate 10%); copertura pooled per-colonna 0.938, min colonna 0.90 su gauss N=400 (gate ≥0.85) |

**Conseguenza API (documentata, non silenziosa).** Il CI numpy usa `rng: np.random.Generator` al posto di `generator: torch.Generator`: i due CI NON sono riproducibili l'uno dall'altro (RNG diversi), l'equivalenza è statistica, non bitwise. `core_np` non importa torch/tensorflow/jax (test AST E2, analogo di B4). **Divergenza documentata sulla validazione dell'input:** `core_np` valida la finitezza DOPO il cast a float32, quindi un input float64 che trabocca a inf in float32 viene rifiutato (ValueError), mentre il core torch controlla `isfinite` sul dtype originale e poi calcola silenziosamente su inf dopo il downcast — il mirror numpy è intenzionalmente più severo.

---

## D8 — Semantica di cattura per backend (E3 / E5): TF = modello ausiliario inference-mode; JAX = capture_intermediates

**Scelta TF: modello ausiliario, non hook e non wrapper.** Keras non ha forward hook. `keras.Model(model.inputs[0], {name: layer.output})` condivide gli oggetti-layer con il modello utente (exp7 G4: identità su d1/d2/d3) e NON tocca il modello utente. Costo documentato: un forward aggiuntivo per osservazione, in `training=False` — con BatchNorm/Dropout le attivazioni osservate differiscono dal passo di training (exp7 G5 — anchor negativa: l'esito atteso del gate è la DIFFERENZA, non un difetto). I modelli subclassed (senza `model.inputs`) non sono supportati in v1: errore esplicito, nessun fallback silenzioso.

**Scelta JAX/Flax: capture_intermediates nello stesso forward.** Gli intermediates escono dallo stesso passo (nessun secondo forward), in una collection mutable fuori dal grad path (exp8 G3). Le attivazioni catturate dei `Dense` sono PRE-attivazione (le attivazioni flax sono funzioni, non moduli — exp8 G4): NESSUNA tabella di strategie è possibile e la per-example richiede `strategy_map` esplicita per path (coerente con D3: nessun default silenzioso). Senza strategia: solo batch-wise, con warning una-tantum per path alla prima osservazione (stessa convenzione di hooks/C2: il degrado automatico non è mai silenzioso; il warning indica che la per-example richiede una voce in strategy_map). Nota strutturale (exp8 + probing): il marcatore `__call__` ricorre a OGNI livello di annidamento dei moduli, non solo alla radice — il walker dei path deve filtrarlo ricorsivamente; le foglie sono tuple-1 di array (unwrap `[0]`).

**Versioni validate (non estrapolare al floor dichiarato dai pacchetti).** Gate misurati su TF 2.21.0 / Keras 3.14 e JAX 0.10.0 / Flax 0.12.7 (CPU). La coppia flax 0.8.0 + jax recente è ROTTA all'import (`from jax.experimental import maps` rimosso): i floor negli extra di pyproject devono citare coppie verificate, non minimi teorici.

---

## D9 — Portata delle garanzie di invarianza per backend (E3 / E5)

| Backend | Garanzia | Evidenza |
|---------|----------|----------|
| PyTorch | B1/B2/B3 bitwise via hook (invariata) | test esistenti |
| TensorFlow | modello utente MAI toccato ⇒ output e gradienti invarianti per costruzione; congelato bitwise (baseline pre-costruzione dell'aux, observe tra 3 step SGD) con `enable_op_determinism()` | exp7 G2/G3 PASS (TF 2.21.0) |
| JAX/Flax | output e gradienti bitwise con/senza cattura, senza jit E sotto `jax.jit` (max \|d\| = 0.00e+00) | exp8 G1–G3 PASS (JAX 0.10.0 / Flax 0.12.7) |

**Portata del claim jit (scoping, onestà).** Il gate G2 confronta due closure jit-compilate INDIPENDENTEMENTE (il pattern d'uso reale: forward utente vs forward strumentato), su CPU, stabile su 5 rerun consecutivi: è evidenza empirica su queste versioni, NON una garanzia del linguaggio JAX. Un bump di jaxlib/XLA può cambiare fusione/ordine delle riduzioni: rieseguire exp8 ad ogni aggiornamento dei pin jax/flax.

**Caveat comune (documentato).** L'osservazione TF è inference-mode (D8): non è una debolezza dell'invarianza (il passo utente non è toccato) ma una semantica di misura diversa dal passo di training con layer stocastici o con statistiche di batch.

---

## D10 — Denominatore della per-esempio (A2 / difetto M1 da peer review): divisione per `s`, senza clamp

**Difetto.** L'implementazione normalizzava ogni riga con
`a / s.clamp(min=1e-38)` (numpy: `np.maximum(s, 1e-38)`). Quando la somma
post-strategia della riga scende sotto 1e-38 la riga viene divisa per 1e-38
invece che per la propria somma: `p` non somma più a 1 e H crolla verso 0
mentre `valid` resta `True`.

**Formula adottata.**

```
s = Σ_i a_i         (a_i ≥ 0 per costruzione, post-strategia)
valid ⟺ s > 0
p_i = a_i / s       [nessun clamp sul denominatore]
```

`s` è somma di termini non negativi, quindi `s ≥ max(a)`: il test `s > 0` è
esatto e non esiste una riga con elementi non nulli e somma nulla. Il clamp
era quindi una difesa contro un caso impossibile che ne creava uno reale.

**Evidenza (exp9, gate G1 dichiarato prima dello sweep).** Riga uniforme di
M=8 — il caso a entropia **massima** — al variare della scala degli elementi,
errore `|H − log M|`:

| scala elemento | C0 clamp (attuale) | C1 senza clamp | C2 rescale/max | C3 float64 |
|---|---|---|---|---|
| 1e-38 | 2.33e-07 | 2.33e-07 | 2.33e-07 | 5.71e-09 |
| 1e-39 | **2.37e-01** | 2.33e-07 | 2.33e-07 | 5.71e-09 |
| 1e-40 | **1.71e+00** | 2.33e-07 | 2.33e-07 | 5.71e-09 |
| 1e-44 | **2.08e+00** | 2.33e-07 | 2.33e-07 | 5.71e-09 |

A 1e-40 il codice difettoso legge H = 0.368412 (vero log 8 = 2.079442) con
`valid=True`: una riga uniforme viene riportata come quasi-one-hot, cioè
esattamente il falso segnale di iper-specializzazione del Difetto 2.
Gate superati da C1/C2/C3: G1 (sweep sopra), G2 (riga zero → NaN + invalid,
convenzione D5 intatta), G3 (anchor D3 invariati: abs 0.7362, relu 0.0856),
G4 (one-hot → 0), G5 (batteria a scale miste, nessun NaN/Inf in righe valide).
G6 richiede che C0 fallisca G1 — verificato, il gate cattura il difetto e non
passa a vuoto.

**Scelta: C1 (nessun clamp).** I tre candidati sono indistinguibili sui gate;
C1 vince per minimalità — zero operazioni aggiunte, mentre C2 aggiunge una
riduzione per riga e C3 raddoppia la memoria del calcolo. C1 **ripristina la
formula di riferimento già validata**: `validation/common.py:147` divideva
per `s` senza clamp. Il clamp è stato introdotto in fase di implementazione
(iterazione 3) e nessun gate esistente sondava righe subnormali — motivo per
cui exp9 esiste.

**Consistenza interna (movente della decisione).** `shift` è stato rimosso in
D3 con la motivazione che «una strategia che trasforma il caso di massima
entropia in un caso indefinito non misura ciò che dichiara di misurare». Il
clamp trasformava lo stesso caso di massima entropia in H≈0.37 dichiarato
valido: stessa classe di difetto, stesso standard, stessa conclusione.

**Portata.** La soglia è la scala di `s`, non quella dei singoli elementi:
con M grande una riga può superare 1e-38 in somma pur avendo elementi molto
più piccoli. Regime raggiungibile con reti in collasso, attivazioni post-
softmax nelle code, o pesi molto piccoli. Congelato in
`tests/test_core_perexample.py::test_underflow_uniform_row` (6 scale × 2
strategie) e nel gemello numpy in `test_core_np_equivalence.py`.

---

## D11 — Ordine di validazione dell'input (A7 / E1 / difetto M5 da peer review): cast prima, controllo dopo

**Difetto.** `_validate_input` del core torch controllava `torch.isfinite`
sul dtype ORIGINALE e faceva il downcast dopo:

```
if not x.dtype.is_floating_point: x = x.float()
if not torch.isfinite(x).all(): raise ValueError(...)
return x.float()                      # <- l'overflow avviene QUI
```

Un float64 finito ma sopra il massimo di float32 (~3.4028e38) supera il
controllo e diventa `inf` nel tensore restituito. **Non solleva errore**:
`batch_wise_entropy` riporta H = 0.0 con `reliable = True` — cioè "neurone
massimamente concentrato, nessun allarme a priori". Stessa classe di
segnale falso del Difetto 2 e di D10.

Il mirror numpy castava già PRIMA di controllare (D7) e rifiutava lo stesso
input: i due core avevano contratti d'ingresso diversi. Il gap era stato
annotato in D5 (nota 2026-07-13) come punto aperto, non deciso in autonomia
(regola 05).

**Formula adottata (decisione utente, 2026-08-03 — opzione 1).**

```
x = x.float()                          # cast per primo
if not isfinite(x).all(): ValueError   # un solo controllo, dopo il cast
return x
```

Un controllo unico post-cast intercetta ENTRAMBE le cause: un NaN/Inf
originale resta NaN/Inf dopo il cast, quindi non serve un secondo passaggio.
Il messaggio d'errore nomina esplicitamente anche il caso overflow.

**Alternative considerate e scartate** (registrate perché fanno parte della
decisione):
- *Mantenere il core torch permissivo e documentare la divergenza*:
  scartata — lascerebbe la libreria a restituire `H=0, reliable=True` su
  input traboccato, esattamente il modo di rottura appena corretto in D10,
  e A7 resterebbe verde sopra un buco noto.
- *Controllare prima E dopo il cast, con messaggi distinti* ("contiene
  NaN/Inf" vs "eccede il range float32"): scartata dall'utente — l'unico
  vantaggio è diagnostico e costa una seconda riduzione `isfinite` ad ogni
  chiamata.

**Evidenza (exp10).** Gate dichiarati prima dello sweep:

| Gate | Contenuto | Esito |
|---|---|---|
| G1 | float64 traboccante → ValueError su 3 entry point × 2 core (6 casi) | PASS |
| G2 | nessuna regressione: NaN, ±Inf, vuoto, 4D, 1D ancora rifiutati | PASS |
| G3 | float64 ordinario ancora ACCETTATO e bitwise uguale al float32 equivalente (incluso 1e37, grande ma rappresentabile) | PASS |
| G4 | percorso float32 intatto: anchor D3 (0.7362 / 0.0856) e regressione D10 (riga uniforme → log 8) invariati | PASS |
| G5 | il comportamento pre-fix DEVE fallire G1 — verificato: accettava con H=0.0, reliable=True | PASS |
| G6 | i due core concordano accept/reject su tutta la batteria (7 casi) | PASS |

**Nota metodologica (bug nel gate, non nella libreria).** La prima stesura
di G3 confrontava le entropie per-esempio con `torch.equal`, che è False in
presenza di NaN per definizione: con strategia `relu`, 7 righe su 256 erano
interamente negative → somma nulla → NaN legittimo (D5). Il gate segnalava
una divergenza float64/float32 inesistente. Corretto in confronto
NaN-aware (maschere identiche + valori finiti bitwise uguali). Registrato
perché è il secondo caso in cui un gate ha prodotto un falso segnale: i
gate stessi vanno letti con la stessa diffidenza del codice.

**Conseguenza API (dichiarata, non silenziosa).** Cambio di comportamento
osservabile: input float64 con valori oltre il range float32, prima
accettati (e silenziosamente sbagliati), ora sollevano `ValueError`. È una
correzione, non una rottura: quei valori producevano numeri privi di
significato. Un'attivazione che trabocca float32 segnala un problema nella
rete osservata — farlo emergere è lo scopo dello strumento.

---

## D12 — Overflow negli intermedi su input float32 finito (A2 / A7 / E1, seguito di D10/D11): somme, quantili e indice di bin in float64

**Difetto.** D11 rifiuta al cast i valori fuori dal range float32. Valori
*dentro* il range traboccano comunque più avanti, in tre punti, in entrambi
i core:

- *per-esempio*: `s = a.sum(1)` → `inf` per una riga grande; `p = a / inf = 0`
  → H = 0 con `valid=True`. Riga uniforme di 8 elementi a 1e38: H = −0.0
  (vero log 8 = 2.0794).
- *batch-wise, indice di bin*: `hi − lo` e `xc − lo` → `inf`;
  `(xc − lo) / width` = NaN; l'indice NaN finisce nel bin 0 per ogni
  campione → H = 0 con `reliable=True`. Colonne di interi × 2^118 (finite,
  max ~3.3e38): H = 0.0 (vero 3.4617 / 3.4633 / 3.4667).
- *batch-wise, interpolazione dei quantili*: `a + w·(b − a)` con
  `b − a = inf` quando le due statistiche d'ordine interpolate stanno a
  cavallo dello zero vicino al massimo → `lo = +inf`, colonna marcata
  degenere. Flag onesto, risposta sbagliata: H = 0, `reliable=False`
  (vero 0.0844, `reliable=True`).

Stessa classe di segnale falso del Difetto 2, di D10 e di D11. Trovato in
code review del branch D10/D11 (2026-10-29): exp10 copriva solo il cast,
nessun gate sondava valori finiti vicini al massimo float32.

**Formula adottata (decisione utente, 2026-11-02 — opzione 1 di 3).**

```
per-esempio:  s = Σ_i float64(a_i);   p_i = float32(float64(a_i) / s)
batch-wise:   x64 = float64(x);  lo, hi = quantile(x64);  width = hi − lo
              idx = floor((clip(x64, lo, hi) − lo) / width · B)
```

Il resto (log, Miller-Madow, normalizzazione) resta in float32. Il float64
contiene la differenza di due float32 qualsiasi e la somma di fino a ~1e270
di essi: nessun input float32 finito può far traboccare questi intermedi.
Effetto collaterale: le somme subnormali di D10 diventano numeri normali.

*Perimetro.* L'opzione proposta nominava "somme e aritmetica dell'indice di
bin". Lo sweep di exp11 ha mostrato che trabocca anche l'interpolazione dei
quantili (G3): stessa causa, stesso rimedio, incluso nell'opzione 1 e
dichiarato qui invece che taciuto.

**Alternative considerate e scartate** (registrate perché fanno parte della
decisione):
- *Rescale per riga / per colonna sul massimo, in float32*: supera gli
  stessi gate con più codice (una riduzione in più per riga e per colonna,
  più il ramo `value_range` da gestire a parte).
- *Rifiutare in validazione ogni input i cui intermedi potrebbero
  traboccare* (es. `max|x| > finfo(float32).max / M`): seconda restrizione
  del contratto d'ingresso dopo D11, con un bound conservativo che
  rifiuterebbe input perfettamente calcolabili.

**Evidenza (exp11).** Candidati C0 (aritmetica float32 attuale) e C1
(float64). Gate dichiarati prima dello sweep:

| Gate | Contenuto | Esito |
|---|---|---|
| G1 | per-esempio, righe uniformi 1e37 / 1e38 / 3e38 / ±3e38 × 2 strategie: \|H − log M_eff\| < 1e-5, `valid` | C1 PASS; C0 H = −0.0 da 1e38 in su |
| G2 | interi × 2^k (k = 60, 100, 118): H e `reliable` bitwise uguali a k = 0 | C1 PASS; C0 H = 0.0 a k = 118 |
| G3 | quantile a cavallo dello zero: 2 × (−1.75), 126 × (+1.75), × 2^127 | C1 PASS (H = 0.084391); C0 H = 0, `reliable=False` |
| G4 | nessuna regressione su input ordinario (C1 vs C0): per-esempio max \|dH\| = 9.54e-07; batch-wise solo edge flip (5/4160 colonne, max 1 campione spostato); anchor D3 e riga subnormale D10 invariati | PASS |
| G5 | C0 DEVE fallire G1, G2, G3 | PASS (li fallisce tutti e tre) |
| G6 | libreria, entrambi i core via API pubblica: G1–G3, counts bitwise tra i core, \|dH\| ≤ 1e-6, `confidence_interval` con ampiezza non nulla sull'input G2 | PASS (pre-fix: 9 FAIL) |

**Nota sugli edge flip (G4).** In float64 un campione che cade entro
l'arrotondamento float32 di un bordo di bin può cambiare bin: 5 colonne su
4160 nella batteria, mai più di un campione per colonna. Non è una
regressione — in float32 quell'assegnazione era essa stessa un artefatto di
arrotondamento — ed è il motivo per cui G4 è un vincolo sugli spostamenti e
non un'uguaglianza bitwise con il comportamento pre-fix.

**Nota metodologica (circolarità di E1, revisione round 1 riga 18).** Prima
del fix il controllo G6 "i due core concordano" PASSAVA: entrambi i core
erano rotti nello stesso modo. Secondo caso concreto (dopo D10) in cui
l'equivalenza torch ≡ numpy non dice nulla sulla correttezza. I gate di
exp11 sono ancorati a riferimenti esatti (log M, invarianza per potenze di
due), non al core gemello.

**Risincronizzazione di exp6.** Il candidato numpy incorporato in
`validation/exp6_np_equivalence.py` (convenzione: copiato VERBATIM in
`core_np.py`) non era stato aggiornato con D10: conteneva ancora il floor
`1e-38` e restava verde solo perché la sua batteria non ha righe
subnormali. Con D12 il core torch è passato al float64 e il candidato
float32 divergeva per edge flip (G1 counts, max \|dH\| 1.74e-04).
Candidato risincronizzato a `core_np.py` (D10 + D12; identità verificata via
AST su `_quantile_hist_counts`, `_miller_madow`, `_validate_input`): exp6 di
nuovo verde con gli stessi numeri di prima (max \|dH\| 4.77e-07, copertura
pooled 0.938).

**Costo.** Copie float64 transitorie delle attivazioni negli intermedi
(~2× memoria in quei passaggi), marcate `ponytail:` in `core.py`. Nessun
rallentamento osservato sulla suite.

**Portata.** Attivazioni finite ma vicine a 3.4e38: reti in divergenza.
Raro, ma è proprio il caso in cui lo strumento deve dare un allarme e non
un "neurone concentrato". Congelato in
`tests/test_core_perexample.py::test_overflow_uniform_row` (3 righe × 2
strategie), `tests/test_core_batchwise.py::test_overflow_power_of_two_scale_invariance`
(k = 60, 100, 118) e `::test_overflow_quantile_straddle`, gemelli numpy in
`test_core_np_equivalence.py`.

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
| D7 | E1, E2 |
| D8 | E3 (parte), E5 (parte) |
| D9 | E3 (parte), E5 (parte) |
| D10 | A2 (regressione), E1 (mirror numpy allineato) |
| D11 | A7 (chiuso il gap noto), E1 (contratto d'ingresso unificato) |
| D12 | A2 (regressione), A7 (input finito ai limiti di float32), E1 (core allineati; candidato exp6 risincronizzato) |

B1–B4, D1–D3 (checklist) restano interamente a carico dell'implementazione
e della suite di test. Gli item E4, E6, E7 (tabella Keras, report/viz backend-agnostici, packaging/CI) non discendono da D7–D9 e restano anch'essi a carico dell'implementazione.
