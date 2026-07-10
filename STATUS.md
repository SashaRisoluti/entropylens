# STATUS — Registro di stato della checklist (05_ITERATION_LOOP.md)

Aggiornato ad ogni ciclo. Stati: TODO / IN_PROGRESS / FAIL / BLOCKED / PASS.
Un item diventa PASS solo dopo che il suo test passa in un'esecuzione
completa della suite (`pytest tests/ -q`), non in isolamento.

| Item | Stato | Iterazione | Note |
|------|-------|------------|------|
| A1   | TODO  | -          | formula validata (MATH_DECISIONS.md D6, validation/exp5); implementazione+test pendenti |
| A2   | TODO  | -          | formula validata (MATH_DECISIONS.md D6, validation/exp5); implementazione+test pendenti |
| A3   | TODO  | -          | soglia <10% approvata dall'utente; formula validata (D1, exp1: 0.182%); implementazione+test pendenti |
| A4   | TODO  | -          | formula validata (D2, exp2: dev ≤0.34% con Miller-Madow); implementazione+test pendenti |
| A5   | TODO  | -          | tabella validata (D3, exp3; `shift` rimosso); implementazione+test pendenti |
| A6   | TODO  | -          | normalizzazione validata (D2); report normalizzato-di-default pendente |
| A7   | TODO  | -          | politiche casi limite decise (D5, inclusa riga N=0 da revisione avversaria); implementazione+test pendenti |
| B1   | TODO  | -          | - |
| B2   | TODO  | -          | - |
| B3   | TODO  | -          | - |
| B4   | TODO  | -          | - |
| C1   | TODO  | -          | formula validata (D1, quantile-winsorized, exp1); implementazione pendente |
| C2   | TODO  | -          | tabella + errore-esplicito per non mappati (D3); implementazione pendente |
| C3   | TODO  | -          | decisione utente: conv rimandato con warning a runtime (D5) |
| C4   | TODO  | -          | decisione utente: IN scope. D4 riaperta da verifica avversaria e ridecisa: bootstrap full-pipeline a intervallo normale (exp4b, copertura 90.5–100%); closed-form rigettata (min 62.5%) |
| D1   | TODO  | -          | - |
| D2   | TODO  | -          | - |
| D3   | TODO  | -          | - |
