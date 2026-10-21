# ECERC Limitation 2, Experiment 8: distilling the zero-shot LLM into ECERC

Run 2026-10-20 19:33 | cuda (Tesla T4) | torch 2.11.0+cu128 | data sha256 a0349614b2b58e8b... | teacher file sha256 8c810de775b0e5a6...

## Teachers (train and dev only)

| teacher | split | macro-F1 | F1 neu | F1 sur | F1 fea | F1 sad | F1 joy | F1 dis | F1 ang |
|---|---|---|---|---|---|---|---|---|---|
| T1, calibrated LLM | train | 51.39 | 74.6 | 61.2 | 34.7 | 49.2 | 57.0 | 33.0 | 50.1 |
| T1, calibrated LLM | dev | 54.32 | 72.0 | 65.5 | 44.2 | 52.2 | 54.1 | 29.0 | 63.2 |
| ECERC cross-fitted | train | 58.50 | 85.1 | 72.6 | 30.1 | 52.2 | 70.4 | 40.1 | 59.0 |
| T2 | train | 58.00 | 81.7 | 69.4 | 38.8 | 52.5 | 64.7 | 42.0 | 56.8 |
| T2 | dev | 60.25 | 77.7 | 68.7 | 46.4 | 53.1 | 61.6 | 45.0 | 69.3 |

## Stage 1: dev screen (mean best-epoch dev macro-F1 over seeds; selection only, not a result)

| candidate | dev macro-F1 | vs D0 |
|---|---|---|
| D0_original | 54.13 | +0.00 |
| D1_T1_a0.3 | 52.80 | -1.33 |
| D2_T1_a0.7 | 50.74 | -3.39 |
| D3_T2_a0.3 | 53.62 | -0.51 |
| D4_T2_a0.7 | 52.04 | -2.09 |
| D5_T2_a0.7_T2 | 51.92 | -2.20 |
| L1_uniform_a0.3 | 52.14 | -1.99 |
| L2_uniform_a0.7 | 50.25 | -3.88 |

Winner (D1-D5): **D3_T2_a0.3** (-0.51 in sample; +1.48 vs L1_uniform_a0.3)
Cross-fitted dev gain **-0.42** (gate +1.00: not passed, stage 2 skipped and the test not evaluated); projected test SE 0.32, MDE80 at the adjusted level 1.08


Checks: 18 passed, 0 warnings
