# ECERC Limitation 2, Experiment 7: a zero-shot LLM fused with ECERC

Run 2026-10-18 21:49 | cuda [('Tesla T4', 14.6), ('Tesla T4', 14.6)] | torch 2.11.0+cu128 | transformers 5.16.1 | LLM unsloth/Qwen2.5-32B-Instruct-bnb-4bit | data sha256 a0349614b2b58e8b...

## Dev (selection only)

- ECERC alone, dev macro-F1: C1 54.42, C2 53.54, C3 54.16
- LLM alone, dev macro-F1: raw 55.03, calibrated 54.32 (per class, calibrated: neu 72.0, sur 65.5, fea 44.2, sad 52.2, joy 54.1, dis 29.0, ang 63.2)
- w* = 0.2; dev macro-F1 59.98 vs 54.04 for ECERC alone
- cross-fitted dev gain **+5.63** (gate +0.50: passed); projected test SE 1.36
- secondary, without calibration: cross-fitted dev gain +5.79
- Fear/Disgust/Sadness on dev, mean per checkpoint: fea ECERC wrong but LLM right 9.3 / reverse 2.0 (n=40); dis ECERC wrong but LLM right 4.3 / reverse 2.0 (n=22); sad ECERC wrong but LLM right 19.0 / reverse 12.0 (n=111)

## Verdict: TRADE-OFF

| condition | met |
|---|---|
| delta_ge_1 | yes |
| ci_lower_gt_0 | yes |
| fear_improves | yes |
| disgust_improves | yes |
| consistent_across_pairs | yes |
| accuracy_guardrail | yes |
| weighted_f1_guardrail | yes |
| class_guardrail | no |

## Test (mean over the three released checkpoints)

| arm | acc | weighted F1 | macro-F1 | F1 neu | F1 sur | F1 fea | F1 sad | F1 joy | F1 dis | F1 ang |
|---|---|---|---|---|---|---|---|---|---|---|
| ECERC alone (w=0) | 67.32 | 66.46 | 50.87 | 79.79 | 58.97 | 26.12 | 40.95 | 64.95 | 31.43 | 53.89 |
| fused (w=0.2) | 68.24 | 67.26 | 54.25 | 79.58 | 62.20 | 33.77 | 46.69 | 61.67 | 40.22 | 55.61 |
| LLM alone, calibrated (w=1) | 63.10 | 63.14 | 51.05 | 74.77 | 58.54 | 31.88 | 45.38 | 55.57 | 37.58 | 53.61 |

- Delta macro-F1 (fused - ECERC alone): **+3.38 [+0.64, +5.73]** (procedure-level two-sample dialogue bootstrap (w re-selected per replicate))
- third-look adjustment: 98.33% interval [+0.00, +6.28], lower bound above 0: no
- positive for 3/3 checkpoints: +2.83, +3.65, +3.65
- Delta accuracy +0.92 [-0.96, +2.50]; Delta weighted F1 +0.80 [-1.00, +2.40]

| class | Delta F1 [95% CI] | Delta precision | Delta recall | support |
|---|---|---|---|---|
| neu | -0.21 [-1.89, +1.27] | -2.46 | +2.55 | 1256 |
| sur | +3.23 [-0.29, +6.54] | +5.05 | +1.42 | 281 |
| fea | +7.64 [-3.10, +18.14] | -0.78 | +13.33 | 50 |
| sad | +5.74 [+0.10, +11.46] | +11.69 | +2.88 | 208 |
| joy | -3.28 [-7.49, +1.07] | -2.92 | -3.65 | 402 |
| dis | +8.79 [+0.18, +16.22] | +1.98 | +12.25 | 68 |
| ang | +1.72 [-2.71, +6.45] | +11.10 | -5.31 | 345 |

- secondary, fusion without calibration (w = 0.2): +3.27 [+0.77, +5.75] (test-only CI)
- LLM alone, raw: macro-F1 50.91, weighted F1 62.93

Checks: 18 passed, 0 warnings
