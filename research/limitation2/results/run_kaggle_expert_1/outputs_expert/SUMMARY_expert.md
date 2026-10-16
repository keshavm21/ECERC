# ECERC Limitation 2, Experiment 4: context-aware text expert fused with ECERC

Run 2026-10-11 22:38 | cuda (Tesla T4) | torch 2.11.0+cu128 | transformers 5.16.1 | data sha256 a0349614b2b58e8b...

## Dev (selection only)

- ECERC alone, dev macro-F1: C1 54.42, C2 53.54, C3 54.16
- expert alone, dev macro-F1: seed 1 54.36 (weighted F1 65.66), seed 2 55.97 (weighted F1 66.85), seed 3 54.95 (weighted F1 66.19)
- w* = 0.5; dev macro-F1 56.33 vs 54.04 for ECERC alone
- cross-fitted dev gain **+0.97** (gate +0.50: passed); projected test SE 0.91
- Fear/Disgust/Sadness on dev, mean per pair: fea ECERC wrong but expert right 9.4 / reverse 3.8 (n=40); dis ECERC wrong but expert right 3.0 / reverse 2.3 (n=22); sad ECERC wrong but expert right 16.3 / reverse 9.7 (n=111)

## Verdict: INCONCLUSIVE

| condition | met |
|---|---|
| delta_ge_1 | no |
| ci_lower_gt_0 | no |
| fear_improves | yes |
| disgust_improves | no |
| consistent_across_pairs | no |
| accuracy_guardrail | yes |
| weighted_f1_guardrail | yes |
| class_guardrail | no |

## Test (mean over the 9 checkpoint x seed pairs)

| arm | acc | weighted F1 | macro-F1 | F1 neu | F1 sur | F1 fea | F1 sad | F1 joy | F1 dis | F1 ang |
|---|---|---|---|---|---|---|---|---|---|---|
| ECERC alone (w=0) | 67.32 | 66.46 | 50.87 | 79.79 | 58.97 | 26.12 | 40.95 | 64.95 | 31.43 | 53.89 |
| fused (w=0.5) | 67.12 | 66.45 | 51.14 | 80.04 | 58.75 | 29.52 | 43.18 | 62.34 | 29.46 | 54.71 |
| expert alone (w=1) | 65.57 | 65.34 | 50.50 | 78.89 | 57.89 | 27.87 | 43.29 | 59.49 | 31.89 | 54.18 |

- Delta macro-F1 (fused - ECERC alone): **+0.27 [-1.92, +2.29]** (procedure-level two-sample dialogue bootstrap (w re-selected per replicate))
- positive in 5/9 pairs: -0.28, -0.53, -0.29, +0.05, +0.68, -0.29, +0.70, +1.18, +1.21
- Delta accuracy -0.20 [-1.98, +1.04]; Delta weighted F1 -0.01 [-1.50, +1.11]

| class | Delta F1 [95% CI] | Delta precision | Delta recall | support |
|---|---|---|---|---|
| neu | +0.25 [-1.15, +1.07] | +1.24 | -0.84 | 1256 |
| sur | -0.22 [-2.94, +1.92] | -2.01 | +1.86 | 281 |
| fea | +3.40 [-5.15, +11.38] | -0.44 | +5.56 | 50 |
| sad | +2.23 [-1.47, +5.82] | +0.91 | +2.72 | 208 |
| joy | -2.61 [-6.03, +1.11] | -3.50 | -1.27 | 402 |
| dis | -1.97 [-9.22, +6.64] | +0.61 | -2.61 | 68 |
| ang | +0.82 [-2.12, +3.92] | +2.06 | -0.39 | 345 |

- secondary, w and tau chosen together on dev ({'w': 0.2, 'tau': 0.1}): +1.17 [-0.01, +2.36]
- descriptive ensembles, test macro-F1: ECERC x3 ensemble 50.31, expert ensemble 51.05, fused ensembles at w* 51.42

Checks: 16 passed, 0 warnings
