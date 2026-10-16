# ECERC Limitation 2: post-hoc logit adjustment (MELD, released checkpoints)

Run 2026-10-11 10:20 | device cuda (Tesla T4) | torch 2.11.0+cu130 | numpy 2.1.3 | sklearn 1.6.1
Data sha256 a0349614b2b58e8b... (486 MB) | checkpoints: C1 (ererc0_meld_valf1), C2 (ererc2613_meld_valf1), C3 (ererc659_meld_valloss) | included: ['C1', 'C2', 'C3']

Checks: 23 passed, 0 warnings

## Frozen on dev before the test set was touched
- tau* = 0.1 (dev mean macro-F1 54.45 vs 54.04 at tau = 0); Neutral-only delta* = 0.45
- per-checkpoint tau {'C1': 0.1, 'C2': 0.2, 'C3': 0.1}; prior-matched tau 0.1
- cross-fitted dev forecast of the gain +0.16; projected test SE 0.33, so ~0.91 points detectable with 80% power

## Verdict: FAILURE (no detectable effect; the +1.0 target is excluded)

| condition | met |
|---|---|
| delta_ge_1 | no |
| ci_lower_gt_0 | no |
| fear_improves | no |
| disgust_improves | yes |
| accuracy_guardrail | yes |
| weighted_f1_guardrail | yes |
| class_guardrail | yes |
| every_checkpoint_positive | no |

## Test results (mean over included checkpoints)

| arm | acc | weighted F1 | macro-F1 | F1 neu | F1 sur | F1 fea | F1 sad | F1 joy | F1 dis | F1 ang |
|---|---|---|---|---|---|---|---|---|---|---|
| BASE | 67.32 | 66.46 | 50.87 | 79.79 | 58.97 | 26.12 | 40.95 | 64.95 | 31.43 | 53.89 |
| LA* | 66.73 | 66.16 | 50.63 | 79.31 | 58.92 | 25.03 | 41.41 | 64.77 | 31.48 | 53.51 |
| NO* | 65.72 | 65.56 | 50.78 | 77.99 | 58.47 | 26.16 | 43.33 | 64.27 | 31.87 | 53.34 |
| LA1 | 53.24 | 57.16 | 44.24 | 64.67 | 56.68 | 13.73 | 38.67 | 62.07 | 25.75 | 48.12 |
| S1_per_checkpoint_tau | 66.42 | 65.99 | 50.54 | 79.00 | 58.54 | 24.85 | 41.55 | 64.80 | 31.49 | 53.57 |
| S2_prior_matched_tau | 66.73 | 66.16 | 50.63 | 79.31 | 58.92 | 25.03 | 41.41 | 64.77 | 31.48 | 53.51 |

- Delta macro-F1 (LA* - BASE): **-0.24 [-1.06, +0.68]** (procedure-level two-sample dialogue bootstrap (tau re-selected per replicate))
- test-only CI with tau* fixed: [-0.74, +0.27]
- per checkpoint: C1 -0.60, C2 +0.50, C3 -0.63
- Delta accuracy -0.59 [-3.95, +0.00]; Delta weighted F1 -0.30 [-2.05, +0.05]

| class | Delta F1 [95% CI] | Delta precision | Delta recall | support |
|---|---|---|---|---|
| neu | -0.48 [-3.48, +0.00] | +0.75 | -1.86 | 1256 |
| sur | -0.06 [-1.25, +0.74] | -0.77 | +0.71 | 281 |
| fea | -1.09 [-4.82, +1.06] | -4.66 | +0.67 | 50 |
| sad | +0.45 [-0.71, +4.14] | -1.57 | +1.44 | 208 |
| joy | -0.18 [-1.29, +0.42] | -0.60 | +0.25 | 402 |
| dis | +0.06 [-1.91, +5.40] | -3.11 | +1.47 | 68 |
| ang | -0.38 [-2.14, +0.32] | -0.93 | +0.19 | 345 |

- Neutral-only offset NO*: -0.10 [-0.81, +0.58]; LA* - NO*: -0.14 [-0.90, +0.78]
- LA1 (tau = 1, untuned): -6.63 [-9.01, -4.16]
- S1 per-checkpoint tau: -0.33 [-0.93, +0.31] | S2 prior-matched tau (0.1): -0.24 [-0.73, +0.27]
- predicted / true count, BASE -> LA*: neu 1.07->1.04, sur 1.02->1.05, fea 0.63->0.76, sad 0.68->0.73, joy 1.02->1.04, dis 0.62->0.71, ang 1.00->1.02
- fidelity: BASE weighted F1 66.46 vs paper 66.46 (ok)

## Dev diagnostics

- pairwise AUROC: fea vs neu 0.949, fea vs sur 0.873, fea vs sad 0.845, fea vs ang 0.842, dis vs neu 0.940, dis vs ang 0.851, sad vs neu 0.870
- one-vs-rest AUROC / AP: neu 0.893/0.853, sur 0.918/0.673, fea 0.829/0.288, sad 0.844/0.477, joy 0.877/0.677, dis 0.861/0.316, ang 0.898/0.642
- checkpoint recall train -> dev: neu 89->85, sur 75->65, fea 31->24, sad 55->37, joy 71->63, dis 40->35, ang 64->54
- RoBERTa-only probe: macro-F1 train 87.7 -> dev 41.3; recall neu 93->75, sur 90->57, fea 93->10, sad 75->36, joy 81->49, dis 96->27, ang 76->36
- dev recall, first (n=343): neu 84 (n=160), sur 57 (n=47), fea 25 (n=8), sad 32 (n=23), joy 72 (n=64), dis 29 (n=7), ang 49 (n=34)
- dev recall, same (n=339): neu 89 (n=176), sur 77 (n=29), fea 10 (n=10), sad 45 (n=32), joy 53 (n=29), dis 0 (n=3), ang 57 (n=60)
- dev recall, shift (n=427): neu 81 (n=134), sur 65 (n=74), fea 30 (n=22), sad 35 (n=56), joy 59 (n=70), dis 47 (n=12), ang 55 (n=59)
- branch margins, fea missed_as_neutral (n~8): total +1.99 = self_contagion +0.65 + cross_emotion +0.60 + self_event +0.46 + cross_event +0.29 + bias -0.01
- branch margins, fea correct (n~10): total +1.47 = self_contagion +0.48 + cross_emotion +0.65 + self_event +0.01 + cross_event +0.33 + bias +0.01
- branch margins, dis missed_as_neutral (n~4): total +3.08 = self_contagion +1.03 + cross_emotion +0.87 + self_event +0.58 + cross_event +0.60 + bias +0.01
- branch margins, dis correct (n~8): total +1.57 = self_contagion +0.55 + cross_emotion +0.80 + self_event +0.07 + cross_event +0.17 + bias -0.01
- branch margins, sad missed_as_neutral (n~40): total +1.75 = self_contagion +0.20 + cross_emotion +0.72 + self_event +0.38 + cross_event +0.45 + bias -0.01
- branch margins, sad correct (n~41): total +1.54 = self_contagion +0.87 + cross_emotion +0.52 + self_event +0.00 + cross_event +0.14 + bias +0.01
- branch margins, ang missed_as_neutral (n~33): total +1.27 = self_contagion +0.11 + cross_emotion +0.40 + self_event +0.38 + cross_event +0.39 + bias -0.02
- branch margins, ang correct (n~83): total +2.14 = self_contagion +1.01 + cross_emotion +0.83 + self_event +0.10 + cross_event +0.18 + bias +0.02
- branch margins, sur missed_as_neutral (n~16): total +2.01 = self_contagion +0.56 + cross_emotion +0.74 + self_event +0.31 + cross_event +0.42 + bias -0.02
- branch margins, sur correct (n~97): total +2.27 = self_contagion +0.88 + cross_emotion +0.82 + self_event +0.37 + cross_event +0.19 + bias +0.02
- branch margins, joy missed_as_neutral (n~36): total +1.47 = self_contagion +0.40 + cross_emotion +0.52 + self_event +0.21 + cross_event +0.34 + bias -0.01
- branch margins, joy correct (n~103): total +1.89 = self_contagion +0.92 + cross_emotion +0.58 + self_event +0.26 + cross_event +0.11 + bias +0.01
- dev errors made by all checkpoints: neu 62/82, sur 47/57, fea 30/31, sad 67/73, joy 54/67, dis 14/15, ang 62/76
