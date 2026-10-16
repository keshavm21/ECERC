# ECERC Limitation 2, Path B: training-time representation changes

Run 2026-10-11 19:26 | device cuda (Tesla T4) | torch 2.11.0+cu128 | data sha256 a0349614b2b58e8b...

## Stage 1: dev screen (mean best-epoch dev macro-F1 over seeds; selection only, not a result)

| candidate | dev macro-F1 | vs control |
|---|---|---|
| C0_original | 54.12 | +0.00 |
| C1_mean4_layers | 54.05 | -0.07 |
| C2_contrast_0.1 | 53.91 | -0.22 |
| C2_contrast_0.3 | 53.78 | -0.34 |
| C2_contrast_1.0 | 53.07 | -1.05 |
| C3_sentiment_0.3 | 54.31 | +0.19 |
| C3_sentiment_1.0 | 53.24 | -0.89 |
| C4_lr_3e-5 | 54.06 | -0.06 |

Winner: **C3_sentiment_0.3** (+0.19 on dev); gate +0.50 -> stage 2 skipped


Checks: 11 passed, 0 warnings
