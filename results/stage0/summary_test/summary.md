# Person B results: test split

All metrics in %. Runs are scored at the epoch with the best validation weighted F1 (R0: the released checkpoint).

## Per run

| run | variant | seed | epoch | Acc | wF1 | macro-F1 | Fear P | Fear R | Fear F1 | Disgust P | Disgust R | Disgust F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0_ECERC_MODEL | R0 | - | - | 67.55 | 66.58 | 51.36 | 37.50 | 24.00 | 29.27 | 44.74 | 25.00 | 32.08 |
| B0_s0 | B0 | 0 | 35 | 67.51 | 66.54 | 51.37 | 37.50 | 24.00 | 29.27 | 43.59 | 25.00 | 31.78 |

## Per variant (mean ± std over runs)

| variant | runs | Acc | wF1 | macro-F1 | F1 Neutral | F1 Surprise | F1 Fear | F1 Sadness | F1 Joy | F1 Disgust | F1 Anger |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | 1 | 67.55 | 66.58 | 51.36 | 79.94 | 58.09 | 29.27 | 40.83 | 65.68 | 32.08 | 53.62 |
| B0 | 1 | 67.51 | 66.54 | 51.37 | 79.80 | 58.19 | 29.27 | 41.19 | 65.44 | 31.78 | 53.91 |

## Fear and Disgust (mean ± std over runs; TP/FP/FN are means per run)

| variant | runs | Fear P | Fear R | Fear F1 | Fear TP/FP/FN | Disgust P | Disgust R | Disgust F1 | Disgust TP/FP/FN |
|---|---|---|---|---|---|---|---|---|---|
| R0 | 1 | 37.50 | 24.00 | 29.27 | 12.0/20.0/38.0 | 44.74 | 25.00 | 32.08 | 17.0/21.0/51.0 |
| B0 | 1 | 37.50 | 24.00 | 29.27 | 12.0/20.0/38.0 | 43.59 | 25.00 | 31.78 | 17.0/22.0/51.0 |

## Confusion matrices (rows: true label; summed over runs)

- R0: confusion_R0.csv, confusion_R0.png
- B0: confusion_B0.csv, confusion_B0.png
