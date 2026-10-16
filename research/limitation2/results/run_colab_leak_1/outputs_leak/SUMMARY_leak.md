# ECERC Limitation 2: do the text emotion features carry the training labels?

Run 2026-10-11 10:54 | 2 CPU cores | sklearn 1.6.1 | data sha256 a0349614b2b58e8b... | train/dev only (test split never read)

## Verdict: NOT CONFIRMED

Rule: LEAK CONFIRMED if gap(eroberta1) >= 15 macro-F1 points and gap(eroberta1) - gap(sroberta1) >= 10 with a 95% bootstrap CI lower bound > 0; otherwise NOT CONFIRMED. gap = macro-F1 on held-out training dialogues (out-of-fold) minus the mean dev macro-F1 of the same fold models.

- gap, emotion text features (eroberta1): **+7.6** points
- gap, plain text features (sroberta1): +0.0 points
- difference: **+7.6 [95% CI +4.5, +10.9]**

| features | fitted train | held-out train | dev | gap [95% CI] | 1-NN train | 1-NN dev |
|---|---|---|---|---|---|---|
| eroberta1 | 91.9 | 47.9 | 40.3 | +7.6 [+4.4, +11.2] | 46.3 | 42.1 |
| sroberta1 | 75.9 | 36.8 | 36.8 | +0.0 [-3.1, +3.3] | 27.2 | 28.2 |
| videoAudio | 36.4 | 29.6 | 22.4 | +7.2 [+4.8, +10.0] | 20.2 | 19.9 |
| videoVisual | 26.4 | 14.0 | 14.0 | -0.0 [-1.7, +1.7] | 15.2 | 15.1 |

Macro-F1 (%) of linear probes fitted on 4/5 of the training dialogues (5 folds). "Held-out train" = the remaining training dialogues; 1-NN = nearest neighbour in other dialogues.

| class | eroberta1 held-out train / dev recall | sroberta1 held-out train / dev recall |
|---|---|---|
| neu | 79 / 74 | 76 / 75 |
| sur | 62 / 52 | 52 / 55 |
| fea | 24 / 11 | 11 / 14 |
| sad | 38 / 31 | 21 / 20 |
| joy | 60 / 52 | 49 / 50 |
| dis | 30 / 25 | 14 / 12 |
| ang | 45 / 37 | 31 / 31 |

Checks: 4 passed, 0 warnings
