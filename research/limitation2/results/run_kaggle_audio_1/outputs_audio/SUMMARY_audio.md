# ECERC Limitation 2, Experiment 6: self-supervised audio evidence

Run 2026-10-18 07:51 | cuda (Tesla T4) | torch 2.11.0+cu128 | transformers 5.16.1 | data sha256 a0349614b2b58e8b...

## Features (train split only)

- audio clips aligned to the feature file; utterances without usable audio: 3 (training mean used)
- selected speech features: **whisper-large-v3 layers 29-32** (out-of-fold train probe macro-F1 27.82); see tables/probe_table.csv

## Stage 1: dev screen (mean best-epoch dev macro-F1 over seeds; selection only, not a result)

| arm | dev macro-F1 | vs A0 |
|---|---|---|
| A0_original | 54.13 | +0.00 |
| A1_routing_fixed | 54.09 | -0.04 |
| A2_ssl_audio | 54.11 | -0.02 |
| A3_ssl_audio_bge_event | 54.21 | +0.08 |

Winner: **A3_ssl_audio_bge_event** (+0.08 on dev); gate +0.50 -> stage 2 skipped, test not evaluated


Checks: 35 passed, 0 warnings
