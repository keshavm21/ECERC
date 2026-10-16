# ECERC Limitation 2, Experiment 5: fine-tuned LLM expert fused with ECERC

Run 2026-10-12 12:13 | cuda ['Tesla T4', 'Tesla T4'] | torch 2.11.0+cu128 | transformers 5.16.1 | peft 0.21.2 | bitsandbytes 0.50.2

## Dev (selection only)

- ECERC alone, dev macro-F1: C1 54.42, C2 53.54, C3 54.16
- LLM alone, dev macro-F1: seed 1 53.51 (weighted F1 66.38), seed 2 52.65 (weighted F1 66.94)
- w* = 0.45; dev macro-F1 54.87 vs 54.04 for ECERC alone
- cross-fitted dev gain **-0.98** (gate +0.50: not passed, so the test was skipped); projected test SE 0.85
- dev, mean per pair: fea ECERC wrong but LLM right 4.5 / reverse 2.7 (n=40); dis ECERC wrong but LLM right 0.5 / reverse 5.2 (n=22); sad ECERC wrong but LLM right 11.0 / reverse 12.5 (n=111)


Checks: 16 passed, 0 warnings
