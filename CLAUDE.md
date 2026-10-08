# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of the official PyTorch implementation of **ECERC** (Evidence-Cause Attention Network for Multi-Modal Emotion Recognition in Conversation, ACL 2025, `TAN-OpenLab/ECERC`). It holds **Person B's part** (Keshav Mishra) of the three-person course project "ECERC Revisited". The proposal is `docs/Project_SOP.pdf`, which is git-ignored. The project keeps the ECERC backbone unchanged and studies three components, one per team member:

1. **Confusion-aware contrastive regularizer**: not in this repo.
2. **Class imbalance on MELD (Limitation 2)**: class-balanced weights, focal exponent and Fear/Disgust oversampling. **This repo.**
3. **Confidence-Aware Cause Gating (CACG)**: not in this repo.

The Limitation 2 study is complete. It found **no test improvement** over the original objective, and that negative result is the outcome. The full write-up is `docs/report.md`. Paper reference numbers are MELD 66.46 wF1 / 67.32 Acc, averaged over 3 released checkpoints. Only `ECERC_MODEL.pkl` was evaluated here; this is "R0" (67.55 / 66.58 on a Colab T4).

## Repository layout

| Path | What it is |
|---|---|
| `README.md` | Evaluator landing page |
| `docs/report.md` | Full report |
| `docs/reproduce.md` | Reproduction steps |
| `docs/upstream_README.md` | Original ECERC README |
| `MELD/train_imbalance.py` | Person B training and evaluation script. Variants: R0 `--eval_only`, B0 (no options), CB `--cb_beta`, FL `--gamma`, OS `--oversample --os_rho`. Adds macro-F1, per-class P/R/F1/TP/FP/FN, confusion matrices and saved predictions. |
| `MELD/analyze_imbalance.py` | `compare` (epoch-by-epoch equality of two runs) and `summary --split valid\|test` (tables, paired Δ vs B0, confusion-matrix PNGs, pilot rule) |
| `notebooks/01–03_*.ipynb` | Executed Colab notebooks: the execution evidence. Code cells and outputs are **never edited**; their structure and quirks are documented in `notebooks/README.md` |
| `results/` | Curated saved outputs of every reported run (stage0, stage1, stage2): metrics.json, config.json, epochs.csv, logs, prediction CSVs and summaries. **No `best_model.pkl`** |
| `scripts/make_figures.py` | Builds `docs/figures/*.png` from `results/stage2/runs/*/metrics.json` |
| `IEMOCAP/`, other `MELD/` files, `requirement.yml`, `OS_info.txt` | Upstream, unchanged |

ECE and reliability diagrams are **not** computed anywhere in this repo.

## Rules for working in this repo

- **Do not modify any upstream file.** This means everything under `IEMOCAP/`, the upstream `MELD/` files and the checkpoints.
- **Do not modify `MELD/train_imbalance.py` or `MELD/analyze_imbalance.py`.** Their SHA-256 hashes are recorded in the notebooks (`2ab7406b…` and `6fe10b21…`) and tie the results to the code. New analysis goes in new files.
- **Never invent or adjust numbers.** Every number in the docs must come from the notebook outputs or the files in `results/`.
- **Describe the configurations exactly:**
  - B0 is the original MELD objective, which is already **focal loss with γ = 1**.
  - CB adds class-balanced α weights **inside that γ = 1 focal loss**, not cross-entropy.
  - FL2 changes γ from 1 to 2.
  - OS changes only the training sampler.
- **Describe the use of the test set exactly.** Selection used validation only: the checkpoint is the epoch with the best validation wF1, and the pilot used validation macro-F1. The test set *was* evaluated and logged every epoch, inherited from upstream `train.py`. Never write that the test set was not evaluated.
- **Scratch output is not committed.** `MELD/results/` is git-ignored script output. Curated copies go in the top-level `results/`.

## Environment and commands

```
conda env create -f requirement.yml -n ecerc   # upstream env: Python 3.9, CUDA 11.4 / cuDNN 8.2 (authors: Windows 10 + A100)
```

The reported Limitation 2 runs did **not** use this environment. They ran on Google Colab with a Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu130, NumPy 2.1.3 and scikit-learn 1.6.1. In that environment the runs were deterministic across sessions.

There are no tests, linter or build step. Each dataset folder is a self-contained script set. **Run scripts from inside that folder**, because imports are local (`from model import ECERC`) and data paths are relative:

```
cd IEMOCAP && python inference.py          # evaluate ECERC_MODEL.pkl on the test set
cd IEMOCAP && python train.py              # train from scratch
cd MELD    && python inference.py --load_model_state_dir ECERC_MODEL.pkl
cd MELD    && python train_imbalance.py --dry_run [variant options]   # checks only; works without a GPU
cd MELD    && python train_imbalance.py --seed 0 [--cb_beta B | --gamma G | --oversample --os_rho R]
cd MELD    && python analyze_imbalance.py summary --split test results/<run dirs>
python scripts/make_figures.py             # from the repo root
```

Data is not in the repo. Download the preprocessed pickles (Google Drive link in `docs/upstream_README.md`) into `data/` at the repo root, which is `../data/` relative to each dataset folder. The dataloaders hard-code these paths and ignore `--data_dir`:
- `../data/iemocap/IEMOCAP_features.pkl`, `iemocap_emotion_features_roberta.pkl`, `iemocap_emotion_semantic_features_roberta.pkl`
- `../data/meld/meld_emotion_semantic_features_roberta.pkl`

Pretrained checkpoints (`ECERC_MODEL.pkl`, plus `_2` and `_3` variants on HF `zt-ai/ECERC`) are state dicts loaded with `torch.load`. The paper's numbers are the average over the 3 checkpoints.

CLI flag spelling differs between datasets: IEMOCAP uses `--batch_size` and `--no_cuda`, MELD uses `--batch-size` and `--no-cuda`. `--class_weight` is `store_true` with `default=True` on IEMOCAP, so it cannot be turned off from the CLI. On MELD it defaults to False.

## Architecture

`IEMOCAP/` and `MELD/` each hold their own copy of `model.py`, `loss.py`, `dataloader.py`, `train.py` and `inference.py`. They are near-duplicates, so **a change to shared logic must be made in both**. The intentional per-dataset differences are:

| | IEMOCAP | MELD |
|---|---|---|
| Positional encoding | added | multiplied by `0.` (disabled) |
| dropout1 / dropout2 | 0.5 / 0.5 | 0.0 / 0.2 |
| `Loss` gamma | 0 (weighted NLL) | 1 (already focal) |
| class weights | `1/freq` | `log(1/freq)`, off by default |
| classes / speakers | 6 / 2 | 7 / 9 |
| seed, lr, epochs, patience, bs | 2007, 1e-4, 200, 50, 64 | 0, 1e-5, 40, 20, 32 |
| validation | first 10% of `trainVid`, not shuffled | official valid split |

**Data flow.** Each batch from `collate_fn` is `(emo_roberta, sem_roberta, audio, visual, qmask, umask, labels, vids)`. The first five are padded sequence-first `(L, B, D)`; `umask` and `labels` are batch-first. In `train_or_eval_model`, when `feature_type == "multi"`, text (1024) + audio + visual are concatenated into `U_e`. `U_s` is the "semantic/event" RoBERTa feature (1024). Audio dims are 1582 (IS10) on IEMOCAP and 300 on MELD; visual is 342 (denseface). The model returns log-probs already flattened over valid utterances, `(sum(seq_lengths), n_classes)`, and labels are flattened to match.

**`ECERC.forward`** (`model.py`) maps onto the paper's stages:
1. **Evidence gating**: `ModalFilterV2` projects T/A/V to `hidden_size=128` each, applies cross-modal sigmoid gates and concatenates the result to 384 dims.
2. **Cause encoding**: `emotion_encoding` runs self-attention over the gated evidence. `event_encoding` runs self-attention over the projected `U_s`.
3. **Evidence-cause interaction**: four cross-attention `Encoder`s, each restricted by a mask:
   - `self_event_attention` uses `imask` (identity)
   - `cross_event_attention` uses `cmask` (earlier utterances by other speakers)
   - `self_contagion_attention` uses `smask & ~imask` (the same speaker's earlier utterances)
   - `cross_emotion_attention` uses `cmask`

   The masks are built by turning the one-hot `qmask` into an integer speaker id and combining it with a causal (`submask`) and padding mask. The two event branches attend only on the first `hidden_size` dims (the text slice) and then re-append the A/V slices.
4. **Feature gating**: each of the 4 cause features gets an **element-wise sigmoid gate**. `gate_reset` covers the two emotion branches and `gate_reset2` the two event branches. The gated features are concatenated to 4×384 dims.
5. **Classification**: `smax_fc` followed by `log_softmax`.

Note for CACG: the code has **no softmax over the four cause types and no temperature**, unlike the SOP's formulation `g = softmax(W[f1..f4]+b)`. Implementing CACG means introducing a cause-level distribution (or adapting the formulation) at the `wR_lis` step. Keep the rest of the backbone intact so that ablations stay clean.

`Loss` (`loss.py`) is a focal loss over **log-probs** (it gathers `logpt` directly) with per-class `alpha`. Pass it log-probabilities, not logits.

## Gotchas

- **CPU runs crash.** In `ECERC.forward`, `imask` and `mask` are assigned only inside `if self.cuda_flag:`, so running without CUDA (e.g. on macOS) raises `NameError`. `torch.load` for the checkpoints may also need `map_location='cpu'`.
- `train_or_eval_model` reads the module-level global `args` (`args.feature_type`), so it only works when `train.py` runs as `__main__`. Importing other helpers from `train.py` is fine; `train_imbalance.py` imports `get_MELD_bert_loaders` and `seed_everything`, and has its own epoch loop.
- Upstream `train.py` **never saves a checkpoint**: `--output_dir` is unused. It prints the test score at the epoch with the best validation F1 (and, on MELD, also at the best validation loss). Early stopping requires both the F1 patience and the loss patience to run out. `train_imbalance.py` keeps this selection logic but also saves the selected epoch's weights as `best_model.pkl`.
- MELD's dataset returns `videoVisual` in slot 3 and `videoAudio` in slot 4, while the training loop unpacks them as `audio, vision`. The model then slices `U_e` using `d_a=300, d_v=342`, so on MELD the "audio" and "visual" slices do not line up with the real modalities. The released checkpoint was trained this way. Do not "fix" the order when evaluating the pretrained weights. If the fix is wanted, retrain and record it as a deviation from the baseline.
- `feature_type='text'` is accepted, but the model always slices `U_e` into T/A/V. Only `multi` actually works.
- Exact reproduction depends on matching the environment (`cudnn.deterministic=True` is set in `seed_everything`).
- Every MELD run evaluates and logs the **test set every epoch**, inherited from `train.py`. Selection reads validation only.
- `train_imbalance.py` refuses to overwrite a finished run folder without `--overwrite`.
