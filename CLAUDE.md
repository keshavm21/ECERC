# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Official PyTorch implementation of **ECERC** (Evidence-Cause Attention Network for Multi-Modal Emotion Recognition in Conversation, ACL 2025), cloned from `TAN-OpenLab/ECERC`. This copy is the base for a course project ("ECERC Revisited", proposal in `docs/Project_SOP.pdf`, git-ignored) that keeps the ECERC backbone unchanged and adds three components, each to be ablated in isolation:

1. **Confusion-aware contrastive regularizer** on the fused pre-softmax features, for the pairs hap/exc, ang/fru (IEMOCAP) and dis/ang (MELD).
2. **Class-balanced / focal loss plus minority oversampling**, targeting Fear and Disgust on MELD.
3. **Confidence-Aware Cause Gating (CACG)**: an input-conditioned gating temperature with an entropy-calibration loss `L_gate`.

The combined objective is `L = L_CB-CE + λ1·L_contrast + λ2·L_gate`. Reproduction targets (paper): IEMOCAP 71.78 wF1 / 71.60 Acc, MELD 66.46 wF1 / 67.32 Acc. The project also reports macro-F1, per-class F1, confusion matrices and ECE/reliability diagrams. None of these are computed by the current code; it reports only weighted F1, accuracy, `classification_report` and the confusion matrix.

## Environment and commands

```
conda env create -f requirement.yml -n ecerc   # Python 3.9, PyTorch, CUDA 11.4 / cuDNN 8.2 (authors used Windows 10 + A100)
```

There are no tests, linter or build step. Each dataset folder is a self-contained script set. **Run scripts from inside that folder**, because imports are local (`from model import ECERC`) and data paths are relative:

```
cd IEMOCAP && python inference.py          # evaluate ECERC_MODEL.pkl on the test set
cd IEMOCAP && python train.py              # train from scratch
cd MELD    && python inference.py --load_model_state_dir ECERC_MODEL.pkl
```

Data is not in the repo. Download the preprocessed pickles (Google Drive link in README.md) into a sibling `../data/` folder. The dataloaders hard-code these paths and ignore `--data_dir`:
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
- `train_or_eval_model` reads the module-level global `args` (`args.feature_type`), so the scripts only work when run as `__main__` and cannot be imported.
- `train.py` **never saves a checkpoint**: `--output_dir` is unused. It prints the test score at the epoch with the best validation F1 (and, on MELD, also at the best validation loss). Early stopping requires both the F1 patience and the loss patience to run out.
- MELD's dataset returns `videoVisual` in slot 3 and `videoAudio` in slot 4, while the training loop unpacks them as `audio, vision`. The model then slices `U_e` using `d_a=300, d_v=342`, so on MELD the "audio" and "visual" slices do not line up with the real modalities. The released checkpoint was trained this way. Do not "fix" the order when evaluating the pretrained weights. If the fix is wanted, retrain and record it as a deviation from the baseline.
- `feature_type='text'` is accepted, but the model always slices `U_e` into T/A/V. Only `multi` actually works.
- Exact reproduction depends on matching the pinned environment (`cudnn.deterministic=True` is set in `seed_everything`).
