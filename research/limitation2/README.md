# Limitation 2 (class imbalance): follow-up experiments on MELD

This folder holds seven pre-registered follow-up experiments on ECERC's Limitation 2, run after the Stage 0–2
study in [`docs/report.md`](../../docs/report.md). Nothing outside `research/limitation2/` was changed. The
original code in `MELD/` and `IEMOCAP/` is untouched; the four files this work needs were copied verbatim into
`ecerc_original/` (their SHA-256 hashes match the originals, see `MANIFEST.json`).

Each experiment's rules, success criteria and gate were written into [PROTOCOL.md](PROTOCOL.md) before it ran.
Every decision is made on train/dev only and frozen to a JSON file before the test set is touched; the test set
is then evaluated at most once per experiment. Full write-ups are in
[`results/FINDINGS.md`](results/FINDINGS.md) (per-run findings) and
[`ECERC_L2_followup_report.pdf`](ECERC_L2_followup_report.pdf) (a 19-page report on experiments 1–5: the
audit of the released code, each experiment, why it happened, possible improvements and limitations).
Experiment 6 is written up in `FINDINGS.md` only. Experiment 7 was pre-registered on 2026-10-19 and has not been run yet.

## Results at a glance

| # | Experiment | Platform | Test evaluated? | Verdict |
|---|---|---|---|---|
| 1 | Post-hoc logit adjustment of the three released checkpoints (`argmax(log p − τ·log π_train)`, τ chosen on dev) | Colab T4 | yes | **FAILURE**: Δ macro-F1 −0.24 [95% CI −1.06, +0.68]; the +1.0 target is excluded |
| 2 | Feature-leak check: do the fine-tuned RoBERTa text features carry training labels? | Colab CPU | no (train/dev only) | **NOT CONFIRMED**: gap +7.6 vs control, below the 15 / 10 rule |
| 3 | Path B: training-time representation changes (RoBERTa layer mean, confusion-aware contrastive loss, sentiment head, higher lr) | Kaggle T4 | no (dev gate not passed) | No candidate beat the original recipe by 0.5 dev macro-F1 (best +0.19) |
| 4 | Context-aware RoBERTa-large text expert, fused with ECERC by a dev-chosen weight | Kaggle T4 | yes | **INCONCLUSIVE**: Δ macro-F1 +0.27 [−1.92, +2.29]; Fear F1 26.1 → 29.5, but Disgust and Joy fell |
| 5 | Qwen2.5-7B-Instruct QLoRA expert, fused with ECERC | Kaggle 2× T4 | no (dev gate not passed) | Cross-fitted dev gain −0.98, below the +0.5 gate |
| 6 | Self-supervised audio evidence: frozen Whisper-large-v3 / WavLM-large features replace openSMILE, audio/visual routing fixed (and a bge event-input arm), unchanged ECERC retrained | Kaggle T4 | no (dev gate not passed) | All four arms within 0.1 dev macro-F1 (best +0.08); the better audio trades Sadness (+2.2) against Joy (−2.4) |
| 7 | Zero-shot Qwen2.5-32B-Instruct (4-bit, never trained on MELD labels), label-free calibration, fused with the released ECERC checkpoints by a dev-chosen weight | Kaggle 2× T4 | — | **Pending**: pre-registered 2026-10-19, not yet run; a dev-only 3B pilot gave a cross-fitted dev gain of +0.76 after calibration ([pilot](results/pilot_local_zeroshot_3b/README.md)) |

Taken together: the released models' minority-class errors are not a decision-threshold problem (1), not a
label leak in the features (2), not fixed by representation or optimizer changes inside ECERC (3), only
partly helped by a second model reading the raw text (4, 5), and not fixed by much better audio features or by
correcting the audio/visual routing (6). The MELD test set was evaluated twice in total (experiments 1 and 4).

## Folder layout

| Path | What it is |
|---|---|
| `notebooks/ECERC_L2_posthoc_LA.ipynb` | Experiment 1 (done): post-hoc logit adjustment. Self-contained |
| `notebooks/ECERC_L2_feature_leak_check.ipynb` | Experiment 2 (done): do the text features carry training labels? |
| `notebooks/ECERC_L2_pathB_representation.ipynb` | Experiment 3 (done): training-time representation changes |
| `notebooks/ECERC_L2_expert_fusion.ipynb` | Experiment 4 (done): context-aware text expert fused with ECERC |
| `notebooks/ECERC_L2_llm_fusion.ipynb` | Experiment 5 (done): fine-tuned 7B LLM expert fused with ECERC |
| `notebooks/ECERC_L2_audio_evidence.ipynb` | Experiment 6 (done): self-supervised audio features for ECERC |
| `notebooks/ECERC_L2_zeroshot_llm_fusion.ipynb` | Experiment 7 (pending): zero-shot LLM fused with ECERC |
| `src/l2_posthoc.py` | Experiment 1 library (all its logic; also provides the safe loader and metrics) |
| `src/l2_feature_leak.py` | Experiment 2 library |
| `src/l2_pathb.py` | Experiment 3 library (training, screening, confirmation) |
| `src/l2_expert.py` | Experiment 4 library (text expert, fusion, gate, test) |
| `src/l2_llm.py`, `src/l2_llm_worker.py` | Experiment 5 library and its per-GPU training/prediction worker |
| `src/l2_audio.py` | Experiment 6 library (audio download and alignment, frozen encoders, train-only selection, screen, confirmation, test) |
| `src/l2_zeroshot.py` | Experiment 7 library (cached zero-shot LLM scoring, label-free calibration; fusion, gate and test reuse `l2_expert.py`) |
| `ecerc_original/` | Verbatim copies of `MELD/model.py`, `dataloader.py`, `inference.py`, `loss.py` |
| `tools/build_notebook.py` | Rebuilds all seven notebooks and `MANIFEST.json` after any edit to `src/` |
| `tests/smoke_test*.py` | Local CPU tests of each notebook on synthetic data |
| `ECERC_L2_followup_report.pdf` | Report on the five experiments (written 2026-10-12) |
| `PROTOCOL.md` | Pre-registered rules, criteria and interpretation |
| `MANIFEST.json` | SHA-256 of every file and download source |
| `results/` | Results zips from Colab/Kaggle, the extracted run folders, and `FINDINGS.md` (see [results/README.md](results/README.md)) |

## How to re-run each experiment

The notebooks are self-contained: each embeds its library and the verbatim ECERC files, downloads the data
and checkpoints, and checks every SHA-256. The sections below are in reverse order of the experiments.

### Experiment 7: zero-shot LLM fusion on Kaggle (about 1 to 1.5 hours on GPU T4 x2)

Same steps as Experiment 3 below, with three differences:
- import `research/limitation2/notebooks/ECERC_L2_zeroshot_llm_fusion.ipynb`;
- choose **GPU T4 x2** (the 4-bit 32B model needs both GPUs) and **Internet → On**, because the model is a 19 GB
  download, pinned to a fixed revision;
- download `ecerc_l2/ecerc_l2_zeroshot_results.zip` from **Output** when it finishes.

If a single GPU is selected, the setup cell stops with a message before downloading anything. The notebook prints
the LLM's dev progress, the dev-only fusion analysis and the gate decision. If the gate fails, it finishes normally
and the test set stays sealed.

### Experiment 6: audio evidence on Kaggle (2.5 to 3.5 hours on a T4 if Stage 2 runs; the recorded run stopped at the gate after 1 h 48 min)

Same steps as Experiment 3 below, with two differences: import
`research/limitation2/notebooks/ECERC_L2_audio_evidence.ipynb`, and download
`ecerc_l2/ecerc_l2_audio_results.zip` from **Output** when it finishes. **Internet → On** is required: the notebook
downloads about 8 GB (the MELD features, the MELD audio and three frozen encoders from Hugging Face), all
SHA-256 checked or pinned to a fixed revision. Only one of the two T4s is used.

What it prints along the way:
- the alignment of the audio with the feature file (about 3 utterances without usable audio are expected);
- extraction progress for Whisper-large-v3 (the longest step, about 25 minutes) and WavLM-large;
- the train-only feature selection table and the selected encoder and layer band;
- 20 dev-screen runs, the Stage 1 table and the gate decision;
- if the gate passed, 40 confirmatory runs and then the single test evaluation.

If the gate fails, the notebook still finishes normally and writes the zip; the test set stays sealed.

### Experiment 5: LLM expert on Kaggle (about 3 to 6 hours on GPU T4 x2; the recorded run took 1 h 40 min)

Same steps as Experiment 3 below, with three differences: import
`research/limitation2/notebooks/ECERC_L2_llm_fusion.ipynb`, choose **GPU T4 x2** (two seeds train in parallel,
one per GPU), and download `ecerc_l2/ecerc_l2_llm_results.zip` from **Output** when it finishes.

### Experiment 4: text expert on Kaggle (about 1.5 to 2 hours on a GPU, runs in the background)

Same steps as Experiment 3 below, with two differences: import
`research/limitation2/notebooks/ECERC_L2_expert_fusion.ipynb`, and download
`ecerc_l2/ecerc_l2_expert_results.zip` from **Output** when it finishes.

### Experiment 3: Path B on Kaggle (about 1.5 to 2.5 hours on a GPU, runs in the background)

1. Sign in at <https://www.kaggle.com>. The first time only: open your profile → **Settings** → **Phone verification**
   (Kaggle needs this before it allows GPUs and internet access).
2. **Create → New Notebook**. In the notebook: **File → Import Notebook** → upload
   `research/limitation2/notebooks/ECERC_L2_pathB_representation.ipynb`.
3. In the right-hand panel (**Session options**): **Accelerator → GPU T4 x2** and **Internet → On**.
4. Top right: **Save Version** → choose **Save & Run All (Commit)** → **Save**. It now runs in the background;
   you can close the tab.
5. When the version finishes (Kaggle emails you, or check **View Active Events**), open the notebook's version,
   go to **Output**, and download `ecerc_l2/ecerc_l2_pathb_results.zip`. Put it in `research/limitation2/results/`.
   If it failed, open the version's **Logs** and send the last error.

If the Google Drive download fails on Kaggle (quota message), wait an hour and commit again. If it keeps
failing, upload the data file as a private Kaggle dataset (**Add Input → Upload**) and add
`data_local_path='/kaggle/input/<your-dataset>/meld_emotion_semantic_features_roberta.pkl'` to `CONFIG`
in the first code cell.

### Experiment 2: feature-leak check on Google Colab (about 10 minutes, CPU)

Same steps as below, with two differences: upload `notebooks/ECERC_L2_feature_leak_check.ipynb`, and choose a
**CPU** runtime (no GPU needed). The download is `ecerc_l2_leak_results.zip`; put it into `results/`.

### Experiment 1: post-hoc logit adjustment on Google Colab (about 15 minutes)

1. Go to <https://colab.research.google.com> and sign in with a Google account.
2. **File → Upload notebook** → choose `research/limitation2/notebooks/ECERC_L2_posthoc_LA.ipynb` from this repository.
3. **Runtime → Change runtime type** → Hardware accelerator **T4 GPU** → **Save**.
   If no GPU is available, choose **CPU**: it still works, just more slowly.
4. **Runtime → Run all**. If Colab warns that the notebook was not authored by Google, click **Run anyway**.
5. Wait until the last cell prints the summary. Your browser then downloads **`ecerc_l2_results.zip`**.
   If it doesn't, click the folder icon on the left, open `ecerc_l2/`, and download `ecerc_l2_results.zip`
   from its ⋮ menu.
6. Put the zip into `research/limitation2/results/` in this repository (no need to unzip it).
   If you can't download it, copy the summary printed by the last cell and paste it instead.

Rules while it runs: don't edit cells and don't re-run single cells. The test set is evaluated exactly
once per run, and the first complete run is the result.

## If something goes wrong

| Message | What to do |
|---|---|
| `Automatic download from Google Drive failed` | Open the link printed in the error, download `meld_emotion_semantic_features_roberta.pkl` in your browser, upload it through the Colab file panel into `ecerc_l2/data/`, then **Runtime → Run all** again |
| `Safe loader refused the data file (blocked global ...)` | In the first code cell, set `TRUST_DATA_PICKLE = True`, then **Run all** again. The file comes from the authors' official link and the Colab VM is isolated |
| A check marked `[FAIL]` stops the run | Copy the full error and send it. Nothing has touched the test set at that point |
| The runtime disconnects | **Runtime → Run all** again. Downloads restart from scratch on a new VM |
| Experiment 6: `retry ... after error` lines, or a Hugging Face download error | Transient network errors are retried five times. If it still stops, commit the version again; nothing has touched the test set at that point |

## Local checks (optional, CPU)

```bash
cd research/limitation2
python3 tools/build_notebook.py          # after editing anything in src/
.venv/bin/python tests/smoke_test.py          # Experiment 1; needs torch, numpy, pandas, scikit-learn, matplotlib
.venv/bin/python tests/smoke_test_leak.py     # Experiment 2
.venv/bin/python tests/smoke_test_pathb.py    # Experiment 3
.venv/bin/python tests/smoke_test_expert.py   # Experiment 4; also needs transformers
.venv/bin/python tests/smoke_test_llm.py      # Experiment 5; also needs transformers and peft
.venv/bin/python tests/smoke_test_audio.py    # Experiment 6; also needs transformers, tokenizers and soundfile
.venv/bin/python tests/smoke_test_zeroshot.py # Experiment 7; also needs transformers and tokenizers
```

The smoke tests run each notebook's real code path on small synthetic data (synthetic numbers are meaningless).
They also check that no test-split label is read before the freeze, and that gated runs skip the test.
Experiment 6's test also plants the awkward cases of the real audio release (a missing clip, an extra clip, a text
mismatch, a gap in the utterance IDs, an over-long and an over-short clip) and checks that the test CSV is read
without its emotion column.
