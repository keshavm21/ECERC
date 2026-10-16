# Limitation 2 (class imbalance): post-hoc logit adjustment on MELD

This folder holds everything added for the investigation of ECERC's Limitation 2. Nothing outside
`research/limitation2/` was changed. The original code in `MELD/` and `IEMOCAP/` is untouched; the four
files this work needs were copied verbatim into `ecerc_original/` (their SHA-256 hashes match the originals,
see `MANIFEST.json`).

The experiment: the three released ECERC MELD checkpoints stay fixed (no training). Each prediction becomes
`argmax(log p - tau * log(train class prior))`, with `tau` chosen on the dev set only. The test set is used
once, after every decision is frozen. Rules and success criteria are in [PROTOCOL.md](PROTOCOL.md).

## Folder layout

| Path | What it is |
|---|---|
| `notebooks/ECERC_L2_posthoc_LA.ipynb` | Experiment 1 (done): post-hoc logit adjustment. Self-contained |
| `notebooks/ECERC_L2_feature_leak_check.ipynb` | Experiment 2 (done): do the text features carry training labels? |
| `notebooks/ECERC_L2_pathB_representation.ipynb` | Experiment 3 (done): training-time representation changes |
| `notebooks/ECERC_L2_expert_fusion.ipynb` | Experiment 4 (done): context-aware text expert fused with ECERC |
| `notebooks/ECERC_L2_llm_fusion.ipynb` | **Experiment 5 (run next, on Kaggle):** fine-tuned 7B LLM expert fused with ECERC |
| `src/l2_posthoc.py` | Experiment 1 library (all its logic; also provides the safe loader and metrics) |
| `src/l2_feature_leak.py` | Experiment 2 library |
| `src/l2_pathb.py` | Experiment 3 library (training, screening, confirmation) |
| `src/l2_expert.py` | Experiment 4 library (text expert, fusion, gate, test) |
| `src/l2_llm.py`, `src/l2_llm_worker.py` | Experiment 5 library and its per-GPU training/prediction worker |
| `ecerc_original/` | Verbatim copies of `MELD/model.py`, `dataloader.py`, `inference.py`, `loss.py` |
| `tools/build_notebook.py` | Rebuilds the notebook and `MANIFEST.json` after any edit to `src/` |
| `tests/smoke_test*.py` | Local CPU tests of each notebook on synthetic data |
| `PROTOCOL.md` | Pre-registered rules, criteria and interpretation |
| `MANIFEST.json` | SHA-256 of every file and download source |
| `results/` | Results zips from Colab, extracted runs, and `FINDINGS.md` |

## Experiment 5: LLM expert on Kaggle (about 3 to 6 hours on GPU T4 x2, runs in the background)

Same steps as Experiment 3 below, with three differences: import
`research/limitation2/notebooks/ECERC_L2_llm_fusion.ipynb`, choose **GPU T4 x2** (two seeds train in parallel,
one per GPU), and download `ecerc_l2/ecerc_l2_llm_results.zip` from **Output** when it finishes.

## Experiment 4: text expert on Kaggle (about 1.5 to 2 hours on a GPU, runs in the background)

Same steps as Experiment 3 below, with two differences: import
`research/limitation2/notebooks/ECERC_L2_expert_fusion.ipynb`, and download
`ecerc_l2/ecerc_l2_expert_results.zip` from **Output** when it finishes.

## Experiment 3: Path B on Kaggle (about 1.5 to 2.5 hours on a GPU, runs in the background)

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

## Experiment 2: feature-leak check on Google Colab (about 10 minutes, CPU)

Same steps as below, with two differences: upload `notebooks/ECERC_L2_feature_leak_check.ipynb`, and choose a
**CPU** runtime (no GPU needed). The download is `ecerc_l2_leak_results.zip`; put it into `results/`.

## Experiment 1: how to run it on Google Colab (about 15 minutes)

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

## Local checks (optional, CPU)

```bash
cd research/limitation2
python3 tools/build_notebook.py   # after editing src/l2_posthoc.py
.venv/bin/python tests/smoke_test.py   # needs a venv with torch, numpy, pandas, scikit-learn, matplotlib
```
