# Reproducing the Limitation 2 experiments

These steps re-run every result in [`report.md`](report.md). They follow what the notebooks in [`../notebooks/`](../notebooks/) actually ran, minus the Colab-specific file shuffling.

## 1. Requirements

**GPU: a CUDA GPU is required.** The upstream `MELD/model.py` builds its attention masks only inside `if self.cuda_flag:`, so training and checkpoint evaluation fail on CPU. `train_imbalance.py --dry_run` runs without a GPU and checks the data and the loss/sampler settings.

**Python packages:** PyTorch, NumPy and scikit-learn. matplotlib is optional: `analyze_imbalance.py` skips the confusion-matrix PNGs without it, and `scripts/make_figures.py` needs it.

**Environment used for every reported result:**

| | Version |
|---|---|
| Platform | Google Colab, NVIDIA Tesla T4 |
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu130 (CUDA 13.0) |
| NumPy | 2.1.3 |
| scikit-learn | 1.6.1 |

This environment differs from the authors' environment:

- the upstream [`requirement.yml`](../requirement.yml) pins Python 3.9.1, PyTorch 2.0.1 (CUDA 11.8 build), NumPy 1.25.0 and scikit-learn 1.3.2;
- [`OS_info.txt`](../OS_info.txt) records Windows 10, an NVIDIA A100 80GB, CUDA 11.4 and cuDNN 8.2.4.

That upstream environment was not used in this work.

**About determinism:**

- **Within my environment:** runs were deterministic. The same configuration and seed gave identical results in separate Colab sessions.
- **On other hardware or software:** expect small differences in the absolute numbers.

## 2. Data and checkpoint

1. **Download the MELD features.** Get `meld_emotion_semantic_features_roberta.pkl` (about 486 MB) from the [ECERC authors' preprocessed features](https://drive.google.com/drive/folders/1DGoPEBvMgMOge7u20wjKn6slYe4awyQi?usp=sharing).
2. **Place the file here:**

   ```
   <repo root>/data/meld/meld_emotion_semantic_features_roberta.pkl
   ```

   The upstream dataloader hard-codes the relative path `../data/meld/…` and ignores `--data_dir`, so it must be read from inside `MELD/`. `data/` is git-ignored.
3. **Check the checkpoint.** The released checkpoint `MELD/ECERC_MODEL.pkl` ships with the upstream repository. Its SHA-256 should be:

   ```
   0f44a6e546592a3f7843342c0f72dcfa591eab1f99cdbd8a7d4ea7aed607441c
   ```

**Run all commands below from inside `MELD/`.** The scripts import the upstream modules locally and write their outputs to `MELD/results/`, which is git-ignored scratch space. Curated copies of my outputs are in the top-level [`results/`](../results/).

### On Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```
```
!git clone https://github.com/keshavm21/ECERC.git /content/ECERC
%cd /content/ECERC/MELD
!mkdir -p ../data/meld
!cp "/content/drive/MyDrive/<your folder>/meld_emotion_semantic_features_roberta.pkl" ../data/meld/
```

Choose a GPU runtime (T4 was used here).

## 3. Stage 0: Baseline verification

```bash
# Optional: confirm the upstream files match the hashes recorded in the notebooks
sha256sum -c ../results/stage0/baseline_sha256.txt

# Data and settings checks (no GPU needed)
python train_imbalance.py --dry_run

# R0: evaluate the released checkpoint on validation and test
python train_imbalance.py --eval_only --load_model_state_dir ECERC_MODEL.pkl
# (equivalently, upstream: python inference.py)

# B0 vs upstream train.py: train both with seed 0, then compare epoch by epoch
python -u train.py 2>&1 | tee results/orig_train_s0.log
python train_imbalance.py --seed 0
python analyze_imbalance.py compare results/orig_train_s0.log results/B0_s0
```

**Expected output**, as recorded in notebook 02:

- R0 test: accuracy 67.55, weighted F1 66.58, macro-F1 51.36.
- B0 seed 0: selected epoch 35, test accuracy 67.51, weighted F1 66.54.
- The comparison ends with `RESULT: IDENTICAL. All 40 epochs match exactly.`

## 4. Stage 1: Pilot (seed 0, validation only)

```bash
python train_imbalance.py --seed 0 --cb_beta 0.999
python train_imbalance.py --seed 0 --cb_beta 0.9999
python train_imbalance.py --seed 0 --gamma 2
python train_imbalance.py --seed 0 --oversample --os_rho 2
python train_imbalance.py --seed 0 --oversample --os_rho 4

python analyze_imbalance.py summary --split valid \
    results/CB0.999_s0 results/CB0.9999_s0 results/FL2_s0 results/OS2_s0 results/OS4_s0 \
    --out results/stage1_summary
```

**How the pilot choice is made.** The `--split valid` summary reads only validation metrics. It then applies the selection rule: highest validation macro-F1 within each family, preferring the milder setting within 0.5 points.

**Expected output:**

```
- CB: beta=0.999 -> 53.16, beta=0.9999 -> 51.89  =>  choose beta=0.999
- OS: rho=2 -> 55.32, rho=4 -> 54.15  =>  choose rho=2
```

## 5. Stage 2: Final experiment (15 runs)

```bash
for s in 0 1 2; do
  python train_imbalance.py --seed $s                                    # B0
  python train_imbalance.py --seed $s --cb_beta 0.999                    # CB0.999
  python train_imbalance.py --seed $s --gamma 2                          # FL2
  python train_imbalance.py --seed $s --oversample --os_rho 2            # OS2
  python train_imbalance.py --seed $s --gamma 2 --oversample --os_rho 2  # FL2-OS2
done

RUNS="results/B0_s0 results/B0_s1 results/B0_s2 \
      results/CB0.999_s0 results/CB0.999_s1 results/CB0.999_s2 \
      results/FL2_s0 results/FL2_s1 results/FL2_s2 \
      results/OS2_s0 results/OS2_s1 results/OS2_s2 \
      results/FL2-OS2_s0 results/FL2-OS2_s1 results/FL2-OS2_s2"
python analyze_imbalance.py summary --split valid $RUNS --out results/stage2_summary_valid
python analyze_imbalance.py summary --split test  $RUNS --out results/stage2_summary_test
```

**Run-folder behavior:**

- **Output location.** Each run writes `results/<variant>_s<seed>/`. The folder holds `config.json`, `train.log`, `epochs.csv`, `best_model.pkl`, `preds_valid.csv`, `preds_test.csv` and `metrics.json`, which is written last.
- **No accidental overwrites.** A folder that already holds a finished run is never overwritten unless you pass `--overwrite`.
- **Seed-0 runs from earlier stages.** B0 and the Stage 1 seed-0 runs already exist from stages 0 and 1, so the loop stops at them with a message. Either skip those commands or reuse the folders. In my environment, re-runs gave identical results.

**Expected output:** the tables in [`report.md` §6](report.md#6-stage-2-final-experiment).

**Runtime:** per-epoch time in the logs is 1.4–2.8 s on a T4 (training plus validation and test passes), so a 40-epoch run takes roughly 1–2 minutes plus data loading.

## 6. Figures

[`scripts/make_figures.py`](../scripts/make_figures.py) reads the 15 per-run `metrics.json` files from a folder laid out as `<dir>/stage2/runs/<run>/metrics.json`. The repository's [`../results/`](../results/) keeps only summaries, not per-run files, so first collect your Stage 2 runs into that layout. From the repository root, after step 5:

```bash
mkdir -p figdata/stage2/runs
cp -R MELD/results/{B0,CB0.999,FL2,OS2,FL2-OS2}_s{0,1,2} figdata/stage2/runs/
python scripts/make_figures.py --results figdata --out docs/figures
```

Only matplotlib is needed. The per-run values the figures plot are listed, rounded to 2 decimals, in `results/stage2/summary_*/per_run.csv`.
