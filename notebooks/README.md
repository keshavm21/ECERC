# Experiment notebooks

These are the three Google Colab notebooks in which every reported run was executed. They are kept **exactly as saved from Colab**, with no edits to code cells or outputs, so they serve as the execution record. Only the file names were changed:

| File | Original Colab name | Stage(s) |
|---|---|---|
| [`01_stage0_released_checkpoint.ipynb`](01_stage0_released_checkpoint.ipynb) | `ECERC_MELD_Baseline.ipynb` | Stage 0: released checkpoint (R0) through upstream `inference.py` |
| [`02_stage0_retrain_and_stage1_pilot.ipynb`](02_stage0_retrain_and_stage1_pilot.ipynb) | `Copy_of_ECERC_MELD_Baseline.ipynb` | Stage 0 (B0 retraining, equivalence check) **and** Stage 1 (pilot) |
| [`03_stage2_final_experiment.ipynb`](03_stage2_final_experiment.ipynb) | `Another_copy_of_ECERC_MELD_Baseline.ipynb` | Stage 2 (15 final runs and summaries) |

The notebooks contain no markdown cells. The guide below lists the cells that matter. Cell numbers count from 0 in the order the cells are stored in the file.

To re-run the experiments, use [`../docs/reproduce.md`](../docs/reproduce.md), not these notebooks. The notebooks depend on my Google Drive layout.

## Code that the notebooks ran

Each notebook clones the **upstream** repository (`TAN-OpenLab/ECERC`) and uploads `train_imbalance.py` and `analyze_imbalance.py` from my machine (`files.upload()`). Notebooks 02 and 03 print the SHA-256 of both uploaded scripts and of the upstream MELD files:

| File | SHA-256 printed in notebooks 02 and 03 |
|---|---|
| `train_imbalance.py` | `2ab7406bb3daf89b06c2cafb4dfcdc53987b5ff625aba6788b615a26c019c7b5` |
| `analyze_imbalance.py` | `6fe10b219a094b647ae070fab0091be93593cc0aae25e9a865a1fe7f1857a412` |
| `model.py` | `a09dbe3aa6fd662237fb9833a0544887e9441e46a08783ca39b363d97d6ccc32` |
| `dataloader.py` | `121db840ea1d57c2814edb061909fcb9c31f9df944f73e1be42b5e6f8c1387ad` |
| `loss.py` | `c348375c27c366e11f2509de30dd0c5c0f224adcf0eb434bc0d0d9fa12045246` |
| `train.py` | `83117a9f2003da23ff84f826a5a7a89c19a3911b48bcf9b3ebab554647f88936` |
| `inference.py` | `21f7ce13ac1693d51f605ccc6298f1582432fab27dc3b989cdec0a3ba388e28f` |
| `ECERC_MODEL.pkl` | `0f44a6e546592a3f7843342c0f72dcfa591eab1f99cdbd8a7d4ea7aed607441c` |

These are identical to the SHA-256 of the corresponding files in this repository's `MELD/` folder, so the code that produced the results is the code in this repository.

## 01: Stage 0, released checkpoint

| Cells | What happens |
|---|---|
| 0–16 | Mount Drive, clone upstream, copy the checkpoint and the MELD features from Drive, print environment (Python 3.13.15, PyTorch 2.11.0+cu130, Tesla T4) |
| 19 | `python inference.py`: R0 on test, **acc 67.55, wF1 66.58**, with classification report and confusion matrix |
| 21–22 | The same run again, saved to Drive with `tee`; identical output |
| 24 | Writes a short text summary to Drive |

## 02: Stage 0 (B0 and equivalence check), then Stage 1 (pilot)

| Cells | What happens |
|---|---|
| 0–14 | Setup; script and upstream-file hashes (above) saved as `results/baseline_sha256.txt` |
| 15 | 1-epoch smoke test (`--epochs 1 --run_dir results/_smoke`); not a reported result |
| 16 | **R0** via `--eval_only`: test acc 67.55, wF1 66.58, macro-F1 51.36; validation numbers too |
| 17–21 | `--dry_run` for B0, CB β=0.999, CB β=0.9999, OS ρ=2, OS ρ=4: split counts, class weights, sampler mix |
| 22 | Upstream `python -u train.py`, seed 0, logged to `results/orig_train_s0.log` |
| 23 | **B0 seed 0** with `train_imbalance.py` |
| 24 | `analyze_imbalance.py compare results/orig_train_s0.log results/B0_s0` → **`RESULT: IDENTICAL. All 40 epochs match exactly.`** |
| 25 | `sha256sum -c results/baseline_sha256.txt`: all upstream files and the checkpoint OK |
| 26 | Test summary of R0 and B0 seed 0 |
| 27–28 | Copy to Drive (`results/stage0`) and list the files |
| **Stage 1** | |
| 30–34 | Pilot runs, seed 0: CB β=0.999, CB β=0.9999, FL γ=2, OS ρ=2, OS ρ=4 |
| 35 | Validation summary with the **pilot choice: β = 0.999, ρ = 2** |
| 36–37 | Sanity comparisons showing that the pilot settings do change training |
| 39 | Same validation summary, saved to `results/stage1_summary` |
| 40 | Writes `stage1_record.txt` (selected settings) |
| 41–42 | Copy to Drive (`results/stage1`) and list the files |

## 03: Stage 2, final experiment

| Cells | What happens |
|---|---|
| 0–14 | Setup, same as notebook 02 (same script and file hashes) |
| 15 | R0 via `--eval_only` (same numbers as in notebook 02) |
| 16, 18, 20, 22, 24 | Seeds 0, 1, 2 of B0, CB0.999, FL2, OS2 and FL2-OS2 (one cell per configuration, three runs per cell) |
| 17, 19, 21, 23, 25 | Copy each configuration's three runs to Drive (`results/stage2`) |
| 26 | **Validation summary** of all 15 runs (`results/stage2_summary_valid`) |
| 29 | **Test summary** of all 15 runs (`results/stage2_summary_test`) |
| 27, 28 | Copy the test and validation summaries to Drive |

## Known quirks

These are recorded here instead of being fixed, so that the notebooks stay unmodified.

- **Missing execution counts (01, 02).** Colab did not keep the execution counts in these two files, so the run order cannot be read from the files themselves. The outputs are consistent with top-to-bottom execution.
- **Cells out of order (03).** The execution counts show that cells 26–29 ran in the order 26, 28, 29, 27. Cell 27, which copies the *test* summary to Drive, appears above cell 29, which creates it, but ran after it.
- **Duplicate cells (02).**
  - The environment and hash checks run twice (cells 5 and 14, cells 7/9 and 13).
  - The validation summary runs twice (cells 35 and 39). The second run only adds `--out results/stage1_summary`, and the printed content is identical.
- **Hashes taken before the Drive copy (02, 03).** Cell 5 hashes the checkpoint that came with the upstream clone, and cell 6 then overwrites it with the copy from Drive. Cell 14 hashes it again after the copy. Both hashes are identical (`0f44a6e5…`).
- **Hand-typed file on Drive (01).** Cell 25 overwrites the `meld_baseline_output.txt` file saved in cell 21 with a hand-pasted copy of the same output, which differs only in whitespace in one confusion-matrix row. The saved R0 outputs used in this repository come from `--eval_only` (notebook 02), not from that file.
- **Test logged during the pilot (02).** The pilot runs evaluate and print the test set every epoch, as upstream `train.py` does. The pilot choice (cells 35/39) reads only validation metrics.
- **Seed-0 runs repeated.** Notebook 03 re-runs B0 seed 0 and the seed-0 runs of CB0.999, FL2 and OS2 from notebook 02, in a new Colab session. They print identical results. The saved run files in the raw Drive export confirm this: `metrics.json` and the prediction files are byte-identical, and `epochs.csv` matches apart from wall-clock time (see [`../results/README.md`](../results/README.md)).
