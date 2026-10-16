# ECERC Revisited — Limitation 2: Class Imbalance on MELD

This repository is my part (**Person B**, Keshav Mishra) of a three-person course project. The project starts from the official implementation of **ECERC** (Evidence-Cause Attention Network for Multi-Modal Emotion Recognition in Conversation, ACL 2025).

**The question:** can standard class-imbalance techniques improve ECERC on MELD's rarest emotions, *Fear* and *Disgust*, without hurting overall performance?

**The answer, under a validation-only selection protocol with 3 seeds per configuration: no.** None of the four interventions beat the original ECERC objective on the held-out test set in mean accuracy, weighted F1 or macro-F1. Every intervention had lower test macro-F1 than the baseline on all three seeds. I report this as a negative result.

| Configuration | Accuracy | Weighted F1 | Macro-F1 | Fear F1 | Disgust F1 |
|---|---|---|---|---|---|
| **B0** (original ECERC objective) | **66.87 ± 0.65** | **66.00 ± 0.54** | **50.40 ± 0.87** | 26.22 ± 3.29 | 30.72 ± 1.43 |
| CB0.999 (class-balanced weights) | 64.88 ± 0.36 | 65.36 ± 0.24 | 49.52 ± 0.70 | 22.37 ± 2.82 | 29.69 ± 1.13 |
| FL2 (focal γ 1 → 2) | 66.76 ± 0.52 | 65.87 ± 0.35 | 50.06 ± 0.69 | 25.49 ± 2.77 | 30.16 ± 1.78 |
| OS2 (Fear/Disgust oversampling) | 66.60 ± 0.27 | 65.78 ± 0.40 | 50.02 ± 0.93 | 25.51 ± 2.29 | 30.94 ± 1.36 |
| FL2-OS2 (FL2 + OS2) | 66.26 ± 0.19 | 65.37 ± 0.28 | 49.13 ± 0.45 | 22.46 ± 0.42 | 30.31 ± 0.88 |

*Table notes: MELD test set (2610 utterances; 50 Fear, 68 Disgust). Mean ± sample std over seeds 0, 1, 2. Each run is scored at the epoch with the best validation weighted F1.*

![Stage 2 test results](docs/figures/fig1_test_overall.png)

**Main findings** (details in the [full report](docs/report.md)):

- **Class-balanced weighting trades precision for recall, at a loss.**
  - Fear recall rose from 22.67 to 36.67, but Fear precision fell from 31.27 to 16.13.
  - Only about 9% of the extra Fear predictions were correct. F1 rises only if more than ~13% are, so Fear F1 fell, and accuracy dropped 2 points.
- **FL2 and OS2 stay within about 0.4 points of B0** on every overall metric.
  - OS2's mean Disgust F1 is nominally higher than B0's: +0.22 ± 0.36 points paired by seed, higher on 2 of 3 seeds. That gain is far smaller than the seed standard deviations (1.36 and 1.43), and OS2's overall metrics are slightly lower than B0's.
- **FL2-OS2 has the lowest test macro-F1 (49.13).** It is below both FL2 and OS2 on accuracy, weighted F1, macro-F1 and Fear F1. Its Disgust F1 falls between the two.
- **Small validation gains did not carry over to test.**
  - All four variants were slightly above B0 on validation weighted F1.
  - FL2 was above B0 on all three seeds on validation, then below B0 on test macro-F1 on all three seeds.
  - Validation has only 40 Fear and 22 Disgust utterances.

---

## Contents

1. [ECERC in brief](#ecerc-in-brief)
2. [Limitation 2 and my contribution](#limitation-2-and-my-contribution)
3. [Experimental design](#experimental-design)
4. [Repository layout](#repository-layout)
5. [Reproducing the results](#reproducing-the-results)
6. [What comes from upstream and what is mine](#what-comes-from-upstream-and-what-is-mine)
7. [Acknowledgments and citation](#acknowledgments-and-citation)

## ECERC in brief

ECERC ([Zhang & Tan, ACL 2025](https://aclanthology.org/2025.acl-long.102/); [upstream code](https://github.com/TAN-OpenLab/ECERC)) recognizes the emotion of each utterance in a conversation. It uses text, audio and visual features in four steps:

1. Gate each modality's *evidence* against the others.
2. Encode a semantic *cause* representation.
3. Model four kinds of evidence-cause interaction with masked cross-attention (self-event, cross-event, self-contagion, cross-emotion).
4. Classify the gated combination of these interactions.

On MELD the paper reports 67.32% accuracy and 66.46 weighted F1 (its Table 3). The upstream README describes these as an average over 3 runs. The authors released three checkpoints.

## Limitation 2 and my contribution

MELD is highly imbalanced. Fear and Disgust are each under 3% of the training utterances, against 47% for Neutral. The released checkpoint reaches only 29.27 F1 on Fear and 32.08 on Disgust, against 79.94 on Neutral.

*ECERC Revisited* keeps the ECERC backbone fixed and studies three limitations, one per team member:

1. a confusion-aware contrastive regularizer;
2. **class imbalance on MELD (this repository)**;
3. confidence-aware cause gating.

Components 1 and 3 and the combined ablation are not part of this repository.

My contribution:

- **Baseline verification on MELD** (Stage 0).
  - Evaluated the released checkpoint (R0) in my environment.
  - Retrained the baseline (B0).
  - Showed that, in my environment, my training script reproduces the upstream `train.py` exactly: all 40 epochs, all logged metrics.
- **A controlled study of four imbalance interventions:**
  - class-balanced loss weights;
  - a stronger focal exponent;
  - dialogue-level oversampling of Fear/Disgust;
  - focal loss combined with oversampling.

  The study used a validation-only pilot (Stage 1) followed by a 15-run final comparison (Stage 2).
- **Evaluation beyond the upstream metrics:** macro-F1, per-class precision/recall/F1, TP/FP/FN counts, confusion matrices, per-seed comparisons against B0, and a validation-vs-test comparison.
- **Analysis of why the interventions did not help:** the precision-recall trade-off on the minority classes and the instability of validation selection with very few minority examples.

## Experimental design

| Stage | Purpose | Runs | Decisions based on |
|---|---|---|---|
| **0: baseline verification** | Released checkpoint and retrained baseline in my environment | R0; upstream `train.py` seed 0; B0 seed 0 | — |
| **1: pilot** | Choose β (CB) and ρ (OS) | CB β ∈ {0.999, 0.9999}; OS ρ ∈ {2, 4}; FL γ = 2; seed 0 | **validation** macro-F1 (rule fixed in code before the pilot ran) |
| **2: final** | Compare the five configurations | 5 configurations × seeds {0, 1, 2} = **15 runs** | none: all runs reported |

What each configuration changes:

- **B0** is the original ECERC MELD objective. The upstream MELD loss is already a **focal loss with γ = 1**, not cross-entropy.
- **CB0.999** applies class-balanced weights (Cui et al., 2019; β = 0.999), computed from training counts, inside that same γ = 1 focal loss.
- **FL2** raises the focal exponent **from γ = 1 to γ = 2**. γ = 2 was fixed in advance, not tuned in the pilot.
- **OS2** changes the **training sampling strategy**. Dialogues containing Fear or Disgust are drawn twice as often, with replacement. The loss is unchanged.
- **FL2-OS2** combines FL2 and OS2.

Everything else matches upstream `MELD/train.py`: architecture, features, optimizer, schedule, and checkpoint selection by best validation weighted F1.

**About the test set.** Every run evaluates and logs the test set at every epoch, as upstream `train.py` does. **The test set was not used for any selection**: not for the checkpoint epoch, not for β or ρ, and not for which configurations to run.

**Environment.** All runs used Google Colab with an NVIDIA Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu130 (CUDA 13.0), NumPy 2.1.3 and scikit-learn 1.6.1. This differs from the authors' environment:

- the upstream [`requirement.yml`](requirement.yml) pins Python 3.9.1, PyTorch 2.0.1 (CUDA 11.8 build), NumPy 1.25.0 and scikit-learn 1.3.2;
- [`OS_info.txt`](OS_info.txt) records Windows 10, an NVIDIA A100 80GB, CUDA 11.4 and cuDNN 8.2.4.

In my environment, R0 scores 67.55 accuracy / 66.58 weighted F1 on test. R0 is a single released checkpoint, not a reproduction of the paper's 3-run average.

## Repository layout

```
├── README.md                    this page
├── docs/
│   ├── report.md                full write-up: design, all results, analysis, limitations
│   ├── reproduce.md             step-by-step reproduction (Colab or local GPU)
│   ├── figures/                 summary figures (generated by scripts/make_figures.py)
│   └── upstream_README.md       the original ECERC README, unchanged
├── notebooks/                   the executed Colab notebooks, kept unmodified as evidence
│   └── README.md                which stage each notebook covers, and known quirks
├── results/                     saved summaries of every reported stage (no per-run files)
│   └── README.md                index of the saved artifacts
├── scripts/make_figures.py      builds docs/figures/ from the per-run metrics files
├── MELD/
│   ├── train_imbalance.py       [mine] training/evaluation for R0, B0, CB, FL, OS, FL+OS
│   ├── analyze_imbalance.py     [mine] equivalence check, summaries, confusion matrices, pilot rule
│   └── model.py, train.py, …    upstream ECERC, unchanged
├── IEMOCAP/                     upstream ECERC, unchanged (not used in this work)
└── requirement.yml, OS_info.txt upstream environment description, unchanged
```

## Reproducing the results

Full instructions are in [`docs/reproduce.md`](docs/reproduce.md). In short, on a CUDA GPU with the preprocessed MELD features in `data/meld/`:

```bash
cd MELD
python train_imbalance.py --eval_only --load_model_state_dir ECERC_MODEL.pkl   # R0
for s in 0 1 2; do
  python train_imbalance.py --seed $s                                          # B0
  python train_imbalance.py --seed $s --cb_beta 0.999                          # CB0.999
  python train_imbalance.py --seed $s --gamma 2                                # FL2
  python train_imbalance.py --seed $s --oversample --os_rho 2                  # OS2
  python train_imbalance.py --seed $s --gamma 2 --oversample --os_rho 2        # FL2-OS2
done
python analyze_imbalance.py summary --split test results/{B0,CB0.999,FL2,OS2,FL2-OS2}_s{0,1,2}
```

Each 40-epoch run takes about 1–2 minutes on a T4, plus data loading. The model only runs on a GPU, but `--dry_run` checks data and settings without one.

## What comes from upstream and what is mine

| Upstream ECERC ([TAN-OpenLab/ECERC](https://github.com/TAN-OpenLab/ECERC)), unchanged | This work |
|---|---|
| `IEMOCAP/` (all files) | `MELD/train_imbalance.py`, `MELD/analyze_imbalance.py` |
| `MELD/model.py`, `dataloader.py`, `loss.py`, `train.py`, `inference.py` | `notebooks/` (executed experiment notebooks) |
| `MELD/ECERC_MODEL.pkl`, `IEMOCAP/ECERC_MODEL.pkl` (released checkpoints) | `results/` (saved summaries) |
| `requirement.yml`, `OS_info.txt`, `docs/upstream_README.md` | `docs/` (except `upstream_README.md`), `scripts/`, this README |

**Proof that the upstream files are unchanged:**

- Git shows no changes to them.
- The notebooks record their SHA-256 hashes and re-check them after training (`sha256sum -c`: all OK).

The experiment scripts import the upstream model, dataset, loss and data loaders. They do not copy or edit them.

## Acknowledgments and citation

**AI assistance.** AI tools were used to assist with the development of the experimental scripts, result-analysis/figure-generation tooling, and project documentation. All experimental configurations, executions, results, and final scientific interpretations were reviewed and verified against the recorded experiment outputs.

**Upstream.** The model, data pipeline, preprocessed features and released checkpoints are by the ECERC authors. Their README, including their license statement, is preserved in [`docs/upstream_README.md`](docs/upstream_README.md). If you use ECERC, please cite:

```bibtex
@inproceedings{zhang-tan-2025-ecerc,
    title = "{ECERC}: Evidence-Cause Attention Network for Multi-Modal Emotion Recognition in Conversation",
    author = "Zhang, Tao and Tan, Zhenhua",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.102/",
    pages = "2064--2077"
}
```

Methods used: class-balanced loss (Cui et al., CVPR 2019) and focal loss (Lin et al., ICCV 2017). Dataset: MELD (Poria et al., ACL 2019).
