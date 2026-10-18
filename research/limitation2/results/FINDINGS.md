# Findings: run_colab_1 (post-hoc logit adjustment, MELD)

**Run:** 2026-10-11, Colab Tesla T4, original CUDA code path. Raw outputs are in `run_colab_1/outputs/`;
the headline numbers are in `SUMMARY.md`.

**Integrity:**
- All 23 checks passed, with no warnings. Split sizes and train label counts match MELD exactly.
- For all three checkpoints, the pipeline reproduces the original `train_or_eval_model` exactly on dev.
- There was a single run: the freeze (10:20:23) preceded the first test access, and there was no rerun.
- The BASE test scores equal the paper's Table 3 to within rounding (acc 67.32, weighted F1 66.46, per-class
  F1 identical). The paper's reported numbers are therefore the mean of these three released checkpoints.

**Checkpoint provenance:**

| Checkpoint | Seed | Selected by |
|---|---|---|
| C1 | 0 | validation F1 |
| C2 | 2613 | validation F1 |
| C3 | 659 | validation loss |

## Verdict: FAILURE. The +1.0 macro-F1 target is excluded
| | Value |
|---|---|
| Δ macro-F1, LA\* − BASE (τ\* = 0.1, chosen on dev) | **−0.24 [95% CI −1.06, +0.68]** |
| Per checkpoint | C1 −0.60, C2 +0.50, C3 −0.63 |
| Fear / Disgust F1 change | −1.09 / +0.06 |
| Accuracy / weighted F1 change | −0.59 / −0.30 (guardrails met) |
| Neutral-only control NO\* | −0.10 [−0.81, +0.58] |
| Full prior correction LA1 (τ = 1) | −6.63 [−9.01, −4.16] |
| Pre-freeze forecast / detectable effect (80% power) | +0.16 / 0.91 |

The experiment had enough power for the target, so this is a negative result, not an inconclusive one.
On the post-freeze descriptive test frontier, no τ in the grid beats τ = 0 (macro-F1 50.87), even when
judged with hindsight.

## What it establishes (for the released ECERC models)
1. **The minority-class failure is not a decision-threshold (class-prior) problem.** At the margin, the
   items that the adjustment re-labels are mostly correct as they were:
   - 19 extra Fear predictions yielded 1 correct one (5% marginal precision, against 34% base precision).
   - Of 129 utterances moved away from Neutral, 70 really were Neutral.

   The paper's account, that Neutral-heavy training data pulls predictions toward Neutral, is not supported
   as a correctable output bias. The scores already rank Fear and Disgust above Neutral well (dev pairwise
   AUROC 0.95 and 0.94).
2. **Confusion between related emotions is at least as large as the loss to Neutral.**
   - Fear goes to Neutral in 33% of cases, and to Surprise, Sadness or Anger in 38%.
   - Disgust goes to Neutral in 35% of cases and to Anger in 26%.
   - Pairwise dev AUROC is lower for related emotions than for Neutral: Fear vs Surprise/Sadness/Anger
     0.84–0.87, and Disgust vs Anger 0.85.
3. **Errors are systematic, not seed noise.** Of the dev Fear items misclassified by any checkpoint, 30 of 31
   are misclassified by all three; for Disgust it is 14 of 15. Ensembling or extra seeds won't fix this.
4. **There is no evidence of a context "copying" failure.** Minority recall is not worse at emotion shifts,
   and errors predict the previous utterance's label less often (25%) than labels actually repeat (39%).
   Per-branch margins spread across all four cause branches; no single branch drives the Neutral pull.

## Main lead for the next step
- **ECERC underfits minority classes even on its own training data.** Its Fear recall on train is 31%, and
  its train macro-F1 is 63.5 against 54 on dev.
- **The text emotion features alone behave very differently.** A linear probe on them fits training data
  almost perfectly (macro-F1 87.7, Fear recall 93%) but collapses on dev (41.3, Fear recall 10%).

This pattern fits two explanations:
- **(H3) The upstream features were contaminated by training labels.** The RoBERTa feature extractor was
  fine-tuned on the training utterances (COSMIC protocol), so features of training utterances look
  separable in a way held-out features do not. This would also explain why training-time reweighting and
  oversampling raise minority recall but not precision.
- **Ordinary overfitting of a 1024-dimensional probe.**

The two explanations make different predictions for training utterances held out from the probe. That
check is cheap and uses no test data.

## Not resolved
- Which of the two explanations above holds.
- Whether representation-level training methods help.
- How much of the error comes from annotation ambiguity (MELD κ = 0.43).
- Whether the authors' choice between validation-F1 and validation-loss checkpoints was informed by test
  results (the training script prints both). This is unknown.

---

# Findings: run_colab_leak_1 (feature-leak check, train and dev only)

**Run:** 2026-10-11, Colab CPU. Outputs are in `run_colab_leak_1/outputs_leak/`. All 4 checks passed; the data
file is identical to run_colab_1's; the test split was never read.

## Verdict: NOT CONFIRMED (pre-registered rule)
Linear-probe macro-F1, with classifiers fitted on 4/5 of the training dialogues:

| Features | Fitted train | Held-out train | Dev | Gap [95% CI] |
|---|---|---|---|---|
| eroberta1 (fine-tuned RoBERTa) | 91.9 | 47.9 | 40.3 | **+7.6 [+4.4, +11.2]** |
| sroberta1 (plain RoBERTa, control) | 75.9 | 36.8 | 36.8 | +0.0 [−3.1, +3.3] |
| audio | 36.4 | 29.6 | 22.4 | +7.2 [+4.8, +10.0] |
| visual | 26.4 | 14.0 | 14.0 | −0.0 [−1.7, +1.7] |

The emotion-feature gap minus the control's is +7.6 [+4.5, +10.9]. The rule needs a gap of at least 15 and a
difference of at least 10. The 1-nearest-neighbour check agrees: eroberta1 46.3 train vs 42.1 dev; the other
features are flat. Note that the sroberta1 probe reached the 1000-iteration cap; this affects its held-out and
dev predictions equally.

## Interpretation
1. **The large train-vs-dev drop seen in D3 (87.7 → 41.3) was mostly ordinary probe overfitting, not a label
   leak.** On training dialogues the probe hadn't seen, it scores 47.9, close to dev. The strong-leak
   hypothesis (H3) is rejected.
2. **A modest train-side advantage does exist for features built with supervision from labels.**
   - The fine-tuned text features (+7.6) and the audio features (+7.2) both show it. MELD's audio features
     came from SVM-based supervised feature selection; that the training split was used for it is
     plausible but unstated.
   - The unsupervised features (plain RoBERTa; visual features from a model trained on external data) show none.
   - Per class, the text-feature gap is largest for Fear (recall 24 vs 11 on held-out train vs dev; dev has
     only 40 Fear items).
   - This is real but small, so it is a secondary factor, not the main cause of Limitation 2.
3. **Minority emotions carry little linear signal even within training data.** Text-feature Fear recall is 24%
   on unseen training dialogues. Together with run_colab_1, the evidence points to genuine overlap between
   related emotions (plus MELD's label ambiguity, κ = 0.43) as the main bottleneck.

## Next step (pre-registered for this verdict)
Confusion-aware representation learning with retraining (Path B in PROTOCOL.md).

---

# Findings: run_kaggle_pathb_1 (Path B, training-time representation changes)

**Run:** 2026-10-11, Kaggle Tesla T4, 38 minutes. Outputs are in `run_kaggle_pathb_1/outputs_pathb/`.

**Integrity:** all 11 checks passed. The data and code hashes match. The training loop's evaluation reproduces
the original `train_or_eval_model` exactly.

## Verdict: the gate stopped the run after Stage 1
No candidate beat the original recipe by the pre-registered 0.5 dev macro-F1, so Stage 2 and the test
evaluation were skipped. **The test set was not touched** and stays clean for future work.

Mean best-epoch dev macro-F1 over 3 seeds; seed SD is about 0.5–1.1:

| Candidate | Dev macro-F1 | vs C0 |
|---|---|---|
| C0 original (our retraining) | 54.12 | 0.00 |
| C1 mean of 4 RoBERTa layers | 54.05 | −0.07 |
| C2 contrastive, weight 0.1 / 0.3 / 1.0 | 53.91 / 53.78 / 53.07 | −0.22 / −0.34 / −1.05 |
| C3 sentiment head, weight 0.3 / 1.0 | 54.31 / 53.24 | +0.19 / −0.89 |
| C4 learning rate 3e-5 | 54.06 | −0.06 |

## Interpretation
1. **Our retraining reproduces the released models** (dev macro-F1 54.12, against 54.04 for the three
   released checkpoints).
2. **ECERC is not limited by under-training.** At 3× the learning rate it reaches the same dev peak at about
   epoch 18 instead of 38, then over-fits (last-5-epoch mean 50.6). The low train-set recall seen in run 1
   reflects the features, not the optimizer.
3. **No representation change inside ECERC moves dev macro-F1 beyond seed noise.** This covers the extra
   RoBERTa layers, a confusion-aware contrastive loss and sentiment supervision. The stronger weights hurt.
   Fear and Disgust stay at about 30–36 dev F1 under every variant.
4. **Combined with runs 1 and 2:** ECERC's minority-class ceiling is not set by its decision rule, its
   objective or its training budget. The evidence points to the precomputed input features and the data
   itself (overlap between related emotions, MELD label ambiguity) as the limit. Improving the rare emotions
   further therefore requires better evidence features.

---

# Findings: run_kaggle_expert_1 (Experiment 4, text expert fused with ECERC)

**Run:** 2026-10-11, Kaggle Tesla T4, about 2 hours 17 minutes. Outputs are in `run_kaggle_expert_1/outputs_expert/`.

**Integrity:** all 16 checks passed. No seed needed a retry. The cross-fitted dev gain was +0.97, so the gate
passed. The freeze preceded the single test pass.

## Verdict: INCONCLUSIVE (pre-registered rules)
| Test result (mean of 9 ECERC × expert pairs) | Accuracy | Weighted F1 | Macro-F1 | Fear F1 | Sadness F1 | Disgust F1 | Joy F1 |
|---|---|---|---|---|---|---|---|
| ECERC alone (released checkpoints) | 67.32 | 66.46 | 50.87 | 26.12 | 40.95 | 31.43 | 64.95 |
| ECERC + text expert (w = 0.5) | 67.12 | 66.45 | 51.14 | **29.52** | **43.18** | 29.46 | 62.34 |
| Text expert alone | 65.57 | 65.34 | 50.50 | 27.87 | 43.29 | 31.89 | 59.49 |

- **Macro-F1 change:** +0.27 [95% CI −1.92, +2.29]. It is positive in 5 of 9 pairs.
- **The +1.0 target is neither met nor excluded.**
- **Per class:** Fear +3.40 (recall +5.6), Sadness +2.23, Anger +0.82; Disgust −1.97 and Joy −2.61. Joy's drop
  breaks the "no class below −2.0" guardrail. All per-class CIs are wide.
- **Secondary** (pre-registered, not confirmatory): w = 0.2 and τ = 0.1 chosen together on dev gives +1.17
  [−0.01, +2.36], with a test-only CI.
- **Descriptive ensembles:** 3× ECERC 50.31; 3× expert 51.05; fused ensembles 51.42.

## Interpretation
1. **The expert sees evidence ECERC misses, for some rare classes.**
   - On dev it alone matches or exceeds ECERC (macro-F1 about 55 vs 54).
   - On test it misses fewer Fear items (69% vs 79%) and Sadness items (61% vs 66%).
   - Fusion lifts Fear F1, the paper's worst class, from 26.1 to 29.5.
2. **The gains are offset elsewhere.** Disgust and Joy lose, so macro-F1 barely moves and the overall gain is
   not established.
3. **The selected weight is unstable.** Dev macro-F1 is nearly flat for w between 0.15 and 0.6, so the
   selected w varies widely across bootstrap replicates. That spread, plus MELD's small test set (50 Fear,
   68 Disgust), makes the confidence interval about ±2 points. Only a true effect of roughly 2+ macro-F1 points
   could be confirmed on this test set.
4. **Test reuse:** the MELD test set has now been evaluated in experiments 1 and 4 (experiments 2 and 3 never
   touched it). Any further test evaluation must be pre-registered and reported as an additional look.

---

# Findings: run_kaggle_llm_1 (Experiment 5, Qwen2.5-7B-Instruct QLoRA expert fused with ECERC)

**Run:** 2026-10-12, Kaggle 2× Tesla T4, about 1 hour 40 minutes. Outputs are in `run_kaggle_llm_1/outputs_llm/`.

**Integrity:** all 16 checks passed; transformers 5.16.1, peft 0.21.2, bitsandbytes 0.50.2. Both seeds trained
in parallel (98 minutes), with no divergence and no retries.

## Verdict: the gate stopped the run before the test
The cross-fitted dev gain was **−0.98** macro-F1, against a pre-registered threshold of +0.5. The in-sample best
weight (w = 0.45) gave +0.83, but it did not hold out of fold. **The test set was not touched**, so it has still
been evaluated only twice (experiments 1 and 4).

| Dev | Accuracy | Weighted F1 | Macro-F1 | Fear F1 | Disgust F1 | Sadness F1 |
|---|---|---|---|---|---|---|
| ECERC (C1 / C2 / C3) | 67.0 / 66.6 / 66.6 | 65.9 / 65.6 / 65.4 | 54.4 / 53.5 / 54.2 | 30.8 / 32.3 / 30.0 | 42.1 / 35.9 / 43.2 | 44.2 / 44.6 / 43.8 |
| LLM (seed 1 / seed 2) | 67.0 / 69.0 | 66.4 / 66.9 | 53.5 / 52.7 | 36.2 / 24.0 | 23.1 / 22.2 | 42.9 / 47.3 |

## Interpretation
1. **The 7B LLM is a better overall classifier than ECERC but a worse rare-emotion classifier.** It is higher on
   accuracy and weighted F1 (dominated by Neutral, Joy and Surprise) but lower on macro-F1. The cause is mainly
   Disgust (22–23 F1 vs 36–43), which ECERC recognizes and the LLM does not.
2. **It does not address Limitation 2.** A model fine-tuned with ordinary cross-entropy on MELD inherits the same
   class imbalance. Fusing it with ECERC trades Disgust for gains elsewhere, and the net rare-class effect out of
   fold is negative.
3. **Training was short and noisy.** One epoch gave dev macro-F1 swings of 40 → 54 → 51 (seed 1) and
   45 → 47 → 53 (seed 2). A longer run might score higher, but nothing on dev suggests it would fix Disgust.
4. **Across five experiments**, only the RoBERTa expert (experiment 4) improved a rare class on test (Fear
   26.1 → 29.5), and its overall macro-F1 gain was not established (+0.27, CI −1.9 to +2.3). Within this setup,
   on MELD's test set, Limitation 2 is robust to decision-level, objective-level, training-level and
   ensemble-level interventions.

---

# Findings: run_kaggle_audio_1 (Experiment 6, self-supervised audio evidence)

**Run:** 2026-10-18, Kaggle Tesla T4, 1 hour 48 minutes. Feature extraction took 62 minutes (Whisper-large-v3 41,
WavLM-large 20.5, bge 0.3), the probes 18 and the dev screen 27. Outputs are in `run_kaggle_audio_1/outputs_audio/`.

**Integrity:**
- All 35 checks passed, with no warnings.
  - The data file and the six audio files match their pinned SHA-256.
  - The alignment rule held for every split: offsets 0 / 1039 / 1153; 1035/1038, 114/114 and 280/280 dialogues
    matched exactly; text agreement 99.99 / 100 / 100%; no train or dev label differs.
  - Three utterances have no usable audio (training mean used): train dia125_utt3 and dev dia110_utt7 (no clip), and
    train dia556_utt6 (MELD's CSV text and clip belong to a different line).
  - All features are finite. Our evaluation loop reproduces the original `train_or_eval_model` exactly, for both the
    original and the new input layout.
- **A0 reproduces Path B's C0 exactly** on the shared seeds (55.12, 54.04, 53.21), so this run and Path B are directly
  comparable.
- The test set was not touched; it has still been evaluated only twice (experiments 1 and 4).

## Verdict: the gate stopped the run after Stage 1
The best arm beat the original recipe by +0.08 dev macro-F1, against a pre-registered gate of +0.5. Stage 2 and the
test evaluation were skipped.

**Feature selection (training split only).** The selected speech features are Whisper-large-v3 layers 29–32. Linear
probe, audio only (macro-F1):

| Features | Out-of-fold train | Dev (mean of fold models) |
|---|---|---|
| Whisper-large-v3, layers 29–32 (selected) | 27.82 | 28.66 |
| WavLM-large, layers 17–20 (its best out-of-fold band) | 26.80 | 29.05 |
| openSMILE (original audio input) | 29.46 | 22.52 |
| `sroberta1` (original event input) | 36.83 | 36.89 |
| bge-large-en-v1.5 (A3's event input) | 36.03 | 36.75 |

The probes of Whisper layers 1–24 and of `sroberta1` reached the pre-registered 1000-iteration cap; the selected band
did not.

**Stage 1 (dev, 5 seeds per arm; each run at its best-dev-macro-F1 epoch):**

| Arm | Dev macro-F1 (SD) | vs A0 | Weighted F1 | Fear F1 | Disgust F1 | Sadness F1 | Joy F1 |
|---|---|---|---|---|---|---|---|
| A0 original | 54.13 (0.84) | 0.00 | 66.15 | 33.80 | 34.84 | 43.73 | 62.99 |
| A1 routing fixed | 54.09 (0.56) | −0.04 | 65.90 | 33.40 | 36.84 | 44.13 | 63.05 |
| A2 + speech features | 54.11 (1.10) | −0.02 | 65.89 | 32.81 | 36.57 | 45.94 | 60.56 |
| A3 + bge event input | 54.21 (1.33) | +0.08 | 66.15 | 32.01 | 35.98 | 46.28 | 61.57 |

Mean over the last five epochs, a measure that does not pick a lucky epoch: A0 53.42, A1 53.23, A2 53.39, A3 53.37.

## Interpretation
1. **The audio/visual routing defect is harmless.** A1 and A0 share their initialisation and data order exactly, so
   their paired difference isolates the routing: −0.04 dev macro-F1 (SD 0.41 across seeds). The original loader's
   slot order is not a cause of Limitation 2.
2. **The new speech features are better audio evidence on their own.**
   - Their dev probes score 28.7 (the selected Whisper band) and up to 29.2 (WavLM), against 22.5 for openSMILE.
   - They show no train-side advantage: out-of-fold train and dev agree. openSMILE shows the same +7 gap as in
     experiment 2 (29.5 train against 22.5 dev).
3. **ECERC does not gain from them.**
   - A2 differs from A0 by −0.02 dev macro-F1 (paired SD 1.79, higher in 3 of 5 seeds).
   - The class changes trade against each other. Sadness F1 rises in all 5 seeds (+2.21 on average), Joy falls in 4
     of 5 (−2.43), and Fear and Disgust move within seed noise (dev has 40 Fear and 22 Disgust items).
   - On top of ECERC's text evidence, the better audio shifts the boundaries between related classes rather than
     separating them. This agrees with MELD being text-dominant (MERBench) and with the small audio gain reported for
     TelME (+0.62 weighted F1). MiSTER-E's +2.0 did not carry over. That is a single-run figure from a different
     architecture with bidirectional context, and the paper does not say whether those encoders were frozen.
4. **A new event encoder adds nothing.** As a linear-probe feature, bge-large is no better than plain RoBERTa (36.0
   against 36.8 out of fold). A3's +0.10 over A2 is within seed noise, and it vanishes on the last-five-epoch measure.
5. **Across the six follow-ups:** ECERC's rare-emotion ceiling is unchanged by the decision rule (1), the training
   objective and optimiser (3), fusion with text and LLM experts (4, 5) and, now, by much better audio evidence and
   correct modality routing (6). The same direction of class trading appeared with new text evidence on test in
   experiment 4 (Sadness +2.23, Joy −2.61). What remains points to the text evidence and the labels themselves:
   overlap between related emotions and annotator agreement κ = 0.43.
