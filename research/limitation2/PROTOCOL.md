# Protocol: post-hoc logit adjustment for ECERC Limitation 2 (MELD)

Fixed before any test-set access. The notebook implements exactly this; `src/l2_posthoc.py` is the
reference, and its SHA-256 is recorded in `MANIFEST.json` and in the run's `frozen_config.json`.

## Question
Does a train-prior logit adjustment of the released ECERC MELD checkpoints improve test macro-F1 (the
paper's Limitation 2: rare emotions lost to Neutral and to related emotions)? Training data, weights and
architecture are all unchanged.

## Models
The three released checkpoints, C1 to C3 (HuggingFace `zt-ai/ECERC`, SHA-256 checked; C1 is identical to the
in-repo `MELD/ECERC_MODEL.pkl`). A checkpoint whose embedded archive name mentions `test` (i.e. selected on
test) is excluded from the primary analysis and reported separately.

## Arms (from the same log-probabilities)
| Arm | Rule | Chosen on |
|---|---|---|
| BASE | argmax log p | — |
| **LA\*** (primary) | argmax [log p − τ·log π_train], τ ∈ {0.0, 0.1, …, 2.0} | dev |
| NO\* (control) | argmax [log p − δ·1(Neutral)], δ ∈ {0.00, 0.15, …, 3.00} | dev |
| LA1 (reference) | τ = 1 | — |

π_train comes from training labels only. Selection: the value with the highest **mean dev macro-F1 over
included checkpoints**; ties go to the smaller value. Sensitivity analyses (reported, never decisive):
S1 per-checkpoint τ; S2 τ chosen on dev reweighted to the train prior; S3 test-only CI with τ fixed;
S4 all checkpoints included.

## Data roles and test lock
- **Train:** class prior; diagnostic D3.
- **Dev:** fidelity checks, τ/δ selection, cross-fitted forecast, power projection, diagnostics, temperature.
- **Test:** nothing before the freeze, not even label counts. `stage_test` refuses to run unless
  `frozen_config.json` exists unchanged and the analysis code hash matches, and it reads every rule from that file.
- The repo's hardcoded class weights (computed with test labels) are never used. `MELD/train.py` and
  `MELD/inference.py` are never run as scripts, because both print test metrics.

## Hard checks (the run stops before the freeze if any fails)
- Checkpoint hashes match; strict state-dict loading into the unchanged `ECERC` class.
- Data layout (17 fields), disjoint dialogue IDs, feature dimensions, train/dev label validity.
- Two dev passes agree; our loop reproduces the original `train_or_eval_model` dev confusion matrix
  (≤ 2 differing predictions); dev accuracy ≥ 55%.

## Statistics
- **Primary estimand:** Δ = mean over included checkpoints of [test macro-F1(LA\*) − macro-F1(BASE)],
  in points.
- **95% CI:** two-sample dialogue-level bootstrap, B = 10,000, seed 20251010. Each replicate resamples dev
  dialogues and re-selects τ and δ, then resamples test dialogues. The CI therefore covers test-sampling and
  selection noise, conditional on these checkpoints; it does not cover training-seed variation.
- **Power:** a cross-fitted dev forecast (5 dialogue folds) projects the test SE before the freeze.

## Verdict (applied mechanically, first match wins)
| Verdict | Rule |
|---|---|
| SUCCESS | Δ ≥ +1.0; CI lower > 0; mean F1 change > 0 for Fear and for Disgust; Δ > 0 for every included checkpoint; accuracy change ≥ −1.0; weighted-F1 change ≥ −0.5; no class's mean F1 change < −2.0 |
| TRADE-OFF | as SUCCESS, but a guardrail (last three conditions) fails |
| PARTIAL | Δ ≥ +1.0 and CI lower > 0, but Fear, Disgust or per-checkpoint consistency fails |
| FAILURE | CI upper < +1.0: the target is excluded (subtype: real but small / no effect / harmful) |
| INCONCLUSIVE | anything else |

Guardrails are judged on point estimates; their CIs are reported. Secondary: LA\* − NO\* (≥ +0.5 with
CI > 0 means a prior-shaped correction matters beyond suppressing Neutral); LA1 shows whether the
balanced-error setting over-corrects. Fidelity flag: the run is flagged, not changed, if BASE weighted F1
differs from the paper's 66.46 by more than 1.5.

## Dev diagnostics (interpretation only; they cannot change arms, grids or criteria)
- **D1:** one-vs-rest AUROC/AP, pairwise AUROC for Fear vs Neutral/Surprise/Sadness/Anger,
  Disgust vs Neutral/Anger, Sadness vs Neutral.
- **D3:** checkpoint recall on train (eval mode) vs dev; a RoBERTa-only logistic probe (eroberta1, C = 1,
  no tuning) on train vs dev.
- **D5:** recall by emotion shift; exact per-branch split of the logit margin (`smax_fc` is linear over the
  four cause branches).
- **D6:** per-item error consistency across checkpoints.
- **Calibration:** NLL and ECE, raw and with dev-fitted temperature.

## Pre-specified interpretation
- **SUCCESS / TRADE-OFF:** part of Limitation 2 is decision-level prior bias in the released models.
  Still open: confusion among related emotions, training-seed generalization (needs retraining), whether
  training-time or representation fixes do better, and the authors' causal claim about Neutral-heavy
  training data.
- **FAILURE / INCONCLUSIVE:** the next direction follows the diagnostics.

  | Diagnostic finding | Next direction |
  |---|---|
  | Low pairwise AUROC (D1) | Confusion-aware representation learning |
  | Train recall much higher than dev, also in the probe (D3) | Upstream feature shift: cross-fitted features or less reliance on them |
  | Low minority recall on train too (D3) | Optimization study |
  | Errors concentrated at emotion shifts with the emotion branches dominating the Neutral margin (D5) | Shift-aware gating |

  IEMOCAP is considered only if MELD shows a real positive effect (CI lower > 0).

## Known limits
Three checkpoints selected by the authors on dev weighted F1, so dev is used twice. MELD's small test set
(about 70 Fear and 70 Disgust items expected) limits resolution: the pre-freeze projection reports the
detectable effect. Published test numbers motivated the question but are not used for any decision.

---

# Addendum (2026-10-11, fixed before running): feature-leak check

Motivated by the first run's D3 diagnostic. A linear classifier on the text emotion features (`eroberta1`)
scored macro-F1 87.7 on the training data it was fitted to, but 41.3 on dev.

## Question
Do the text emotion features carry the training labels? They come from a RoBERTa model fine-tuned on the
MELD training utterances. Only train and dev are used; the test split is never read.

## Design
- Same classifier as D3: standardized features, logistic regression, C = 1, max_iter = 1000.
- Each classifier is fitted on 4/5 of the training dialogues (5 folds, seed 2026) and scored on the
  held-out 1/5 and on dev.
- **gap** = macro-F1 on held-out training dialogues (out-of-fold) − mean dev macro-F1 of the same fold models.
- **Features:** `eroberta1` (primary); `sroberta1`, plain RoBERTa, not fine-tuned (control); audio and visual
  (descriptive).
- A 1-nearest-neighbour check (neighbours from other dialogues only) is reported descriptively.
- Uncertainty: dialogue bootstrap, B = 2000, seed 777.

## Rule
**LEAK CONFIRMED** if gap(eroberta1) ≥ 15 points **and** gap(eroberta1) − gap(sroberta1) ≥ 10 with a 95% CI
lower bound > 0. Otherwise **NOT CONFIRMED**.

## Next step by outcome
- **LEAK CONFIRMED:** rebuild the text evidence features with cross-fitting (k-fold RoBERTa fine-tuning,
  out-of-fold features for training utterances), retrain ECERC with the original objective, then re-measure
  Limitation 2 and re-test the cheap fixes on clean features.
- **NOT CONFIRMED:** the features transfer as expected and the bottleneck is genuine overlap between emotions.
  Next is confusion-aware representation learning (anchor or prototype contrastive loss for Fear vs
  Sadness/Anger/Surprise and Disgust vs Anger), retrained over 10 seeds.

---

# Addendum (2026-10-11, fixed before running): Path B, training-time representation changes

The feature-leak check's verdict was NOT CONFIRMED, which leads to Path B. All model training uses the
unchanged ECERC architecture, dataset class and loss. Extra heads exist only during training.

## Stage 1: dev screen
3 seeds (101, 102, 103) per candidate; 40 epochs; each run keeps its best-dev-macro-F1 epoch.

| Candidate | Change from C0 |
|---|---|
| C0 | Original recipe: Adam lr 1e-5, weight decay 2e-4, batch 32, focal γ = 1, layer-1 RoBERTa features |
| C1 | Inputs = mean of the four provided RoBERTa layers (same 1024-d shape) |
| C2 | + SPCL-style prototype contrastive loss on the fused 1536-d representation (projection 256 → 128, T = 0.1, EMA prototypes with momentum 0.9). Confusable negatives (Fear vs Sadness, Anger, Surprise; Disgust vs Anger) count ×2. Weight 0.1 / 0.3 / 1.0 |
| C3 | + sentiment head on MELD's training sentiment labels. Weight 0.3 / 1.0 |
| C4 | Learning rate 3e-5 |
| C5 | The best member of each family that beats C0 on dev, combined (3 seeds) |

**Winner:** the highest mean dev macro-F1 among C1 to C5.
**Gate:** Stage 2 runs only if the winner beats C0 by at least 0.5 dev macro-F1.

## Freeze
The winner, seeds, checkpoint rule and criteria are written to `frozen_config_pathb.json` before any test access.

## Stage 2: confirmatory
- Seeds 201 to 210, × {C0, winner}. Each seed gives both arms the same initialization and data order.
- Each run keeps its best-dev-macro-F1 epoch. The test set is evaluated once, after all training.

## Statistics
- **Primary:** mean paired difference in test macro-F1 (winner − C0) across the 10 seeds.
- **95% CI:** hierarchical bootstrap that resamples seeds and test dialogues (B = 10,000, seed 20261011).
  A seed-level t-interval is also reported.

## Verdict
The thresholds are the same as experiment 1. Consistency now means the difference is > 0 in at least 70% of seeds.

| Verdict | Rule |
|---|---|
| SUCCESS | Δ ≥ +1.0; CI lower > 0; mean F1 change > 0 for Fear and for Disgust; consistency; accuracy change ≥ −1.0; weighted-F1 change ≥ −0.5; no class's mean F1 change < −2.0 |
| TRADE-OFF | as SUCCESS, but a guardrail fails |
| PARTIAL | Δ ≥ +1.0 and CI lower > 0 only |
| FAILURE | CI upper < +1.0 |
| INCONCLUSIVE | anything else |

## Notes
- The baseline here is our own retraining of C0, not the released checkpoints. Its test scores are reported
  next to the paper's for reference.
- C3 uses MELD's training sentiment annotations, which the original ECERC does not use. This is disclosed as
  extra training-time supervision from the same dataset.

---

# Addendum (2026-10-12, fixed before running): Experiment 4, a context-aware text expert fused with ECERC

## Motivation
- **Runs 1–3:** ECERC's rare-emotion ceiling is set by its precomputed input features.
- **Literature:** a RoBERTa model fine-tuned end to end on the dialogue text reaches about the same MELD score
  alone as ECERC (TelME's text teacher: 66.57 weighted F1). It learns from raw text with context rather than
  from frozen features, so its errors may differ from ECERC's.

## Expert
- **Model:** RoBERTa-large (`FacebookAI/roberta-large`).
- **Input:** past turns and the current turn, each written "Speaker k: text" and joined by `</s>`, followed by the
  prompt "Now Speaker k feels `<mask>`". The input is cut from the left at 256 tokens, so the prompt is always
  kept. This is past context only, ECERC's task setting.
- **Classifier:** a linear head on the `<mask>` token's final hidden state.
- **Training:** cross-entropy; AdamW (lr 1e-5, weight decay 0.01); linear warmup over 10% of steps; batch 16;
  6 epochs; fp16; gradient clipping at 1.0. The epoch with the best dev macro-F1 is kept.
- **Seeds:** 1, 2 and 3. A seed whose best dev macro-F1 stays below 35 gets one retry, with seed + 1000.

## Fusion
- score = (1 − w)·log p_ECERC + w·log p_expert, with w ∈ {0.00, 0.05, …, 1.00}. w = 0 is ECERC alone.
- The pairs are the released checkpoints C1–C3 × the 3 expert seeds, giving 9 pairs.
- w\* is the value with the highest mean dev macro-F1 over pairs; ties go to the smaller w.

## Gate
The test runs only if the **cross-fitted** dev gain (5 dialogue folds, seed 12345) is at least +0.5 macro-F1.

## Freeze
w\*, the experts, the gate and the criteria are written to `frozen_config_expert.json` before any test access.

## Test
Run once.
- **Primary:** the mean over the 9 pairs of test macro-F1(fused at w\*) − macro-F1(ECERC alone).
- **95% CI:** a two-sample dialogue bootstrap that re-selects w in every replicate (B = 10,000, seed 20261012).
- **Secondary:** (w, τ) chosen jointly on dev, with a test-only CI. The full ensembles are reported descriptively.

## Verdict
The thresholds are the same as experiments 1 and 3. Consistency means Δ > 0 in at least 70% of pairs.

## Disclosure
The evaluated system is ECERC, unchanged, plus a second model that reads MELD's utterance text. That text is
part of the dataset and is the source of ECERC's own text features. Results will be reported as "ECERC +
text expert", never as a change to ECERC itself.

---

# Addendum (2026-10-12, fixed before running): Experiment 5, a fine-tuned LLM expert fused with ECERC

## Motivation
- Experiment 4 was INCONCLUSIVE: Fear improved, but macro-F1 rose only +0.27.
- Fine-tuned LLMs are the strongest single-model family reported on MELD (InstructERC: 69.15 weighted F1, against
  ECERC's 66.46).

## Expert
- **Model:** Qwen2.5-7B-Instruct (Apache-2.0).
- **Fine-tuning:** QLoRA (a 4-bit NF4 base with LoRA r = 16, α = 32, dropout 0.05 on every attention and MLP
  projection), plus a classification head.
- **Input:** an instruction, as many preceding turns ("Speaker k: text") as fit, and the target utterance. The
  input is capped at 256 tokens with the most recent turns kept. Past context only.
- **Training:** cross-entropy; AdamW (lr 1e-4); cosine schedule with 5% warmup; effective batch 16 (8 × 2);
  1 epoch; fp16.
- **Checkpoint:** dev macro-F1 is evaluated at 1/3, 2/3 and the end, and the best is kept. Training is capped at
  7 hours per seed.
- **Seeds:** 1 and 2, in parallel, one per T4. A seed whose best dev macro-F1 stays below 35 gets one retry, with
  seed + 1000. A run whose loss becomes non-finite more than 20 times stops with an error.

## Fusion, selection, gate, freeze and test
Identical to Experiment 4:
- score = (1 − w)·log p_ECERC + w·log p_LLM.
- 3 released checkpoints × 2 seeds = 6 pairs; w\* chosen on dev.
- The test runs only if the cross-fitted dev gain is at least +0.5.
- Everything is frozen before the test, and the test runs once.
- Consistency means Δ > 0 in at least 70% of pairs, i.e. at least 5 of 6.

## Multiplicity
This is the third test look, after experiments 1 and 4. The verdict uses the pre-registered 95% rules. A
Bonferroni-adjusted 98.33% interval is also reported, with a flag for whether its lower bound is above 0.

## Disclosure
- The result is reported as "ECERC + LLM expert"; ECERC itself is unchanged.
- The LLM's pretraining data may include transcripts of the TV series MELD comes from. This affects the absolute
  LLM scores, as it does for every published LLM result on MELD. It does not affect how the experiment was
  conducted (dev-only decisions, a single test pass).

---

# Addendum (2026-10-18, fixed before running): Experiment 6, self-supervised audio evidence

## Motivation
- **Runs 1–5:** ECERC's rare-emotion ceiling is set by its precomputed input evidence; only new evidence (experiment 4's
  text expert) moved a rare class on test.
- **The audio stream is the weakest input.**
  - It is 300-d openSMILE functionals: linear-probe dev macro-F1 22.4, with a +7.2-point train-side advantage from
    supervised feature selection (experiment 2).
  - The original loader routes it into the wrong model slots. The 300-d "audio" input holds the first 300 visual
    dimensions; the "visual" input holds the last 42 visual dimensions plus all of the audio.
- **Literature:** on MELD, replacing OpenSMILE with wav2vec features in a fixed architecture raised weighted F1 from
  66.4 to 68.4 (MiSTER-E, Table 3). Frozen Whisper-large features beat eGeMAPS by 12 weighted-F1 points as an
  audio-only classifier (MERBench).

## Audio
- **Source:** 16 kHz mono FLAC clips extracted from the official MELD videos, Hugging Face dataset `ajyy/MELD_audio`
  at revision `90175e52…` (GPL-3.0, like MELD). The three archives and three CSVs are SHA-256 checked (`MANIFEST.json`).
- **Alignment with the authors' feature file:**
  - The data-file dialogue key equals the MELD `Dialogue_ID` plus a per-split offset. The offset is the one shared by
    the dialogues whose normalised utterance text matches exactly; at least 95% must match.
  - Every dialogue must have the same number of utterances. Utterance i is the i-th CSV row by `Utterance_ID`.
  - Normalised text must agree for at least 99% of each split's utterances.
  - Train and dev CSV labels must equal the data-file labels.
  - The test CSV is parsed without its emotion column, so no test label is read.
- **Utterances without usable audio:** a missing clip, or text that disagrees with the data file. They get the training
  mean (0 after standardisation). At most 20 are allowed, or the run stops.
- **Clip length:** clips longer than 30 s are truncated to their first 30 s; clips shorter than 0.1 s are zero-padded.

## Features
All encoders are frozen (no MELD fine-tuning). They run identically on every split; this is label-free preprocessing,
like the authors' precomputed features.
- **Speech encoders:** Whisper-large-v3 encoder (`openai/whisper-large-v3` @`06f233fe…`) and WavLM-large
  (`microsoft/wavlm-large` @`c1423ed9…`). For each utterance and layer, the hidden states are averaged over the frames
  that cover real audio.
- **Candidates:** means over bands of 4 consecutive layers: Whisper 1–4 … 29–32 (8 bands) and WavLM 1–4 … 21–24
  (6 bands), 14 in all.
- **Event text (arm A3):** `BAAI/bge-large-en-v1.5` @`d4aa6901…`. Its [CLS] embedding, L2-normalised, of the utterance
  text with experiment 4's character clean-up; punctuation kept; at most 128 tokens.
- **Standardisation:** the new features are standardised with the training utterances' mean and SD.

## Selection (training split only)
- The feature-leak check's probe: standardised features, logistic regression, C = 1, max_iter = 1000.
- 5 folds of training dialogues, fold seed 2026.
- The candidate with the highest out-of-fold macro-F1 is selected; ties go to the earlier candidate in the list above.
- Dev probe scores are reported, not used. Probes on openSMILE, `sroberta1` and bge are reported for reference.

## Arms
ECERC's architecture and the original recipe are unchanged in every arm: `Loss(gamma = 1)`, Adam lr 1e-5, weight
decay 2e-4, batch 32, 40 epochs.

| Arm | Inputs |
|---|---|
| A0 | Original features and slot routing (= Path B's C0) |
| A1 | Original features; audio and visual routed to the slots the model expects |
| A2 | A1 with openSMILE replaced by the selected speech features |
| A3 | A2 with the event input `sroberta1` replaced by bge-large-en-v1.5 |

## Stage 1: dev screen
- 5 seeds (101 to 105) per arm. Each run keeps its best-dev-macro-F1 epoch (earliest on ties).
- **Winner:** the highest mean dev macro-F1 among A1 to A3.
- **Gate:** Stage 2 runs only if the winner beats A0 by at least 0.5 dev macro-F1.
- **Fidelity (reported, not decisive):** A0 seeds 101 to 103 against Path B's C0 (55.12, 54.04, 53.21).

## Freeze
The selected features (with the SHA-256 of the final feature arrays), winner, seeds, checkpoint rule and criteria are
written to `frozen_config_audio.json` before any test access.

## Stage 2: confirmatory
- Seeds 201 to 220, × {A0, winner}. Same checkpoint rule.
- The test set is evaluated once, after all training.

## Statistics
- **Primary:** mean paired difference in test macro-F1 (winner − A0) across the 20 seeds.
- **95% CI:** hierarchical bootstrap that resamples seeds and test dialogues (B = 10,000, seed 20261018).
- **Also reported:** seed-level t-intervals, and the number of seeds with a positive difference, for macro-F1, weighted
  F1, accuracy, Fear F1 and Disgust F1.

## Verdict
The thresholds are those of experiments 1 and 3 to 5. Consistency means the difference is > 0 in at least 70% of seeds
(14 of 20).

## Multiplicity
This would be the third test look, after experiments 1 and 4. The verdict uses the pre-registered 95% rule. A
Bonferroni-adjusted 98.33% interval is also reported, with a flag for whether its lower bound is above 0.

## Disclosure
- The result is reported as "ECERC with self-supervised audio evidence" (with a new event encoder if A3 wins). The
  architecture and objective are unchanged; the input features differ.
- Whisper's speech-recognition training data and the pretraining data of WavLM and bge are public web corpora that may
  include Friends audio or transcripts, but no emotion labels. emotion2vec was not used, because its pretraining
  included MELD audio.
- The clips come from a third-party extraction of the official MELD videos; their alignment with the authors' feature
  file is checked as described above.

---

# Addendum (2026-10-19, fixed before running): Experiment 7, a zero-shot LLM fused with ECERC

## Motivation
- **Experiments 4 and 5:** experts fine-tuned on MELD inherit its label skew. Experiment 5's LLM scored 22–23 dev
  Disgust F1, and fusion traded one class for another.
- **Experiment 6:** better non-text evidence shifted the boundaries between related emotions (Sadness up, Joy down)
  without separating them.
- **A zero-shot LLM is never trained on MELD's labels.** Published zero-shot results on the MELD test set
  (arXiv 2609.05806) show the error profile that ECERC lacks:
  - LLaMA-70B: Fear 33.3, Disgust 39.8, Sadness 47.3.
  - Qwen-32B: Fear 26.8, Disgust 38.9, Sadness 47.1.
  - ECERC: 26.1, 31.4 and 41.0.
  - Both LLMs are weaker than ECERC on Neutral and Joy.
- **Dev-only pilot** (2026-10-19, `results/pilot_local_zeroshot_3b/`; the test split was not read). Zero-shot
  Qwen2.5-3B-Instruct with the prompt below, against the released checkpoints' saved dev predictions:
  - The LLM alone scored 35.6 dev macro-F1.
  - Plain fusion had a cross-fitted dev gain of −0.91.
  - After label-free prior matching (below), the cross-fitted gain was +0.76.
  - The pilot fixed the prompt and the calibration step. Because it used dev, the dev gate below is not independent
    of it.

## LLM
- **Model:** `unsloth/Qwen2.5-32B-Instruct-bnb-4bit` at revision `aa79e347…`. It is a pre-quantized 4-bit (NF4) copy of
  `Qwen/Qwen2.5-32B-Instruct` (Apache-2.0), loaded frozen across two T4 GPUs.
- **Input:** one chat per utterance.
  - System message: "You are an expert annotator of emotions in TV-show dialogue."
  - User message: "Here is a dialogue, in order:", then every turn up to and including the target as "Speaker k: text"
    (generic speaker labels, Experiment 4's text clean-up), then "Which emotion does Speaker k express in the last
    line? Answer with one word: neutral, surprise, fear, sadness, joy, disgust, or anger."
  - The model's chat template is applied, with the generation prompt.
  - Past context only, ECERC's task setting.
- **Scoring:**
  - For each class, take the next-token log-probabilities of the first token of its label word, written four ways
    (lower or capitalised, with or without a leading space), and combine them by log-sum-exp.
  - Renormalise over the seven classes.
  - The first tokens must be distinct across classes, or the run stops.
  - The key/value cache of the shared prefix is reused across turns; cached and full computation agree to about
    1e-7.
- **Calibration (label-free prior matching):**
  - p̃(c) ∝ p(c) / q(c) · π_train(c), where q is the LLM's mean prediction over the dev inputs.
  - q is frozen before the test and applied unchanged to the test inputs.

## Fusion, selection and gate
These are identical to Experiment 4.
- **Pairs:** the three released checkpoints C1–C3, each with the one LLM.
- **Score:** (1 − w)·log p_ECERC + w·log p̃_LLM, with w ∈ {0.00, 0.05, …, 1.00}.
- **Selection:** w\* has the highest mean dev macro-F1 over the three pairs; ties go to the smaller w.
- **Gate:** the test runs only if the cross-fitted dev gain (5 dialogue folds, seed 12345) is at least +0.5.

## Freeze
q, w\*, the model revision, the label token ids, the prompt's SHA-256 and the criteria are written to
`frozen_config_zeroshot.json` before any test access.

## Test
Run once.
- **Primary:** the mean over the three pairs of test macro-F1(fused at w\*) − macro-F1(ECERC alone).
- **95% CI:** a two-sample dialogue bootstrap that re-selects w in every replicate (B = 10,000, seed 20261019).
- **Secondary, descriptive:**
  - fusion without calibration at its own dev-chosen w, with a test-only CI;
  - the LLM alone;
  - weighted F1, accuracy and per-class changes.

## Verdict
The thresholds are those of experiments 1 and 3 to 6. Consistency means Δ > 0 in at least 70% of pairs, which for
three pairs means all three.

## Multiplicity
This would be the third test look, after experiments 1 and 4. The verdict uses the pre-registered 95% rule. A
Bonferroni-adjusted 98.33% interval is also reported, with a flag for whether its lower bound is above 0.

## Disclosure
- The result is reported as "ECERC + zero-shot LLM"; ECERC itself is unchanged.
- The LLM's pretraining data may include Friends transcripts or the public MELD files with their labels; this cannot be
  excluded. Generic speaker labels avoid giving character names.
- The pilot used a 3B model on an Apple laptop (MLX). Experiment 7 uses a 32B model on Kaggle (transformers,
  bitsandbytes), so its LLM numbers are not expected to match the pilot's.
