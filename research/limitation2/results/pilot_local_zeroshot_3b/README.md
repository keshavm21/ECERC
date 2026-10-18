# Dev-only pilot for Experiment 7: a zero-shot 3B LLM next to ECERC

Run on 2026-10-19, on an 8 GB Apple M1 laptop with MLX. **Dev split only**: the test split was never read.

The pilot asked a single question before Experiment 7 was designed: does a zero-shot LLM, which has never been
trained on MELD's labels, carry dev evidence that ECERC lacks?

- **Model:** `mlx-community/Qwen2.5-3B-Instruct-4bit` at revision `4f83f8f1`.
  - A 7B model was tried first but did not fit in this laptop's memory, so it was stopped.
- **Prompt:** written before any dev result was seen.
  - Input: past context only, as "Speaker k: text" lines with generic speaker labels.
  - Scoring: next-token probabilities of the seven label words.
  - Experiment 7 uses the same prompt unchanged.
- **Comparison:** the three released ECERC checkpoints' saved dev predictions (`../run_colab_1/outputs/dumps/dev_C*.npz`),
  combined with the LLM by Experiment 4's own fusion and cross-fitting functions.

## Result (from `pilot_output.txt`)

| Dev | Macro-F1 | Fear F1 | Disgust F1 |
|---|---|---|---|
| Zero-shot LLM alone | 35.58 | 9.3 | 10.3 |
| Zero-shot LLM, prior-matched (label-free) | 39.32 | 20.8 | 14.3 |

How fusion with ECERC changes macro-F1 (released checkpoints, mean dev macro-F1 54.04):

| Fusion | Cross-fitted dev gain |
|---|---|
| Plain fusion | −0.91 |
| Fusion with the prior-matched LLM | +0.76 |

The 3B model alone is weak, especially on Fear and Disgust. After label-free prior matching, it still adds a small
amount out of fold. This motivated Experiment 7's two choices:
- a much larger zero-shot model (Qwen2.5-32B-Instruct);
- the calibration step, fixed in advance.

Because the pilot used dev, Experiment 7's dev gate is not independent of it; `PROTOCOL.md` discloses this.

## Files

- `pilot_zeroshot_dev.py`: the script as run.

  ```
  python -I pilot_zeroshot_dev.py SRC PKL DUMPS OUT MODEL REVISION
  ```

  It needs `mlx-lm` (Apple silicon), `numpy` and `torch`.
- `pilot_output.txt`: its output, with download messages removed.
- `dev_llm_logp_Qwen2.5-3B-Instruct-4bit.npz`: the LLM's dev log-probabilities. The array has shape 1109 × 7, in
  the data file's dev order.
