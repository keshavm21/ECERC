# Results

One results zip per experiment, exactly as downloaded from Colab or Kaggle, plus its extracted copy in a
`run_*` folder. [`FINDINGS.md`](FINDINGS.md) interprets every run; each run's `SUMMARY*.md` holds the verdict
and headline numbers as printed by the notebook.

| Experiment | Zip | Extracted run | Platform, date |
|---|---|---|---|
| 1. Post-hoc logit adjustment | `ecerc_l2_results.zip` | `run_colab_1/outputs/` | Colab T4, 2026-10-11 |
| 2. Feature-leak check | `ecerc_l2_leak_results.zip` | `run_colab_leak_1/outputs_leak/` | Colab CPU, 2026-10-11 |
| 3. Path B (representation changes) | `ecerc_l2_pathb_results.zip` | `run_kaggle_pathb_1/outputs_pathb/` | Kaggle T4, 2026-10-11 |
| 4. RoBERTa text expert fusion | `ecerc_l2_expert_results.zip` | `run_kaggle_expert_1/outputs_expert/` | Kaggle T4, 2026-10-11 |
| 5. LLM expert fusion | `ecerc_l2_llm_results.zip` | `run_kaggle_llm_1/outputs_llm/` | Kaggle 2× T4, 2026-10-12 |

## What a run folder contains

Not every file exists in every run; experiments 2, 3 and 5 never reached the test stage.

- `SUMMARY*.md`: the verdict and key numbers.
- `frozen_config*.json`: every decision fixed before any test access (experiments 1, 3, 4 and 5). In
  experiments 3 and 5 the dev gate then stopped the run, so the test was never evaluated.
- `checks.json`: the integrity checks and their outcome; `environment.json`: software versions;
  `run_log.jsonl`: a timestamped log. All runs except experiment 2 also have `data_gate.json` (data checks) and
  `provenance.json` (file hashes); experiments 3 to 5 have `design.json`.
- Experiment-specific outputs:
  - experiment 1: `test_results.json`, `TEST_EVALUATED.json`, `dev_selection.json`, `dev_diagnostics.json`,
    `fidelity.json`;
  - experiment 2: `leak_results.json`, `rule.json`;
  - experiment 3: `stage1_summary.json`;
  - experiment 4: `expert_training.json`, `dev_selection.json`, `test_results_expert.json`;
  - experiment 5: `llm_training.json`, `dev_selection.json`, `logs/` (per-seed training logs).
- `tables/` (CSV), `figures/` (PNG) and `dumps/` (per-model log-probabilities as `.npz`) where produced.

Model weights are not saved.
