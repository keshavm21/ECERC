# ECERC Limitation 2, Experiment 8a: zero-shot LLM soft labels for the training split

Run 2026-10-19 14:22 | cuda [('Tesla T4', 14.6), ('Tesla T4', 14.6)] | torch 2.11.0+cu128 | transformers 5.16.1 | LLM unsloth/Qwen2.5-32B-Instruct-bnb-4bit | data sha256 a0349614b2b58e8b...

- dev re-scoring (20 dialogues, 229 utterances): argmax agreement with Experiment 7 100.0%, max |diff| 0.0000
- training split: 9989 utterances, 526792 new tokens, 445 minutes, label-word probability mass 0.998 on average
- `dumps/train_llm.npz` SHA-256: `8c810de775b0e5a6cbac1c4f4a56aed8648d55469e7e709fc19d5147c1d586ae`
- the test split was not scored and no test label was read

Checks: 17 passed, 0 warnings
