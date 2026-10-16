#!/usr/bin/env python3
"""Local test of the Experiment 5 notebook on synthetic data (CPU, no downloads, no quantisation).

A tiny randomly initialised Qwen2 classifier and a byte-level BPE tokenizer are built offline, so the real
code path runs: LoRA fine-tuning in worker subprocesses, dev selection, the gate, the freeze and the test.
Scenario 1 forces the gate open; scenario 2 forces it shut and also exercises the collapsed-seed retry.
In both, no test-split label may be read before the test stage.

    research/limitation2/.venv/bin/python research/limitation2/tests/smoke_test_llm.py
"""
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from smoke_test import LoggingDict, make_synthetic  # noqa: E402

ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(ROOT))
SMOKE = os.path.join(HERE, '_smoke_llm')
NOTEBOOK = os.path.join(ROOT, 'notebooks', 'ECERC_L2_llm_fusion.ipynb')
CKPT = os.path.join(REPO, 'MELD', 'ECERC_MODEL.pkl')


def tiny_qwen(path):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForSequenceClassification
    os.makedirs(path)
    corpus = ([f'Speaker {k}: utterance {v}-{i}\n' for k in range(1, 10) for v in range(40) for i in range(5)]
              + ['Task: identify the emotion that the target speaker expresses in the last utterance of this '
                 'conversation from a TV show. Answer with one of: neutral, surprise, fear, sadness, joy, disgust, '
                 'anger. Conversation: Target utterance Emotion:'] * 20)
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(corpus, vocab_size=500, min_frequency=1, special_tokens=['<pad>', '<unk>', '<eos>'])
    bpe.save(os.path.join(path, 'tokenizer.json'))
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(path, 'tokenizer.json'), pad_token='<pad>',
                                  unk_token='<unk>', eos_token='<eos>')
    tok.save_pretrained(path)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024, num_labels=7,
                      pad_token_id=tok.pad_token_id)
    Qwen2ForSequenceClassification(cfg).save_pretrained(path)


def run(case, gate, model_dir, floor):
    work = os.path.join(SMOKE, case)
    os.makedirs(work)
    os.chdir(work)
    data_path = os.path.join(work, 'synthetic_meld.pkl')
    splits = make_synthetic(data_path)
    test_ids = set(splits['test'])
    with open(CKPT, 'rb') as f:
        ck_sha = hashlib.sha256(f.read()).hexdigest()
    overrides = dict(work_dir=os.path.join(work, 'work'), offline=True, data_local_path=data_path,
                     checkpoints=[dict(name=n, local_path=CKPT, sha256=ck_sha) for n in ('C1', 'C2', 'C3')],
                     expected_split_utterances=None, expected_split_dialogues=None, min_dev_acc=0.0,
                     llm_model=model_dir, llm_quantize=False, llm_seeds=[1, 2], llm_epochs=1, llm_micro_batch=4,
                     llm_grad_accum=2, llm_eval_every=0.5, llm_eval_batch=8, llm_max_len=128, llm_min_dev_mf1=floor,
                     gate_min_gain=gate, power_B=200, bootstrap_B=400, poll_seconds=1)
    with open(NOTEBOOK) as f:
        cells = [''.join(c['source']) for c in json.load(f)['cells'] if c['cell_type'] == 'code']
    ns = {'__name__': '__smoke__'}
    LoggingDict.accessed = set()
    for i, src in enumerate(cells):
        if 'LL.stage_test(S)' in src:
            leaked = LoggingDict.accessed & test_ids
            assert not leaked, f'test labels read before the test stage: {sorted(leaked)[:5]}'
            print('  [smoke] OK: no test label read before the test stage')
        exec(compile(src, f'<{case} cell {i}>', 'exec'), ns)
        if i == 0:
            ns['CONFIG'].update(overrides)
        if 'stage_setup' in src:
            P = ns['l2_posthoc']
            original = P.load_pickle

            def logged_load(path, trust=False, _orig=original):
                obj = list(_orig(path, trust))
                obj[1] = LoggingDict(obj[1])  # emotion_labels
                return tuple(obj)

            P.load_pickle = logged_load
    out = os.path.join(overrides['work_dir'], 'outputs_llm')
    for name in ('SUMMARY_llm.md', 'frozen_config_llm.json', 'dev_selection.json', 'design.json', 'checks.json',
                 'llm_training.json'):
        assert os.path.exists(os.path.join(out, name)), f'missing {name}'
    with open(os.path.join(out, 'dev_selection.json')) as f:
        sel = json.load(f)
    with open(os.path.join(out, 'llm_training.json')) as f:
        runs = json.load(f)
    print(f'\n[smoke] {case}: w* {sel["w_star"]}, cross-fitted gain {sel["forecast"]["crossfit_delta_macro_f1"]:+.2f}, '
          f'proceed {sel["proceed"]}, seeds {[r["seed"] for r in runs]}')
    if gate < 0:
        assert sel['proceed'] and (LoggingDict.accessed & test_ids), 'the test stage did not run'
        with open(os.path.join(out, 'test_results_llm.json')) as f:
            res = json.load(f)
        print(f'[smoke] {case}: verdict on synthetic data (meaningless): {res["verdict"]}; adjusted CI '
              f'{res["ci_multiplicity_adjusted"]["interval"]}')
    else:
        assert not sel['proceed'] and not (LoggingDict.accessed & test_ids)
        assert not os.path.exists(os.path.join(out, 'test_results_llm.json'))
        assert all(r['retried_after'] is not None and r['seed'] > 1000 for r in runs), 'retry rule not applied'
        print(f'[smoke] {case}: the test was skipped as required, and every collapsed seed was retried once')


def main():
    if os.path.exists(SMOKE):
        shutil.rmtree(SMOKE)
    model_dir = os.path.join(SMOKE, 'tiny_qwen')
    tiny_qwen(model_dir)
    run('gate_open', -100.0, model_dir, floor=0.0)
    run('gate_shut', 100.0, model_dir, floor=100.0)
    print('\n[smoke] PASSED')


if __name__ == '__main__':
    sys.exit(main())
