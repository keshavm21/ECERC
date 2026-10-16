#!/usr/bin/env python3
"""Local test of the Experiment 4 notebook on synthetic data (CPU, no downloads).

A tiny randomly initialised RoBERTa and a byte-level BPE tokenizer are built offline and saved with
save_pretrained, so the notebook's real from_pretrained code path runs unchanged. Scenario 1 forces the gate
open (every stage runs, including the test); scenario 2 forces it shut (the test must be skipped). In both,
no test-split label may be read before the test stage.

    research/limitation2/.venv/bin/python research/limitation2/tests/smoke_test_expert.py
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
SMOKE = os.path.join(HERE, '_smoke_expert')
NOTEBOOK = os.path.join(ROOT, 'notebooks', 'ECERC_L2_expert_fusion.ipynb')
CKPT = os.path.join(REPO, 'MELD', 'ECERC_MODEL.pkl')


def tiny_roberta(path):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast
    os.makedirs(path)
    corpus = [f'Speaker {k}: utterance {v}-{i} Now Speaker {k} feels' for k in range(1, 10) for v in range(50) for i in range(5)]
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(corpus, vocab_size=400, min_frequency=1,
                            special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])
    bpe.save_model(path)
    tok = RobertaTokenizerFast(vocab_file=os.path.join(path, 'vocab.json'), merges_file=os.path.join(path, 'merges.txt'))
    tok.save_pretrained(path)
    cfg = RobertaConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=130, type_vocab_size=1,
                        pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    RobertaModel(cfg).save_pretrained(path)


def run(case, gate, model_dir, floor=0.0):
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
                     expert_model=model_dir, expert_seeds=[1, 2], expert_epochs=2, expert_batch=8, expert_max_len=64,
                     gate_min_gain=gate, power_B=200, bootstrap_B=400, expert_min_dev_mf1=floor)
    with open(NOTEBOOK) as f:
        cells = [''.join(c['source']) for c in json.load(f)['cells'] if c['cell_type'] == 'code']
    ns = {'__name__': '__smoke__'}
    LoggingDict.accessed = set()
    for i, src in enumerate(cells):
        if 'LE.stage_test(S)' in src:
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
    out = os.path.join(overrides['work_dir'], 'outputs_expert')
    for name in ('SUMMARY_expert.md', 'frozen_config_expert.json', 'dev_selection.json', 'design.json', 'checks.json',
                 'expert_training.json'):
        assert os.path.exists(os.path.join(out, name)), f'missing {name}'
    with open(os.path.join(out, 'dev_selection.json')) as f:
        sel = json.load(f)
    print(f'\n[smoke] {case}: w* {sel["w_star"]}, cross-fitted gain {sel["forecast"]["crossfit_delta_macro_f1"]:+.2f}, '
          f'proceed {sel["proceed"]}')
    if gate < 0:
        assert sel['proceed'] and (LoggingDict.accessed & test_ids), 'the test stage did not run'
        with open(os.path.join(out, 'test_results_expert.json')) as f:
            res = json.load(f)
        print(f'[smoke] {case}: verdict on synthetic data (meaningless): {res["verdict"]}')
    else:
        assert not sel['proceed'] and not (LoggingDict.accessed & test_ids)
        assert not os.path.exists(os.path.join(out, 'test_results_expert.json'))
        with open(os.path.join(out, 'expert_training.json')) as f:
            assert all(e['retried_after'] is not None for e in json.load(f)), 'retry rule not applied'
        print(f'[smoke] {case}: the test was skipped as required, and every collapsed seed was retried once')


def main():
    if os.path.exists(SMOKE):
        shutil.rmtree(SMOKE)
    model_dir = os.path.join(SMOKE, 'tiny_roberta')
    tiny_roberta(model_dir)
    run('gate_open', -100.0, model_dir)
    run('gate_shut', 100.0, model_dir, floor=100.0)  # also exercises the retry rule
    print('\n[smoke] PASSED')


if __name__ == '__main__':
    sys.exit(main())
