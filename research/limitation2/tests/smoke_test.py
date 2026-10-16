#!/usr/bin/env python3
"""Local end-to-end test of the notebook on synthetic data (CPU, no downloads).

Executes the notebook's code cells in order with a few config overrides, using the in-repo
MELD/ECERC_MODEL.pkl as all three checkpoints and a small synthetic pickle with the real data layout.
It also verifies that no test label is read before the freeze and that the test stage refuses to run
without a freeze. Synthetic numbers are meaningless; only the pipeline is being tested.

    research/limitation2/.venv/bin/python research/limitation2/tests/smoke_test.py
"""
import hashlib
import json
import os
import pickle
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(ROOT))
SMOKE = os.path.join(HERE, '_smoke')
NOTEBOOK = os.path.join(ROOT, 'notebooks', 'ECERC_L2_posthoc_LA.ipynb')
CKPT = os.path.join(REPO, 'MELD', 'ECERC_MODEL.pkl')


class LoggingDict(dict):
    """Records every key read, to prove test labels stay untouched until the freeze."""
    accessed = set()

    def __getitem__(self, key):
        LoggingDict.accessed.add(key)
        return super().__getitem__(key)


def make_synthetic(path, seed=0):
    rng = np.random.default_rng(seed)
    probs = np.array([0.47, 0.12, 0.03, 0.07, 0.17, 0.03, 0.11])
    splits = dict(train=list(range(0, 70)), valid=list(range(1000, 1030)), test=list(range(2000, 2040)))
    speakers, emo, sent, sentences, audio, visual = {}, {}, {}, {}, {}, {}
    rob = {f'{p}roberta{i}': {} for p in 'es' for i in range(1, 5)}
    for ids in splits.values():
        for k, vid in enumerate(ids):
            n = int(rng.integers(8, 13)) if k == 0 else int(rng.integers(2, 13))
            y = rng.choice(7, size=n, p=probs)
            if k == 0:
                y[:7] = np.arange(7)  # every class occurs in every split
            spk = rng.integers(0, 3, size=n)
            speakers[vid] = [[1 if j == s else 0 for j in range(9)] for s in spk]
            emo[vid] = [int(v) for v in y]
            sent[vid] = [int(rng.integers(0, 3)) for _ in range(n)]
            sentences[vid] = [f'utterance {vid}-{i}' for i in range(n)]
            for name in rob:
                rob[name][vid] = [rng.normal(size=1024).astype(np.float32) + 0.3 * y[i] for i in range(n)]
            audio[vid] = [rng.normal(size=300).astype(np.float32) for _ in range(n)]
            visual[vid] = [rng.normal(size=342).astype(np.float32) for _ in range(n)]
    data = (speakers, emo, sent, rob['eroberta1'], rob['eroberta2'], rob['eroberta3'], rob['eroberta4'],
            rob['sroberta1'], rob['sroberta2'], rob['sroberta3'], rob['sroberta4'], audio, visual, sentences,
            splits['train'], splits['test'], splits['valid'])
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=2)
    return splits


def main():
    if os.path.exists(SMOKE):
        shutil.rmtree(SMOKE)
    os.makedirs(SMOKE)
    os.chdir(SMOKE)
    data_path = os.path.join(SMOKE, 'synthetic_meld.pkl')
    splits = make_synthetic(data_path)
    with open(CKPT, 'rb') as f:
        ck_sha = hashlib.sha256(f.read()).hexdigest()
    overrides = dict(work_dir=os.path.join(SMOKE, 'work'), offline=True, data_local_path=data_path,
                     checkpoints=[dict(name=n, local_path=CKPT, sha256=ck_sha) for n in ('C1', 'C2', 'C3')],
                     expected_split_utterances=None, expected_split_dialogues=None, min_dev_acc=0.0,
                     bootstrap_B=300, power_B=200)
    with open(NOTEBOOK) as f:
        cells = [''.join(c['source']) for c in json.load(f)['cells'] if c['cell_type'] == 'code']
    ns = {'__name__': '__smoke__'}
    test_ids = set(splits['test'])
    for i, src in enumerate(cells):
        if 'L.stage_freeze(S)' in src:
            L = ns['L']
            leaked = LoggingDict.accessed & test_ids
            assert not leaked, f'test labels read before the freeze: {sorted(leaked)[:5]}'
            try:
                L.stage_test(ns['S'])
                raise AssertionError('stage_test ran without a freeze')
            except L.CheckFailed:
                print('  [smoke] OK: test stage refuses to run before the freeze; no test label read so far')
        print(f'\n[smoke] ---- code cell {i} ----')
        exec(compile(src, f'<cell {i}>', 'exec'), ns)
        if i == 0:
            ns['CONFIG'].update(overrides)
        if 'stage_setup' in src:  # wrap the loader so every label lookup is recorded
            L = ns['L']
            original = L.load_pickle

            def logged_load(path, trust=False, _orig=original):
                obj = list(_orig(path, trust))
                obj[1] = LoggingDict(obj[1])
                return tuple(obj)

            L.load_pickle = logged_load
    assert LoggingDict.accessed & test_ids, 'test labels were never read, so the test stage did not run'
    out = os.path.join(overrides['work_dir'], 'outputs')
    for name in ('frozen_config.json', 'test_results.json', 'SUMMARY.md', 'dev_diagnostics.json', 'checks.json',
                 'tables/test_arms_per_checkpoint.csv', 'figures/dev_frontier.png'):
        assert os.path.exists(os.path.join(out, name)), f'missing output {name}'
    assert os.path.exists(os.path.join(overrides['work_dir'], 'ecerc_l2_results.zip'))
    with open(os.path.join(out, 'test_results.json')) as f:
        res = json.load(f)
    print('\n[smoke] PASSED. verdict on synthetic data (meaningless):', res['verdict'])


if __name__ == '__main__':
    sys.exit(main())
