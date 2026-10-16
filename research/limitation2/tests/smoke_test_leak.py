#!/usr/bin/env python3
"""Local test of the feature-leak notebook on synthetic data (CPU, no downloads).

Runs the notebook twice: once with a planted label leak in the training-set emotion features (the verdict
must be LEAK CONFIRMED) and once without (NOT CONFIRMED). Also checks that no test-split entry is ever read.

    research/limitation2/.venv/bin/python research/limitation2/tests/smoke_test_leak.py
"""
import json
import os
import pickle
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from smoke_test import LoggingDict, make_synthetic  # noqa: E402

ROOT = os.path.dirname(HERE)
SMOKE = os.path.join(HERE, '_smoke_leak')
NOTEBOOK = os.path.join(ROOT, 'notebooks', 'ECERC_L2_feature_leak_check.ipynb')


def plant_leak(path):
    """Make the training-set emotion features encode the label strongly; dev keeps the original weak signal."""
    with open(path, 'rb') as f:
        data = list(pickle.load(f))
    labels, eroberta1, train_ids = data[1], data[3], data[14]
    for vid in train_ids:
        for i, y in enumerate(labels[vid]):
            eroberta1[vid][i][:7] += 6.0 * np.eye(7, dtype=np.float32)[y]
    with open(path, 'wb') as f:
        pickle.dump(tuple(data), f, protocol=2)


def run(case, leak):
    work = os.path.join(SMOKE, case)
    os.makedirs(work)
    os.chdir(work)
    data_path = os.path.join(work, 'synthetic_meld.pkl')
    splits = make_synthetic(data_path)
    if leak:
        plant_leak(data_path)
    overrides = dict(work_dir=os.path.join(work, 'work'), offline=True, data_local_path=data_path,
                     expected_data_sha256=None, expected_utterances=None, bootstrap_B=300)
    with open(NOTEBOOK) as f:
        cells = [''.join(c['source']) for c in json.load(f)['cells'] if c['cell_type'] == 'code']
    ns = {'__name__': '__smoke__'}
    LoggingDict.accessed = set()
    for i, src in enumerate(cells):
        exec(compile(src, f'<{case} cell {i}>', 'exec'), ns)
        if i == 0:
            ns['CONFIG'].update(overrides)
        if 'stage_setup' in src:
            P = ns['l2_posthoc']
            original = P.load_pickle

            def logged_load(path, trust=False, _orig=original):
                obj = list(_orig(path, trust))
                obj[1] = LoggingDict(obj[1])   # emotion_labels
                obj[3] = LoggingDict(obj[3])   # eroberta1
                return tuple(obj)

            P.load_pickle = logged_load
    leaked = LoggingDict.accessed & set(splits['test'])
    assert not leaked, f'test split read: {sorted(leaked)[:5]}'
    out = os.path.join(overrides['work_dir'], 'outputs_leak')
    for name in ('SUMMARY_leak.md', 'leak_results.json', 'rule.json', 'checks.json', 'tables/leak_probe_table.csv',
                 'figures/leak_probe.png'):
        assert os.path.exists(os.path.join(out, name)), f'missing {name}'
    with open(os.path.join(out, 'leak_results.json')) as f:
        res = json.load(f)
    expected = 'LEAK CONFIRMED' if leak else 'NOT CONFIRMED'
    print(f'\n[smoke] {case}: verdict {res["verdict"]} (expected {expected}); gap eroberta1 {res["gap_primary"]:+.1f}, '
          f'sroberta1 {res["gap_control"]:+.1f}, difference CI {res["difference_ci95"]}')
    assert res['verdict'] == expected
    return res


def main():
    if os.path.exists(SMOKE):
        shutil.rmtree(SMOKE)
    run('planted_leak', leak=True)
    run('no_leak', leak=False)
    print('\n[smoke] PASSED: verdicts correct in both cases and the test split was never read')


if __name__ == '__main__':
    sys.exit(main())
