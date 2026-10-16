#!/usr/bin/env python3
"""Local test of the Path B notebook on synthetic data (CPU, no downloads, tiny epoch counts).

Scenario 1 forces the gate open, so every stage runs including the test evaluation. Scenario 2 forces it shut,
so stage 2 and the test must be skipped. In both, no test-split label may be read before the test stage.

    research/limitation2/.venv/bin/python research/limitation2/tests/smoke_test_pathb.py
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from smoke_test import LoggingDict, make_synthetic  # noqa: E402

ROOT = os.path.dirname(HERE)
SMOKE = os.path.join(HERE, '_smoke_pathb')
NOTEBOOK = os.path.join(ROOT, 'notebooks', 'ECERC_L2_pathB_representation.ipynb')


def run(case, gate, epochs):
    work = os.path.join(SMOKE, case)
    os.makedirs(work)
    os.chdir(work)
    data_path = os.path.join(work, 'synthetic_meld.pkl')
    splits = make_synthetic(data_path)
    test_ids = set(splits['test'])
    overrides = dict(work_dir=os.path.join(work, 'work'), offline=True, data_local_path=data_path,
                     expected_data_sha256=None, expected_split_utterances=None, expected_split_dialogues=None,
                     epochs=epochs, stage1_seeds=[1, 2], stage2_seeds=[11, 12, 13], gate_min_gain=gate,
                     bootstrap_B=200)
    with open(NOTEBOOK) as f:
        cells = [''.join(c['source']) for c in json.load(f)['cells'] if c['cell_type'] == 'code']
    ns = {'__name__': '__smoke__'}
    LoggingDict.accessed = set()
    for i, src in enumerate(cells):
        if 'LB.stage_test(S)' in src:
            leaked = LoggingDict.accessed & test_ids
            assert not leaked, f'test labels read before the test stage: {sorted(leaked)[:5]}'
            print(f'  [smoke] OK: no test label read before the test stage')
        exec(compile(src, f'<{case} cell {i}>', 'exec'), ns)
        if i == 0:
            ns['CONFIG'].update(overrides)
        if 'stage_setup' in src:
            P = ns['l2_posthoc']
            original = P.load_pickle

            def logged_load(path, trust=False, _orig=original):
                obj = list(_orig(path, trust))
                obj[1] = LoggingDict(obj[1])  # emotion_labels
                obj[2] = LoggingDict(obj[2])  # sentiment_labels
                return tuple(obj)

            P.load_pickle = logged_load
    out = os.path.join(overrides['work_dir'], 'outputs_pathb')
    for name in ('SUMMARY_pathb.md', 'frozen_config_pathb.json', 'stage1_summary.json', 'design.json', 'checks.json',
                 'tables/stage1_runs.csv'):
        assert os.path.exists(os.path.join(out, name)), f'missing {name}'
    with open(os.path.join(out, 'stage1_summary.json')) as f:
        st1 = json.load(f)
    print(f'\n[smoke] {case}: stage-1 winner {st1["winner"]["name"]}, proceed {st1["proceed_to_stage2"]}')
    if gate < 0:
        assert st1['proceed_to_stage2']
        assert LoggingDict.accessed & test_ids, 'the test stage never read test labels'
        for name in ('test_results_pathb.json', 'tables/stage2_test_per_seed.csv', 'tables/stage2_dev_runs.csv'):
            assert os.path.exists(os.path.join(out, name)), f'missing {name}'
        with open(os.path.join(out, 'test_results_pathb.json')) as f:
            res = json.load(f)
        print(f'[smoke] {case}: verdict on synthetic data (meaningless): {res["verdict"]}')
    else:
        assert not st1['proceed_to_stage2']
        assert not (LoggingDict.accessed & test_ids), 'test labels read although stage 2 was skipped'
        assert not os.path.exists(os.path.join(out, 'test_results_pathb.json'))
        print(f'[smoke] {case}: stage 2 and the test were skipped as required')


def main():
    if os.path.exists(SMOKE):
        shutil.rmtree(SMOKE)
    run('gate_open', gate=-100.0, epochs=2)
    run('gate_shut', gate=100.0, epochs=1)
    print('\n[smoke] PASSED')


if __name__ == '__main__':
    sys.exit(main())
