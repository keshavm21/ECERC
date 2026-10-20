#!/usr/bin/env python3
"""Local test of the two Experiment 8 notebooks on synthetic data (CPU, no downloads, tiny models).

Notebook 8a runs with a tiny randomly initialised Qwen2 model (built as in the Experiment 7 smoke test) on a synthetic
feature file: first without a reference (the dev re-scoring check is a warning), then with the first run's dev output
as the reference (the check must pass). 8a must never read a test label. Notebook 8b then trains on 8a's real output:
scenario 1 forces the gate open (every stage runs, including the test), scenario 2 forces it shut (the test must be
skipped), scenario 3 gives a wrong teacher SHA-256 (it must stop before training). In 8b, no test label may be read
before the test stage.

    research/limitation2/.venv/bin/python research/limitation2/tests/smoke_test_distill.py
"""
import hashlib
import json
import os
import pickle
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from smoke_test import LoggingDict, make_synthetic  # noqa: E402
from smoke_test_zeroshot import tiny_qwen  # noqa: E402

ROOT = os.path.dirname(HERE)
SMOKE = os.path.join(HERE, '_smoke_distill')
TEACHER_NB = os.path.join(ROOT, 'notebooks', 'ECERC_L2_distill_teacher.ipynb')
STUDENT_NB = os.path.join(ROOT, 'notebooks', 'ECERC_L2_distill_student.ipynb')


def sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_notebook(path, case, overrides, test_marker, stop_after=None):
    """Executes the notebook's code cells with config overrides; returns the ids of label dialogues read before the
    cell containing `test_marker` and in total. Raises whatever a cell raises."""
    work = os.path.join(SMOKE, case)
    os.makedirs(work)
    os.chdir(work)
    with open(path) as f:
        cells = [''.join(c['source']) for c in json.load(f)['cells'] if c['cell_type'] == 'code']
    ns = {'__name__': '__smoke__'}
    LoggingDict.accessed = set()
    before = None
    for i, src in enumerate(cells):
        if test_marker and test_marker in src:
            before = set(LoggingDict.accessed)
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
        if stop_after and stop_after in src:
            break
    return (before if before is not None else set(LoggingDict.accessed)), set(LoggingDict.accessed)


def teacher(case, data_path, model_dir, reference, n_dev, test_ids):
    work = os.path.join(SMOKE, case)
    ov = dict(work_dir=os.path.join(work, 'work'), cache_dir=os.path.join(work, 'cache'), offline=True,
              data_local_path=data_path, expected_data_sha256=None, expected_split_utterances=None,
              expected_split_dialogues=None, llm_model=model_dir, llm_revision=None, expected_label_token_ids=None,
              reference_dev_dump=reference, repro_dialogues=n_dev)
    _, total = run_notebook(TEACHER_NB, case, ov, None)
    assert not (total & test_ids), f'8a read test labels: {sorted(total & test_ids)[:5]}'
    print(f'  [smoke] {case}: OK, no test label read at all')
    out = os.path.join(ov['work_dir'], 'outputs_distill_teacher')
    for name in ('SUMMARY_distill_teacher.md', 'design.json', 'checks.json', 'llm_inference.json',
                 'dumps/train_llm.npz', 'dumps/repro_dev_llm.npz'):
        assert os.path.exists(os.path.join(out, name)), f'missing {name}'
    assert os.path.exists(os.path.join(ov['work_dir'], 'ecerc_l2_distill_teacher_results.zip'))
    with open(os.path.join(out, 'checks.json')) as f:
        checks = {c['name']: c['passed'] for c in json.load(f)}
    with open(os.path.join(out, 'llm_inference.json')) as f:
        info = json.load(f)
    return out, checks, info


def student(case, data_path, teacher_npz, teacher_sha, ref, gate, test_ids, stop_after=None):
    work = os.path.join(SMOKE, case)
    ref_lp = np.load(ref)['logp'].astype(np.float64)
    ov = dict(work_dir=os.path.join(work, 'work'), cache_dir=os.path.join(work, 'cache'), offline=True,
              data_local_path=data_path, expected_data_sha256=None, expected_split_utterances=None,
              expected_split_dialogues=None, teacher_local_path=teacher_npz, teacher_sha256=teacher_sha,
              dev_llm_local_path=ref, dev_llm_sha256=sha(ref), log_q_dev=list(np.log(np.exp(ref_lp).mean(0))),
              oof_seeds=[7], epochs=2, stage1_seeds=[1, 2], stage2_seeds=[11, 12, 13], gate_min_gain=gate,
              power_B=50, a0_dev_macro_f1=None, bootstrap_B=200)
    before, total = run_notebook(STUDENT_NB, case, ov, 'LD.stage_test(S)', stop_after)
    assert not (before & test_ids), f'8b read test labels before the test stage: {sorted(before & test_ids)[:5]}'
    print(f'  [smoke] {case}: OK, no test label read before the test stage')
    return ov['work_dir'], total


def main():
    if os.path.exists(SMOKE):
        shutil.rmtree(SMOKE)
    os.makedirs(SMOKE)
    data_path = os.path.join(SMOKE, 'synthetic_meld.pkl')
    splits = make_synthetic(data_path)
    test_ids = set(splits['test'])
    with open(data_path, 'rb') as f:
        D = pickle.load(f)
    y_dev = np.concatenate([np.asarray(D[1][v]) for v in splits['valid']])
    model_dir = os.path.join(SMOKE, 'tiny_qwen')
    tiny_qwen(model_dir)

    # 8a without a reference: every dev dialogue is re-scored and becomes the reference for the next run
    out, checks, _ = teacher('teacher_no_reference', data_path, model_dir, None, 1000, test_ids)
    assert checks['dev re-scoring compared with Experiment 7'] is False
    z = np.load(os.path.join(out, 'dumps', 'repro_dev_llm.npz'))
    assert len(z['logp']) == len(y_dev)
    ref = os.path.join(SMOKE, 'reference_dev_llm.npz')  # Experiment 7's dev_llm.npz layout: logp and y
    np.savez_compressed(ref, logp=z['logp'], y=y_dev)

    # 8a with the reference: the check must pass
    out, checks, info = teacher('teacher_reference', data_path, model_dir, ref, 5, test_ids)
    name = [k for k in checks if k.startswith('dev re-scoring reproduces')]
    assert name and checks[name[0]], checks
    assert info['reproducibility']['argmax_agreement'] == 1.0, info['reproducibility']
    teacher_npz = os.path.join(out, 'dumps', 'train_llm.npz')
    teacher_sha = sha(teacher_npz)
    assert info['train_llm_npz_sha256'] == teacher_sha
    print(f'[smoke] 8a: train_llm.npz {teacher_sha[:16]}..., re-scoring agreement 100%')

    # 8b, gate open: every stage runs
    work, total = student('student_gate_open', data_path, teacher_npz, teacher_sha, ref, -100.0, test_ids)
    out = os.path.join(work, 'outputs_distill')
    for n in ('SUMMARY_distill.md', 'frozen_config_distill.json', 'stage1_summary.json', 'teachers.json', 'design.json',
              'checks.json', 'tables/stage1_runs.csv', 'tables/stage2_dev_runs.csv', 'tables/stage2_test_per_seed.csv',
              'dumps/stage1_dev_preds.npz', 'dumps/teachers_train.npz', 'test_results_distill.json'):
        assert os.path.exists(os.path.join(out, n)), f'missing {n}'
    assert total & test_ids, 'the test stage never read test labels'
    preds = np.load(os.path.join(out, 'dumps', 'stage1_dev_preds.npz'))
    assert not np.array_equal(preds['D0_original__seed1'], preds['D2_T1_a0.7__seed1']), 'the KL term changed nothing'
    assert not np.array_equal(preds['D0_original__seed1'], preds['L2_uniform_a0.7__seed1']), 'label smoothing changed nothing'
    with open(os.path.join(out, 'test_results_distill.json')) as f:
        res = json.load(f)
    with open(os.path.join(out, 'stage1_summary.json')) as f:
        st1 = json.load(f)
    assert abs(res['multiplicity']['level'] - 98.75) < 1e-9 and 'adjusted_ci_lower_gt_0' in res['conditions']
    assert st1['winner']['name'][0] == 'D' and st1['winner']['name'] != 'D0_original'
    assert len(res['per_seed_delta']) == 3
    print(f'[smoke] gate open: winner {st1["winner"]["name"]}, cross-fitted gain '
          f'{st1["gate"]["crossfit_delta_macro_f1"]:+.2f}; verdict on synthetic data (meaningless): {res["verdict"]}')

    # 8b, gate shut: the test must be skipped and no test label read at all
    work, total = student('student_gate_shut', data_path, teacher_npz, teacher_sha, ref, 100.0, test_ids)
    out = os.path.join(work, 'outputs_distill')
    assert not (total & test_ids), 'test labels read although the gate was shut'
    assert not os.path.exists(os.path.join(out, 'test_results_distill.json'))
    assert os.path.exists(os.path.join(work, 'ecerc_l2_distill_results.zip'))
    print('[smoke] gate shut: stage 2 and the test were skipped as required')

    # 8b with a wrong teacher hash: it must stop at the teacher stage
    try:
        student('student_wrong_teacher', data_path, teacher_npz, '0' * 64, ref, -100.0, test_ids,
                stop_after='LD.stage_teacher(S)')
    except Exception as e:
        assert type(e).__name__ == 'CheckFailed' and 'train_llm.npz' in str(e), e
        print('[smoke] wrong teacher SHA-256: stopped as required')
    else:
        raise AssertionError('a wrong teacher SHA-256 was accepted')
    print('\n[smoke] PASSED')


if __name__ == '__main__':
    sys.exit(main())
