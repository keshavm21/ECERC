#!/usr/bin/env python3
"""Local test of the Experiment 7 notebook on synthetic data (CPU, no downloads).

A tiny randomly initialised Qwen2 causal LM and a byte-level BPE tokenizer with a Qwen-style chat template are built
offline and saved with save_pretrained, so the notebook's real from_pretrained and cached zero-shot scoring code path
runs unchanged. The in-repo MELD/ECERC_MODEL.pkl stands in for all three released checkpoints. Scenario 1 forces the
gate open (every stage runs, including the test); scenario 2 forces it shut (the test must be skipped). In both, no
test-split label may be read before the test stage.

    research/limitation2/.venv/bin/python research/limitation2/tests/smoke_test_zeroshot.py
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
SMOKE = os.path.join(HERE, '_smoke_zeroshot')
NOTEBOOK = os.path.join(ROOT, 'notebooks', 'ECERC_L2_zeroshot_llm_fusion.ipynb')
CKPT = os.path.join(REPO, 'MELD', 'ECERC_MODEL.pkl')
LABELS = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
TEMPLATE = ("{% for m in messages %}{{'<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n'}}{% endfor %}"
            "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}")


def tiny_qwen(path):
    import torch
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    os.makedirs(path)
    words = [v for w in LABELS for v in (w, w.capitalize())]
    corpus = ([f'Speaker {k}: utterance {v}-{i}\n' for k in range(1, 10) for v in range(40) for i in range(5)]
              + ['You are an expert annotator of emotions in TV-show dialogue. Here is a dialogue, in order: Which '
                 'emotion does Speaker express in the last line? Answer with one word: neutral, surprise, fear, '
                 'sadness, joy, disgust, or anger.'] * 20
              + [w + '\n' for w in words] * 60 + [' '.join(words)] * 60)
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(corpus, vocab_size=700, min_frequency=1,
                            special_tokens=['<|endoftext|>', '<|im_start|>', '<|im_end|>'])
    bpe.save(os.path.join(path, 'tokenizer.json'))
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(path, 'tokenizer.json'), eos_token='<|im_end|>',
                                  pad_token='<|endoftext|>', additional_special_tokens=['<|im_start|>', '<|im_end|>'])
    tok.chat_template = TEMPLATE
    tok.save_pretrained(path)
    torch.manual_seed(0)
    Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                 num_attention_heads=2, num_key_value_heads=1,
                                 max_position_embeddings=2048)).save_pretrained(path)


def run(case, gate, model_dir):
    work = os.path.join(SMOKE, case)
    os.makedirs(work)
    os.chdir(work)
    data_path = os.path.join(work, 'synthetic_meld.pkl')
    splits = make_synthetic(data_path)
    test_ids = set(splits['test'])
    with open(CKPT, 'rb') as f:
        ck_sha = hashlib.sha256(f.read()).hexdigest()
    overrides = dict(work_dir=os.path.join(work, 'work'), cache_dir=os.path.join(work, 'cache'), offline=True,
                     data_local_path=data_path,
                     checkpoints=[dict(name=n, local_path=CKPT, sha256=ck_sha) for n in ('C1', 'C2', 'C3')],
                     expected_split_utterances=None, expected_split_dialogues=None, min_dev_acc=0.0,
                     llm_model=model_dir, llm_revision=None, gate_min_gain=gate, power_B=200, bootstrap_B=400)
    with open(NOTEBOOK) as f:
        cells = [''.join(c['source']) for c in json.load(f)['cells'] if c['cell_type'] == 'code']
    ns = {'__name__': '__smoke__'}
    LoggingDict.accessed = set()
    for i, src in enumerate(cells):
        if 'LZ.stage_test(S)' in src:
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
    out = os.path.join(overrides['work_dir'], 'outputs_zeroshot')
    for name in ('SUMMARY_zeroshot.md', 'frozen_config_zeroshot.json', 'dev_selection.json', 'design.json',
                 'checks.json', 'llm_inference.json', 'dumps/dev_llm.npz'):
        assert os.path.exists(os.path.join(out, name)), f'missing {name}'
    with open(os.path.join(out, 'dev_selection.json')) as f:
        sel = json.load(f)
    print(f'\n[smoke] {case}: w* {sel["w_star"]}, cross-fitted gain {sel["forecast"]["crossfit_delta_macro_f1"]:+.2f}, '
          f'proceed {sel["proceed"]}')
    if gate < 0:
        assert sel['proceed'] and (LoggingDict.accessed & test_ids), 'the test stage did not run'
        for name in ('test_results_zeroshot.json', 'dumps/test_llm.npz', 'dumps/test_ecerc_C1.npz'):
            assert os.path.exists(os.path.join(out, name)), f'missing {name}'
        with open(os.path.join(out, 'test_results_zeroshot.json')) as f:
            res = json.load(f)
        print(f'[smoke] {case}: verdict on synthetic data (meaningless): {res["verdict"]}')
    else:
        assert not sel['proceed'] and not (LoggingDict.accessed & test_ids)
        assert not os.path.exists(os.path.join(out, 'test_results_zeroshot.json'))
        print(f'[smoke] {case}: the test was skipped as required')
    assert os.path.exists(os.path.join(overrides['work_dir'], 'ecerc_l2_zeroshot_results.zip'))


def main():
    if os.path.exists(SMOKE):
        shutil.rmtree(SMOKE)
    model_dir = os.path.join(SMOKE, 'tiny_qwen')
    tiny_qwen(model_dir)
    run('gate_open', -100.0, model_dir)
    run('gate_shut', 100.0, model_dir)
    print('\n[smoke] PASSED')


if __name__ == '__main__':
    sys.exit(main())
