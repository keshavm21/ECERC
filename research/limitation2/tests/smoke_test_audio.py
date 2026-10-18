#!/usr/bin/env python3
"""Local test of the Experiment 6 notebook on synthetic data (CPU, no downloads, tiny models).

Builds a synthetic feature file with the real 17-field layout, MELD-style CSVs and FLAC archives laid out like the
Hugging Face audio release, and tiny randomly initialised Whisper, WavLM and BERT models saved with save_pretrained,
so the notebook's real from_pretrained code path runs. The audio release has the awkward cases of the real one: a
missing clip, an extra clip, an utterance whose text differs, a gap in the utterance IDs, a clip longer than the cap
and one shorter than the minimum. Scenario 1 forces the gate open (every stage runs, including the test); scenario 2
forces it shut (the test must be skipped). In both, no test-split label may be read before the test stage, neither
from the feature file nor from the test CSV.

    research/limitation2/.venv/bin/python research/limitation2/tests/smoke_test_audio.py
"""
import csv
import io
import json
import os
import pickle
import shutil
import sys
import tarfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from smoke_test import LoggingDict, make_synthetic  # noqa: E402

ROOT = os.path.dirname(HERE)
SMOKE = os.path.join(HERE, '_smoke_audio')
NOTEBOOK = os.path.join(ROOT, 'notebooks', 'ECERC_L2_audio_evidence.ipynb')
NAMES = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
OFFSET = dict(train=0, valid=1000, test=2000)  # make_synthetic's dialogue keys = Dialogue_ID + offset
LAYOUT = dict(train=('train.csv', 'archive/train.tar.gz', 'train'), valid=('dev.csv', 'archive/dev.tar.gz', 'dev'),
              test=('test.csv', 'archive/test.tar.gz', 'test'))
MISSING = ('train', 5, 1)      # (split, dialogue key, position): clip left out of the archive
TEXT_DIFF = ('train', 7, 0)    # CSV text differs from the feature file
ID_GAP = ('train', 3)          # Utterance_IDs skip 2
LONG = ('valid', 1000, 0)      # 3 s, above the 2 s cap of this test
SHORT = ('test', 2000, 0)      # 200 samples, below the 1600-sample minimum


def make_audio_release(data_path, out_dir, seed=0):
    import soundfile as sf
    rng = np.random.default_rng(seed)
    with open(data_path, 'rb') as f:
        speakers, emo, _, *_, sentences, train_ids, test_ids, valid_ids = pickle.load(f)
    keys = dict(train=train_ids, valid=valid_ids, test=test_ids)
    os.makedirs(os.path.join(out_dir, 'archive'), exist_ok=True)
    for split, (csv_name, tar_name, folder) in LAYOUT.items():
        rows, clips = [], []
        for vid in keys[split]:
            dia = vid - OFFSET[split]
            ids = list(range(len(sentences[vid])))
            if (split, vid) == ID_GAP:
                ids = [u if u < 2 else u + 1 for u in ids]
            for pos, utt in enumerate(ids):
                label = int(emo[vid][pos])
                text = 'something else entirely' if (split, vid, pos) == TEXT_DIFF else sentences[vid][pos]
                rows.append({'Sr No.': len(rows) + 1, 'Utterance': text, 'Speaker': f'S{np.argmax(speakers[vid][pos])}',
                             'Emotion': 'SENTINEL' if split == 'test' else NAMES[label], 'Sentiment': 'neutral',
                             'Dialogue_ID': dia, 'Utterance_ID': utt, 'Season': 1, 'Episode': 1,
                             'StartTime': '00:00:00,000', 'EndTime': '00:00:01,000'})
                if (split, vid, pos) == MISSING:
                    continue
                n = 3 * 16000 if (split, vid, pos) == LONG else 200 if (split, vid, pos) == SHORT else \
                    int(rng.integers(4800, 19200))
                t = np.arange(n) / 16000
                x = 0.3 * np.sin(2 * np.pi * (200 + 80 * label) * t) + 0.05 * rng.normal(size=n)
                clips.append((f'{folder}/dia{dia}_utt{utt}.flac', x.astype(np.float32)))
        if split == 'train':
            clips.append(('train/dia9999_utt0.flac', np.zeros(8000, dtype=np.float32)))  # not in the CSV
        with open(os.path.join(out_dir, csv_name), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with tarfile.open(os.path.join(out_dir, tar_name), 'w:gz') as tf:
            for name, x in clips:
                buf = io.BytesIO()
                sf.write(buf, x, 16000, format='FLAC')
                blob = buf.getvalue()  # not buf.tell(): libsndfile seeks back to rewrite the header
                info = tarfile.TarInfo(name)
                info.size = len(blob)
                tf.addfile(info, io.BytesIO(blob))
    corpus = [s for vid in sentences for s in sentences[vid]]
    return corpus


def tiny_models(path, corpus):
    import torch
    from tokenizers import BertWordPieceTokenizer
    from transformers import (BertConfig, BertModel, BertTokenizerFast, Wav2Vec2FeatureExtractor, WavLMConfig,
                              WavLMModel, WhisperConfig, WhisperFeatureExtractor, WhisperModel)
    torch.manual_seed(0)  # the same tiny encoders in every run
    dirs = {k: os.path.join(path, k) for k in ('whisper', 'wavlm', 'bert')}
    for d in dirs.values():
        os.makedirs(d)
    WhisperModel(WhisperConfig(num_mel_bins=80, d_model=32, encoder_layers=4, encoder_attention_heads=2, decoder_layers=1,
                               decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64,
                               max_source_positions=1500, max_target_positions=32, vocab_size=100, pad_token_id=0,
                               bos_token_id=1, eos_token_id=2, decoder_start_token_id=3)).save_pretrained(dirs['whisper'])
    WhisperFeatureExtractor(feature_size=80).save_pretrained(dirs['whisper'])
    WavLMModel(WavLMConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=2, intermediate_size=64,
                           conv_dim=(32,) * 7, feat_extract_norm='layer', do_stable_layer_norm=True, num_buckets=16,
                           max_bucket_distance=50)).save_pretrained(dirs['wavlm'])
    Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, do_normalize=True,
                             return_attention_mask=True).save_pretrained(dirs['wavlm'])
    wp = BertWordPieceTokenizer(lowercase=True)
    wp.train_from_iterator(corpus, vocab_size=300, min_frequency=1)
    wp.save_model(dirs['bert'])
    tok = BertTokenizerFast(vocab_file=os.path.join(dirs['bert'], 'vocab.txt'))
    tok.save_pretrained(dirs['bert'])
    BertModel(BertConfig(vocab_size=len(tok), hidden_size=1024, num_hidden_layers=1, num_attention_heads=4,
                         intermediate_size=64, max_position_embeddings=64)).save_pretrained(dirs['bert'])
    return dirs


def run(case, gate, epochs):
    work = os.path.join(SMOKE, case)
    os.makedirs(work)
    os.chdir(work)
    data_path = os.path.join(work, 'synthetic_meld.pkl')
    splits = make_synthetic(data_path)
    test_ids = set(splits['test'])
    audio_dir = os.path.join(work, 'audio_release')
    dirs = tiny_models(os.path.join(work, 'models'), make_audio_release(data_path, audio_dir))
    overrides = dict(
        work_dir=os.path.join(work, 'work'), cache_dir=os.path.join(work, 'cache'), offline=True,
        data_local_path=data_path, expected_data_sha256=None, expected_split_utterances=None,
        expected_split_dialogues=None, audio_local_dir=audio_dir, audio_dataset=dict(repo='synthetic', revision=None),
        audio_encoders=[dict(name='tiny-whisper', repo=dirs['whisper'], revision=None, kind='whisper', batch=4),
                        dict(name='tiny-wavlm', repo=dirs['wavlm'], revision=None, kind='wavlm', batch=1)],
        text_encoder=dict(name='tiny-bert', repo=dirs['bert'], revision=None, max_len=32, batch=16),
        layer_band=2, max_audio_seconds=2.0, max_missing_audio=5, probe_max_iter=200, probe_jobs=2,
        epochs=epochs, stage1_seeds=[1, 2], stage2_seeds=[11, 12, 13], gate_min_gain=gate,
        pathb_c0_dev_macro_f1=None, bootstrap_B=200)
    with open(NOTEBOOK) as f:
        cells = [''.join(c['source']) for c in json.load(f)['cells'] if c['cell_type'] == 'code']
    ns = {'__name__': '__smoke__'}
    LoggingDict.accessed = set()
    csv_reads = []
    for i, src in enumerate(cells):
        if 'LA.stage_test(S)' in src:
            leaked = LoggingDict.accessed & test_ids
            assert not leaked, f'test labels read before the test stage: {sorted(leaked)[:5]}'
            print('  [smoke] OK: no test label read before the test stage')
        exec(compile(src, f'<{case} cell {i}>', 'exec'), ns)
        if i == 0:
            ns['CONFIG'].update(overrides)
        if 'stage_setup' in src:
            P, LA = ns['l2_posthoc'], ns['LA']
            original_load, original_csv = P.load_pickle, LA.read_meld_csv

            def logged_load(path, trust=False, _orig=original_load):
                obj = list(_orig(path, trust))
                obj[1] = LoggingDict(obj[1])  # emotion_labels
                obj[2] = LoggingDict(obj[2])  # sentiment_labels
                return tuple(obj)

            def logged_csv(path, with_labels, _orig=original_csv):
                csv_reads.append((os.path.basename(path), with_labels))
                return _orig(path, with_labels)

            P.load_pickle, LA.read_meld_csv = logged_load, logged_csv
    assert ('test.csv', False) in csv_reads and ('test.csv', True) not in csv_reads, f'test CSV read with labels: {csv_reads}'
    out = os.path.join(overrides['work_dir'], 'outputs_audio')
    for name in ('SUMMARY_audio.md', 'frozen_config_audio.json', 'stage1_summary.json', 'design.json', 'checks.json',
                 'audio_alignment.json', 'probe_selection.json', 'feature_extraction.json', 'tables/stage1_runs.csv',
                 'tables/probe_table.csv'):
        assert os.path.exists(os.path.join(out, name)), f'missing {name}'
    with open(os.path.join(out, 'audio_alignment.json')) as f:
        al = json.load(f)
    reasons = sorted(m['reason'] for m in al['no_audio'])
    assert reasons == ['no clip', 'text differs'], f'unexpected utterances without audio: {al["no_audio"]}'
    assert al['splits']['train']['archive_clips_not_needed'] == 1 and al['splits']['train']['offset'] == 0
    assert al['splits']['valid']['offset'] == 1000 and al['splits']['test']['offset'] == 2000
    with open(os.path.join(out, 'feature_extraction.json')) as f:
        ex = json.load(f)
    for name in ('tiny-whisper', 'tiny-wavlm'):
        assert ex[name]['truncated'] == 1 and ex[name]['padded'] == 1, f'{name}: {ex[name]}'
    with open(os.path.join(out, 'stage1_summary.json')) as f:
        st1 = json.load(f)
    with open(os.path.join(out, 'probe_selection.json')) as f:
        sel = json.load(f)
    print(f'\n[smoke] {case}: selected {sel["selected"]["name"]}; stage-1 winner {st1["winner"]["name"]}, '
          f'proceed {st1["proceed_to_stage2"]}')
    if gate < 0:
        assert st1['proceed_to_stage2']
        assert LoggingDict.accessed & test_ids, 'the test stage never read test labels'
        for name in ('test_results_audio.json', 'tables/stage2_test_per_seed.csv', 'tables/stage2_dev_runs.csv'):
            assert os.path.exists(os.path.join(out, name)), f'missing {name}'
        with open(os.path.join(out, 'test_results_audio.json')) as f:
            res = json.load(f)
        print(f'[smoke] {case}: verdict on synthetic data (meaningless): {res["verdict"]}')
    else:
        assert not st1['proceed_to_stage2']
        assert not (LoggingDict.accessed & test_ids), 'test labels read although stage 2 was skipped'
        assert not os.path.exists(os.path.join(out, 'test_results_audio.json'))
        print(f'[smoke] {case}: stage 2 and the test were skipped as required')
    assert os.path.exists(os.path.join(overrides['work_dir'], 'ecerc_l2_audio_results.zip'))


def main():
    if os.path.exists(SMOKE):
        shutil.rmtree(SMOKE)
    run('gate_open', gate=-100.0, epochs=2)
    run('gate_shut', gate=100.0, epochs=1)
    print('\n[smoke] PASSED')


if __name__ == '__main__':
    sys.exit(main())
