"""
ECERC Limitation 2, Experiment 6: better evidence features (self-supervised speech features, fixed audio routing).

New code for research/limitation2. Experiments 1-5 placed ECERC's rare-emotion ceiling in its precomputed input
evidence. The weakest stream is audio: 300-d openSMILE functionals with a +7.2-point train-side probe advantage
(Experiment 2), routed by the original loader into the wrong model slots (the 300-d "audio" input holds the first
300 visual dimensions; the "visual" input holds the last 42 visual dimensions plus all of the audio). This
experiment computes frozen self-supervised speech features from the MELD audio and retrains the unchanged ECERC
model (architecture, loss, optimiser, schedule) on them with many paired seeds.

Arms (every evaluated model is the original ECERC class; only its input features change):
  A0  original features and slot routing (the original recipe, Path B's C0)
  A1  original features, audio and visual routed to the slots the model expects
  A2  A1 with openSMILE replaced by the selected speech-encoder features
  A3  A2 with the event input (sroberta1, plain RoBERTa-large) replaced by bge-large-en-v1.5 sentence embeddings
Feature selection (encoder and layer band) uses the TRAIN split only: the feature-leak check's linear probe on
5 folds of training dialogues, highest out-of-fold macro-F1.
Stage 1, dev only: A0-A3 x 5 seeds. The winner (A1-A3) has the highest mean dev macro-F1; stage 2 runs only if it
beats A0 by at least CONFIG['gate_min_gain'].
Freeze: the selected features (with the SHA-256 of the final arrays), winner, seeds and criteria go to
frozen_config_audio.json before any test access.
Stage 2, confirmatory: fresh seeds x {A0, winner}; each run keeps its best-dev-macro-F1 epoch; the test set is
evaluated once at the end.
Test-set rule: the frozen encoders run on the audio and text of every split (label-free preprocessing, like the
authors' precomputed features). Before the freeze no ECERC model runs on test dialogues and no test label is read,
neither from the data file nor from the MELD CSV.
"""
import gc
import hashlib
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

import l2_posthoc as P        # safe loader, data gate, dataset wrapper, inference and metrics (unchanged)
import l2_feature_leak as LK  # the linear probe of the feature-leak check (unchanged)
import l2_pathb as LB         # pickle download, seeding, hierarchical bootstrap and verdict rules (unchanged)

C, CLASSES = P.C, P.CLASSES
NEU, SUR, FEA, SAD, JOY, DIS, ANG = range(C)
SPLITS = ('train', 'valid', 'test')
EMOTION_IDS = {'neutral': NEU, 'surprise': SUR, 'fear': FEA, 'sadness': SAD, 'joy': JOY, 'disgust': DIS, 'anger': ANG}
AUDIO_FILES = {'train': ('train.csv', 'archive/train.tar.gz'), 'valid': ('dev.csv', 'archive/dev.tar.gz'),
               'test': ('test.csv', 'archive/test.tar.gz')}
ARMS = [
    dict(name='A0_original', routing='original', audio='opensmile', event='sroberta1'),
    dict(name='A1_routing_fixed', routing='fixed', audio='opensmile', event='sroberta1'),
    dict(name='A2_ssl_audio', routing='fixed', audio='ssl', event='sroberta1'),
    dict(name='A3_ssl_audio_bge_event', routing='fixed', audio='ssl', event='bge'),
]
BASE = ARMS[0]
CRITERIA = dict(LB.CRITERIA, primary='mean over stage-2 seeds of test macro-F1(winner) - macro-F1(A0), paired by seed, '
                                     'in points')
ALIGNMENT_RULE = ('data-file dialogue key = MELD Dialogue_ID + a per-split offset, taken from the dialogues whose '
                  'normalised utterance text matches exactly (>= 95% required); every dialogue must have the same number '
                  'of utterances; utterance i <-> the i-th CSV row by Utterance_ID; normalised text must agree for '
                  '>= min_text_agreement of the utterances; train/dev CSV labels must equal the data-file labels. '
                  'Utterances whose text disagrees or whose clip is missing get no audio (the training mean).')
_FIX = {'\x92': "'", '\x91': "'", '\x93': '"', '\x94': '"', '\x96': '-', '\x97': '-', '\x85': '...', '\xa0': ' '}


def _clean(s):  # Experiment 4's clean-up of the cp1252 characters in MELD's text; punctuation is kept
    s = str(s)
    for a, b in _FIX.items():
        s = s.replace(a, b)
    return ' '.join(s.split())


def norm_text(s):
    return re.sub(r'[^0-9a-z]+', '', _clean(s).lower())


def build_model(S, d_a, d_v=342):
    """MELD/inference.py's constructor (feature_type='multi', base_layer=1) with the audio width as a parameter.
    With d_a = 300 it is l2_posthoc.build_model exactly, so A0 draws the same random numbers as Path B's C0."""
    args = SimpleNamespace(feature_type='multi', Dataset='MELD', base_layer=1)
    return S['m_model'].ECERC(args, d_t=1024, d_a=d_a, d_v=d_v, base_layer=1, input_size=1024 + d_a + d_v,
                              hidden_size=128, n_speakers=9, n_classes=C, cuda_flag=True)


# ----------------------------------------------------------------------------- setup and the feature file
def stage_setup(CONFIG, code_dir):
    out = os.path.join(CONFIG['work_dir'], 'outputs_audio')
    for sub in ('', 'tables'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    cache = CONFIG['cache_dir']  # large downloads, features and checkpoints; kept out of the saved notebook output
    for sub in ('hf', 'features', 'checkpoints'):
        os.makedirs(os.path.join(cache, sub), exist_ok=True)
    import pandas
    import sklearn
    import transformers
    try:
        import soundfile
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'soundfile'], check=True)
        import soundfile
    S = dict(cfg=CONFIG, out=out, cache=cache, code_dir=code_dir, checks=[], torch=torch)
    cuda = torch.cuda.is_available()
    if not cuda:
        P.install_cpu_cuda_shim(torch)
    S['device'] = 'cuda' if cuda else 'cpu'
    torch.use_deterministic_algorithms(True, warn_only=True)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    for name in ('model', 'dataloader', 'loss', 'inference'):
        S['m_' + name] = __import__(name)
    S['env'] = dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                    transformers=transformers.__version__, soundfile=soundfile.__version__, numpy=np.__version__,
                    pandas=pandas.__version__, sklearn=sklearn.__version__, device=S['device'],
                    gpu=torch.cuda.get_device_name(0) if cuda else None, cpus=os.cpu_count(),
                    code_sha256={f: P.sha256_file(os.path.join(code_dir, f))
                                 for f in sorted(os.listdir(code_dir)) if f.endswith('.py')})
    P.write_json(os.path.join(out, 'environment.json'), S['env'])
    P.write_json(os.path.join(out, 'design.json'),  # written before any data is loaded
                 dict(arms=ARMS, audio_dataset=CONFIG['audio_dataset'], audio_encoders=CONFIG['audio_encoders'],
                      text_encoder=CONFIG['text_encoder'], layer_band=CONFIG['layer_band'],
                      audio_limits=dict(max_seconds=CONFIG['max_audio_seconds'], min_samples=CONFIG['min_audio_samples']),
                      alignment_rule=ALIGNMENT_RULE, max_missing_audio=CONFIG['max_missing_audio'],
                      min_text_agreement=CONFIG['min_text_agreement'],
                      feature_standardisation='training-utterance mean and SD per dimension (new features only)',
                      selection=dict(rule='highest out-of-fold train macro-F1; ties to the earlier candidate',
                                     probe="the feature-leak check's: standardised features, logistic regression",
                                     C=CONFIG['probe_C'], max_iter=CONFIG['probe_max_iter'], folds=CONFIG['probe_folds'],
                                     fold_seed=CONFIG['probe_fold_seed']),
                      epochs=CONFIG['epochs'], stage1_seeds=CONFIG['stage1_seeds'], stage2_seeds=CONFIG['stage2_seeds'],
                      gate_min_gain=CONFIG['gate_min_gain'], test_looks=CONFIG['test_looks'], criteria=CRITERIA))
    if not cuda:
        print('WARNING: no GPU. Feature extraction and training will be very slow; switch the accelerator to GPU.')
    P.log_event(S, 'setup', device=S['device'], gpu=S['env']['gpu'], torch=torch.__version__,
                transformers=transformers.__version__)
    return S


def stage_prepare(S):
    """The authors' feature file (Google Drive) and the data gate of the first experiment (train/dev labels only)."""
    LB.stage_download(S)
    P.stage_load_data(S)
    D = S['D']
    ok = all(len(D['sentences'][v]) == len(D['videoAudio'][v]) and all(isinstance(x, str) for x in D['sentences'][v])
             for s in SPLITS for v in S['keys'][s])
    P.record_check(S, 'utterance text is present for every utterance (label-free check)', ok, hard=True)
    S['order'] = [(s, v, i) for s in SPLITS for v in S['keys'][s] for i in range(len(D['videoAudio'][v]))]
    S['split_of'] = np.array([SPLITS.index(o[0]) for o in S['order']])
    P._save_checks(S)
    P.log_event(S, 'feature file ready', utterances=len(S['order']))


# ----------------------------------------------------------------------------- audio and alignment
def _get_audio_file(S, name):
    cfg = S['cfg']
    if cfg.get('audio_local_dir'):
        path, status = os.path.join(cfg['audio_local_dir'], name), 'local file'
    else:
        from huggingface_hub import hf_hub_download
        ds = cfg['audio_dataset']
        for attempt in range(1, 6):
            try:
                path = hf_hub_download(ds['repo'], name, repo_type='dataset', revision=ds['revision'],
                                       cache_dir=os.path.join(S['cache'], 'hf'))
                break
            except Exception as e:  # transient network errors are common on free GPUs
                if attempt == 5:
                    raise
                print(f'  retry {attempt} for {name} after error: {e}')
                time.sleep(10 * attempt)
        status = 'downloaded'
    got = P.sha256_file(path)
    expected = cfg['audio_dataset'].get('sha256', {}).get(name)
    if expected:
        P.record_check(S, f'{name}: SHA-256 matches the pinned release', got == expected, hard=True, detail=status)
    return path, got


def read_meld_csv(path, with_labels):
    """Rows of a MELD CSV. Without labels, the Emotion and Sentiment columns never leave this function."""
    import csv
    rows = []
    with open(path, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            rows.append(dict(dia=int(r['Dialogue_ID']), utt=int(r['Utterance_ID']), text=r['Utterance'],
                             emotion=r['Emotion'].strip().lower() if with_labels else None))
    return rows


def align_split(S, split, rows, with_labels):
    """Maps every utterance of the data file to a MELD (Dialogue_ID, Utterance_ID); see ALIGNMENT_RULE."""
    D, keys, cfg = S['D'], S['keys'][split], S['cfg']
    by_dia = {}
    for r in rows:
        by_dia.setdefault(r['dia'], []).append(r)
    for rs in by_dia.values():
        rs.sort(key=lambda r: r['utt'])
    index = {}
    for dia, rs in by_dia.items():
        index.setdefault(tuple(norm_text(r['text']) for r in rs), []).append(dia)
    votes = Counter()
    for vid in keys:
        cands = index.get(tuple(norm_text(t) for t in D['sentences'][vid]), [])
        if len(cands) == 1:
            votes[int(vid) - cands[0]] += 1
    offset, n_exact = votes.most_common(1)[0] if votes else (0, 0)
    P.record_check(S, f'{split}: data-file key = MELD Dialogue_ID + {offset} (from the exactly matching dialogues)',
                   n_exact >= 0.95 * len(keys), hard=True, detail=f'{n_exact}/{len(keys)} dialogues match exactly')
    out, count_bad, text_bad, label_bad = [], [], [], 0
    for vid in keys:
        rs = by_dia.get(int(vid) - offset)
        n = len(D['videoAudio'][vid])
        if rs is None or len(rs) != n:
            count_bad.append(int(vid))
            continue
        labels = D['emotion_labels'][vid] if with_labels else None  # never for the test split
        for i, r in enumerate(rs):
            ok = norm_text(D['sentences'][vid][i]) == norm_text(r['text'])
            if not ok:
                text_bad.append(dict(key=int(vid), position=i, dialogue_id=r['dia'], utterance_id=r['utt']))
            if with_labels and EMOTION_IDS.get(r['emotion'], -1) != int(labels[i]):
                label_bad += 1
            out.append(dict(dia=r['dia'], utt=r['utt'], text_ok=ok))
    P.record_check(S, f'{split}: every dialogue has a MELD counterpart with the same number of utterances',
                   not count_bad, hard=True, detail=f'{len(count_bad)} mismatched: {count_bad[:10]}')
    n_utt = sum(len(D['videoAudio'][v]) for v in keys)
    agree = 1 - len(text_bad) / max(n_utt, 1)
    P.record_check(S, f'{split}: utterance text agrees with the MELD CSV', agree >= cfg['min_text_agreement'], hard=True,
                   detail=f'{100 * agree:.2f}% ({len(text_bad)} differ)')
    if with_labels:
        P.record_check(S, f'{split}: MELD CSV labels equal the data-file labels', label_bad == 0, hard=True,
                       detail=f'{label_bad} differ')
    return out, dict(offset=int(offset), exact_dialogues=int(n_exact), dialogues=len(keys), utterances=n_utt,
                     text_mismatches=text_bad)


def read_archive(path, wanted):
    """FLAC bytes of the wanted (Dialogue_ID, Utterance_ID) clips, read in memory (nothing is extracted to disk)."""
    import tarfile
    got, extra = {}, 0
    with tarfile.open(path, 'r:gz') as tf:
        for m in tf:
            hit = re.fullmatch(r'(?:.*/)?dia(\d+)_utt(\d+)\.flac', m.name) if m.isfile() else None
            if hit is None:
                continue
            k = (int(hit.group(1)), int(hit.group(2)))
            if k in wanted:
                got[k] = tf.extractfile(m).read()
            else:
                extra += 1
    return got, extra


def stage_audio(S):
    """Downloads the MELD audio (16 kHz FLAC, SHA-256 checked) and aligns it with the data file."""
    cfg, t0 = S['cfg'], time.time()
    files = {name: _get_audio_file(S, name) for s in SPLITS for name in AUDIO_FILES[s]}
    clips, stats = [], {}
    for s in SPLITS:
        with_labels = s != 'test'
        aligned, st = align_split(S, s, read_meld_csv(files[AUDIO_FILES[s][0]][0], with_labels), with_labels)
        blobs, extra = read_archive(files[AUDIO_FILES[s][1]][0], {(a['dia'], a['utt']) for a in aligned})
        for a in aligned:
            a['blob'] = blobs.get((a['dia'], a['utt'])) if a['text_ok'] else None
            a['missing'] = 'text differs' if not a['text_ok'] else ('no clip' if a['blob'] is None else None)
        st.update(clips_used=sum(a['blob'] is not None for a in aligned),
                  clips_missing=sum(a['missing'] == 'no clip' for a in aligned), archive_clips_not_needed=extra)
        clips += aligned
        stats[s] = st
    P.record_check(S, 'every utterance of the data file is aligned', len(clips) == len(S['order']), hard=True)
    no_audio = [dict(split=o[0], key=int(o[1]), position=o[2], dialogue_id=a['dia'], utterance_id=a['utt'],
                     reason=a['missing']) for o, a in zip(S['order'], clips) if a['missing']]
    P.record_check(S, f'utterances without usable audio <= {cfg["max_missing_audio"]}',
                   len(no_audio) <= cfg['max_missing_audio'], hard=True,
                   detail=f'{len(no_audio)}: ' + ', '.join(f'{m["split"]} dia{m["dialogue_id"]}_utt{m["utterance_id"]} '
                                                           f'({m["reason"]})' for m in no_audio))
    S['clips'] = clips
    S['has_audio'] = np.array([a['missing'] is None for a in clips])
    S['audio_files'] = {n: dict(sha256=h) for n, (_, h) in files.items()}
    P.write_json(os.path.join(S['out'], 'audio_alignment.json'),
                 dict(dataset=cfg['audio_dataset'].get('repo'), revision=cfg['audio_dataset'].get('revision'),
                      files=S['audio_files'], rule=ALIGNMENT_RULE, splits=stats, no_audio=no_audio))
    P._save_checks(S)
    P.log_event(S, 'audio aligned', seconds=round(time.time() - t0, 1), no_audio=len(no_audio),
                clips={s: stats[s]['clips_used'] for s in SPLITS})


# ----------------------------------------------------------------------------- frozen encoders
def _decode(blob, max_samples, min_samples):
    import soundfile as sf
    x, sr = sf.read(io.BytesIO(blob), dtype='float32', always_2d=True)
    if sr != 16000:
        raise P.CheckFailed(f'audio clip with sample rate {sr}; 16000 expected')
    x = x.mean(1)
    n = len(x)
    x = x[:max_samples]
    if len(x) < min_samples:
        x = np.pad(x, (0, min_samples - len(x)))
    return x, n


def _speech_batch(model, fe, kind, xs, hop, dev, half):
    """(B, layers + 1, D): each layer's hidden states averaged over the frames that cover real audio."""
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=half):
        if kind == 'whisper':  # Whisper always reads a 30 s window; frames beyond the clip are padding
            feats = fe(xs, sampling_rate=16000, return_tensors='pt')['input_features'].to(dev)
            hs = model(feats, output_hidden_states=True).hidden_states
            valid = [max(1, math.ceil(len(x) / hop)) for x in xs]
        else:  # batch of one: no padding, every frame is real audio
            iv = fe(xs[0], sampling_rate=16000, return_tensors='pt')
            hs = model(iv['input_values'].to(dev), output_hidden_states=True).hidden_states
            valid = [hs[0].shape[1]]
    T = hs[0].shape[1]
    n = torch.tensor([min(T, v) for v in valid], device=dev)
    mask = (torch.arange(T, device=dev)[None, :] < n[:, None]).float()[..., None]
    return torch.stack([(h.float() * mask).sum(1) / n[:, None] for h in hs], 1).cpu().numpy()


def extract_speech(S, enc):
    """Per utterance and layer: the mean over valid frames of the frozen encoder's hidden states, (N, L+1, D)."""
    from transformers import AutoFeatureExtractor, AutoModel, WhisperModel
    cfg, dev = S['cfg'], S['device']
    cuda = dev == 'cuda'
    kw = dict(revision=enc.get('revision'), cache_dir=os.path.join(S['cache'], 'hf'))
    fe = AutoFeatureExtractor.from_pretrained(enc['repo'], **kw)
    if enc['kind'] == 'whisper':
        full = WhisperModel.from_pretrained(enc['repo'], dtype=torch.float16 if cuda else torch.float32, **kw)
        model, hop = full.get_encoder().float(), 2 * fe.hop_length  # the second convolution has stride 2
        del full
        bs = enc['batch']
    else:
        model, hop, bs = AutoModel.from_pretrained(enc['repo'], dtype=torch.float32, **kw), None, 1
    model.to(dev).eval()
    max_s, min_s = int(cfg['max_audio_seconds'] * 16000), int(cfg['min_audio_samples'])
    idx = [i for i, a in enumerate(S['clips']) if a['blob'] is not None]
    out, info, t0, report, bad = None, dict(truncated=0, padded=0, fp32_retries=0), time.time(), 0, 0
    with torch.no_grad():
        for b in range(0, len(idx), bs):
            items = idx[b:b + bs]
            wav = [_decode(S['clips'][i]['blob'], max_s, min_s) for i in items]
            info['truncated'] += sum(n > max_s for _, n in wav)
            info['padded'] += sum(n < min_s for _, n in wav)
            xs = [x for x, _ in wav]
            means = _speech_batch(model, fe, enc['kind'], xs, hop, dev, half=cuda)
            if not np.isfinite(means).all():  # fp16 overflow: recompute this batch in fp32
                means = _speech_batch(model, fe, enc['kind'], xs, hop, dev, half=False)
                info['fp32_retries'] += 1
            if out is None:
                out = np.full((len(S['clips']),) + means.shape[1:], np.nan, dtype=np.float16)
            m16 = means.astype(np.float16)  # stored in half precision to keep memory low
            bad += int(not np.isfinite(m16).all())
            out[items] = m16
            if b + len(items) >= report:
                print(f'  [{enc["name"]}] {b + len(items)}/{len(idx)} clips, {time.time() - t0:.0f}s', flush=True)
                report += max(1, len(idx) // 10)
    P.record_check(S, f'{enc["name"]}: features are finite for every clip', bad == 0, hard=True,
                   detail=f'shape {out.shape}, fp32 retries {info["fp32_retries"]}, non-finite batches {bad}')
    info.update(name=enc['name'], layers=int(out.shape[1] - 1), dim=int(out.shape[2]), seconds=round(time.time() - t0, 1))
    del model
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return out, info


def extract_text(S, enc):
    """bge-large-en-v1.5 sentence embeddings: [CLS] of the last layer, L2-normalised (the model's own pooling)."""
    from transformers import AutoModel, AutoTokenizer
    dev = S['device']
    cuda = dev == 'cuda'
    kw = dict(revision=enc.get('revision'), cache_dir=os.path.join(S['cache'], 'hf'))
    tok = AutoTokenizer.from_pretrained(enc['repo'], **kw)
    model = AutoModel.from_pretrained(enc['repo'], dtype=torch.float32, **kw).to(dev).eval()
    P.record_check(S, f'{enc["name"]}: 1024-d output, the width of the event input it replaces',
                   model.config.hidden_size == 1024, hard=True, detail=model.config.hidden_size)
    D, t0 = S['D'], time.time()
    texts = [_clean(D['sentences'][v][i]) for _, v, i in S['order']]
    order = np.argsort([len(t) for t in texts], kind='stable')
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(texts), enc['batch']):
            idx = order[b:b + enc['batch']]
            batch = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=enc['max_len'],
                        return_tensors='pt').to(dev)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cuda):
                h = model(**batch).last_hidden_state[:, 0]
            out[idx] = F.normalize(h.float(), dim=-1).cpu().numpy()
    info = dict(name=enc['name'], dim=int(out.shape[1]), seconds=round(time.time() - t0, 1),
                truncated=int(sum(len(tok(t)['input_ids']) > enc['max_len'] for t in texts)))
    del model
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    return out, info


def _cached(S, enc, compute):
    """Features computed once per alignment and encoder revision (a re-run in the same session skips the work)."""
    key = hashlib.sha256(json.dumps(dict(enc=enc, data=S['data_sha256'], clips=[
        (a['dia'], a['utt'], a['missing']) for a in S['clips']], limits=[S['cfg']['max_audio_seconds'],
                                                                         S['cfg']['min_audio_samples']]),
                                    sort_keys=True).encode()).hexdigest()[:16]
    path = os.path.join(S['cache'], 'features', f'{enc["name"]}_{key}')
    if os.path.exists(path + '.json'):
        with open(path + '.json') as f:
            info = json.load(f)
        print(f'  [{enc["name"]}] features loaded from this session\'s cache')
        return np.load(path + '.npy'), info
    arr, info = compute()
    np.save(path + '.npy', arr)
    P.write_json(path + '.json', info)
    return arr, info


def stage_extract(S):
    """Frozen speech features for every clip (both encoders, every layer) and bge embeddings for every utterance."""
    cfg, t0 = S['cfg'], time.time()
    S['layer_means'], S['extraction'] = {}, {}
    for enc in cfg['audio_encoders']:
        S['layer_means'][enc['name']], S['extraction'][enc['name']] = _cached(S, enc, lambda e=enc: extract_speech(S, e))
    te = cfg['text_encoder']
    S['text_emb'], S['extraction'][te['name']] = _cached(S, te, lambda: extract_text(S, te))
    for a in S['clips']:
        a['blob'] = None  # the audio is no longer needed
    gc.collect()
    P.write_json(os.path.join(S['out'], 'feature_extraction.json'), S['extraction'])
    P._save_checks(S)
    P.log_event(S, 'features extracted', minutes=round((time.time() - t0) / 60, 1),
                **{k: v.get('seconds') for k, v in S['extraction'].items()})


# ----------------------------------------------------------------------------- selection (train split only)
def _bands(n_layers, size):
    return [(lo, min(lo + size - 1, n_layers)) for lo in range(1, n_layers + 1, size)]


def _per_dialogue(S, Z):
    out, k = {}, 0
    for s in SPLITS:
        for v in S['keys'][s]:
            n = len(S['D']['videoAudio'][v])
            out[v] = Z[k:k + n]
            k += n
    return out


def _standardise(X, fit_rows, fill_rows=None):
    mu, sd = X[fit_rows].mean(0), X[fit_rows].std(0) + 1e-6
    Z = ((X - mu) / sd).astype(np.float32)
    if fill_rows is not None:
        Z[fill_rows] = 0.0  # no usable audio: the training mean
    return Z


def stage_select(S):
    """Chooses the speech encoder and layer band on the training split, then builds every arm's inputs."""
    from joblib import Parallel, delayed
    cfg, D = S['cfg'], S['D']
    tr, dv = S['split_of'] == 0, S['split_of'] == 1
    rows_tr, rows_dv = tr & S['has_audio'], dv & S['has_audio']
    y = {s: np.concatenate([np.asarray(D['emotion_labels'][v], dtype=np.int64) for v in S['keys'][s]])
         for s in ('train', 'valid')}  # train and dev labels only
    y_tr, y_dv = y['train'][S['has_audio'][tr]], y['valid'][S['has_audio'][dv]]
    dlg = np.concatenate([np.full(len(D['videoAudio'][v]), j) for j, v in enumerate(S['keys']['train'])])[S['has_audio'][tr]]
    rng = np.random.default_rng(cfg['probe_fold_seed'])  # the feature-leak check's folds
    fold_of = np.empty(len(S['keys']['train']), dtype=int)
    for k, part in enumerate(np.array_split(rng.permutation(len(S['keys']['train'])), cfg['probe_folds'])):
        fold_of[part] = k
    fold = fold_of[dlg]
    cands = []
    for enc in cfg['audio_encoders']:
        n_layers = S['layer_means'][enc['name']].shape[1] - 1
        for lo, hi in _bands(n_layers, cfg['layer_band']):
            cands.append(dict(name=f'{enc["name"]} layers {lo}-{hi}', encoder=enc['name'], layers=[lo, hi]))

    def matrix(c):
        return S['layer_means'][c['encoder']][:, c['layers'][0]:c['layers'][1] + 1].astype(np.float32).mean(1)

    orig = lambda f: np.concatenate([np.asarray(D[f][v], dtype=np.float32).reshape(len(D['videoAudio'][v]), -1)  # noqa: E731
                                     for s in SPLITS for v in S['keys'][s]])
    refs = [dict(name='openSMILE (original audio input)', X=orig('videoAudio')),
            dict(name='sroberta1 (original event input)', X=orig('sroberta1')),
            dict(name=f'{cfg["text_encoder"]["name"]} (event input of A3)', X=S['text_emb'])]
    feats = [matrix(c) for c in cands] + [r['X'] for r in refs]
    names = [c['name'] for c in cands] + [r['name'] for r in refs]
    t0 = time.time()
    print(f'  fitting {len(feats) * cfg["probe_folds"]} linear probes on {os.cpu_count()} CPU cores...', flush=True)
    jobs = [(j, k) for j in range(len(feats)) for k in range(cfg['probe_folds'])]
    res = Parallel(n_jobs=cfg.get('probe_jobs', -1))(
        delayed(LK._fit_predict)(feats[j][rows_tr][fold != k], y_tr[fold != k],
                                 [feats[j][rows_tr][fold == k], feats[j][rows_dv]], cfg['probe_C'], cfg['probe_max_iter'])
        for j, k in jobs)
    oof = np.full((len(feats), len(y_tr)), -1)
    dev_mf1 = np.zeros((len(feats), cfg['probe_folds']))
    hit_cap = np.zeros(len(feats), dtype=bool)
    for (j, k), (preds, n_iter) in zip(jobs, res):
        oof[j, fold == k] = preds[0]
        dev_mf1[j, k] = 100 * float(P.scores(LB._cm(y_dv, preds[1]))['mf1'])
        hit_cap[j] |= n_iter >= cfg['probe_max_iter']
    table = []
    for j, name in enumerate(names):
        m = P.scores(LB._cm(y_tr, oof[j]))
        table.append(dict(feature=name, role='candidate' if j < len(cands) else 'reference', dim=int(feats[j].shape[1]),
                          oof_train_macro_f1=100 * float(m['mf1']), oof_train_weighted_f1=100 * float(m['wf1']),
                          oof_fear_f1=100 * float(m['f1'][FEA]), oof_disgust_f1=100 * float(m['f1'][DIS]),
                          dev_macro_f1_mean_of_folds=float(dev_mf1[j].mean()), hit_max_iter=bool(hit_cap[j])))
    sel = 0
    for j in range(1, len(cands)):
        if table[j]['oof_train_macro_f1'] > table[sel]['oof_train_macro_f1'] + 1e-9:
            sel = j
    S['selected'] = dict(cands[sel], oof_train_macro_f1=table[sel]['oof_train_macro_f1'])
    Za = _standardise(matrix(cands[sel]), tr & S['has_audio'], ~S['has_audio'])
    Zt = _standardise(S['text_emb'], tr)
    S['ssl_audio'], S['bge_event'] = _per_dialogue(S, Za), _per_dialogue(S, Zt)
    S['feature_sha256'] = dict(audio=hashlib.sha256(Za.tobytes()).hexdigest(), event=hashlib.sha256(Zt.tobytes()).hexdigest())
    S['arm_data'] = {a['name']: arm_inputs(S, a) for a in ARMS}
    for a in ARMS:
        A = S['arm_data'][a['name']]
        A['datasets'] = {s: P.make_dataset(dict(S, D=A['D']), s) for s in ('train', 'valid')}
    P.write_csv(os.path.join(S['out'], 'tables', 'probe_table.csv'), table)
    P.write_json(os.path.join(S['out'], 'probe_selection.json'),
                 dict(rule='highest out-of-fold train macro-F1 among the candidates; ties to the earlier one',
                      selected=S['selected'], table=table, train_utterances_probed=int(len(y_tr)),
                      feature_sha256=S['feature_sha256'],
                      arm_input_widths={k: dict(d_a=v['d_a'], d_v=v['d_v']) for k, v in S['arm_data'].items()}))
    print('\nFEATURE SELECTION (train split only): out-of-fold linear-probe macro-F1')
    for r in table:
        print(f'  {r["feature"]:<44s} {r["oof_train_macro_f1"]:6.2f}  (dev {r["dev_macro_f1_mean_of_folds"]:5.2f}, '
              f'Fear {r["oof_fear_f1"]:4.1f}, Disgust {r["oof_disgust_f1"]:4.1f}) {r["role"]}')
    print(f'  selected: {S["selected"]["name"]}')
    P._save_checks(S)
    P.log_event(S, 'features selected', minutes=round((time.time() - t0) / 60, 1), selected=S['selected']['name'],
                oof_macro_f1=round(S['selected']['oof_train_macro_f1'], 2))


def arm_inputs(S, arm):
    """The arm's copy of the feature file. MELD/dataloader.py returns (..., videoVisual, videoAudio, ...) and the
    training loop concatenates them as [text, slot 3, slot 4], which the model splits as [text | audio | visual]."""
    D = S['D']
    v0 = S['keys']['train'][0]
    if arm['routing'] == 'original':  # [text, visual 342, audio 300] split as audio 300 | visual 342
        return dict(D=D, d_a=P._dim(D['videoAudio'][v0]), d_v=P._dim(D['videoVisual'][v0]))
    audio = D['videoAudio'] if arm['audio'] == 'opensmile' else S['ssl_audio']
    Dv = dict(D, videoVisual=audio, videoAudio=D['videoVisual'])  # slot 3 (the model's audio) <- audio
    if arm['event'] == 'bge':
        Dv['sroberta1'] = S['bge_event']
    return dict(D=Dv, d_a=P._dim(audio[v0]), d_v=P._dim(D['videoVisual'][v0]))


# ----------------------------------------------------------------------------- training (train/dev only)
def train_run(S, arm, seed, epochs):
    """One run of the unchanged ECERC model and recipe on the arm's inputs; Path B's C0 code path. Keeps the
    epoch with the best dev macro-F1 (earliest on ties). Never touches the test split."""
    from torch.utils.data import DataLoader
    dev, A = S['device'], S['arm_data'][arm['name']]
    LB.seed_everything(seed)
    model = build_model(S, A['d_a'], A['d_v']).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=2e-4)  # MELD/train.py: Adam, l2 = 0.0002
    loss_f = S['m_loss'].Loss(alpha=None)  # original MELD objective (focal-style, gamma = 1, no class weights)
    ds = A['datasets']
    loader = DataLoader(ds['train'], batch_size=32, shuffle=True, collate_fn=ds['train'].collate_fn, num_workers=0)
    curve, best, t0 = [], None, time.time()
    for epoch in range(epochs):
        model.train()
        for data in loader:
            opt.zero_grad()
            emo, sem, audio, vision, qmask, umask, label = [d.to(dev) for d in data[:-1]]
            seq = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
            emo = torch.cat([emo, audio, vision], dim=-1)  # as in MELD/train.py
            sem = torch.cat([sem], dim=-1)
            log_prob = model(emo, sem, qmask, umask, seq)
            y = torch.cat([label[j][:seq[j]] for j in range(len(label))])
            loss = loss_f(log_prob, y)
            loss.backward()
            opt.step()
        d = P.run_inference(S, model, ds['valid'])
        m = P.scores(LB._cm(d['y'], d['logp'].argmax(1)))
        curve.append(dict(epoch=epoch, macro_f1=100 * float(m['mf1']), weighted_f1=100 * float(m['wf1']),
                          acc=100 * float(m['acc'])))
        if best is None or 100 * float(m['mf1']) > best['dev_macro_f1'] + 1e-9:
            best = dict(epoch=epoch, dev_macro_f1=100 * float(m['mf1']), dev_weighted_f1=100 * float(m['wf1']),
                        dev_acc=100 * float(m['acc']), dev_f1=100 * m['f1'],
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    best.update(variant=arm['name'], seed=seed, seconds=round(time.time() - t0, 1), curve=curve)
    return best


def check_eval_path(S, arm, r):
    """Once per input layout: our evaluation loop must reproduce the original train_or_eval_model on dev."""
    from sklearn.metrics import confusion_matrix
    A = S['arm_data'][arm['name']]
    model = build_model(S, A['d_a'], A['d_v'])
    model.load_state_dict(r['state'], strict=True)
    model.to(S['device']).eval()
    ours = P.run_inference(S, model, A['datasets']['valid'])
    _, _, cm_o = P.original_function_eval(S, model, A['datasets']['valid'])
    cm = confusion_matrix(ours['y'], ours['logp'].argmax(1))
    diff = int(np.abs(cm_o - cm).sum()) if cm_o.shape == cm.shape else -1
    P.record_check(S, f'{arm["name"]}: evaluation loop reproduces the original train_or_eval_model on dev',
                   0 <= diff <= 4, hard=True, detail=f'sum |confusion diff| = {diff}')


def stage_screen(S):
    cfg = S['cfg']
    results, means, t0 = {}, {}, time.time()
    for arm in ARMS:
        res = []
        for sd in cfg['stage1_seeds']:
            r = train_run(S, arm, sd, cfg['epochs'])
            print(LB._run_line('stage 1', r), flush=True)
            if sd == cfg['stage1_seeds'][0] and arm['name'] in ('A0_original', 'A2_ssl_audio'):
                check_eval_path(S, arm, r)
            r.pop('state')
            res.append(r)
        results[arm['name']] = res
        means[arm['name']] = float(np.mean([r['dev_macro_f1'] for r in res]))
        P.log_event(S, 'stage 1 arm done', arm=arm['name'], mean_dev_macro_f1=round(means[arm['name']], 2))
    ref = cfg.get('pathb_c0_dev_macro_f1') or {}
    pairs = [(r['seed'], r['dev_macro_f1'], ref[r['seed']]) for r in results[BASE['name']] if r['seed'] in ref]
    if pairs:
        P.record_check(S, "A0 reproduces Path B's C0 on dev for the shared seeds (same code path; library versions "
                          "may differ)", all(abs(a - b) < 0.01 for _, a, b in pairs), hard=False,
                       detail=', '.join(f'seed {s}: {a:.2f} vs {b:.2f}' for s, a, b in pairs))
    base = means[BASE['name']]
    winner = ARMS[1]
    for arm in ARMS[2:]:
        if means[arm['name']] > means[winner['name']] + 1e-12:
            winner = arm
    gain = means[winner['name']] - base
    proceed = gain >= cfg['gate_min_gain']
    S['stage1'] = dict(results=results, means=means, winner=winner, gain=gain, proceed=proceed)
    rows = []
    for name, rs in results.items():
        for r in rs:
            row = dict(arm=name, seed=r['seed'], best_epoch=r['epoch'], dev_macro_f1=r['dev_macro_f1'],
                       dev_weighted_f1=r['dev_weighted_f1'], dev_acc=r['dev_acc'], seconds=r['seconds'])
            row.update({f'F1_{CLASSES[c]}': r['dev_f1'][c] for c in range(C)})
            rows.append(row)
    P.write_csv(os.path.join(S['out'], 'tables', 'stage1_runs.csv'), rows)
    P.write_json(os.path.join(S['out'], 'stage1_summary.json'),
                 dict(mean_dev_macro_f1=means, winner=winner, gain_over_A0=gain, gate=cfg['gate_min_gain'],
                      proceed_to_stage2=proceed, curves={n: [r['curve'] for r in rs] for n, rs in results.items()}))
    print('\nSTAGE 1 (dev only): mean best-epoch dev macro-F1 over seeds')
    for name in results:
        print(f'  {name:<24s} {means[name]:6.2f}  ({means[name] - base:+.2f} vs A0)')
    print(f'  winner: {winner["name"]} ({gain:+.2f}); gate {cfg["gate_min_gain"]:+.2f} -> '
          + ('stage 2 will run' if proceed else 'stage 2 skipped (the winner did not beat A0 by the required margin)'))
    P._save_checks(S)
    P.log_event(S, 'stage 1 done', minutes=round((time.time() - t0) / 60, 1), winner=winner['name'],
                gain=round(gain, 2), proceed=proceed)


# ----------------------------------------------------------------------------- freeze, confirmation, test
def stage_freeze(S):
    st, cfg = S['stage1'], S['cfg']
    frozen = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),
                  note='Written before any test-set access. Stage 2 and the test evaluation read their rules from here.',
                  code_sha256=S['env']['code_sha256'], data_sha256=S['data_sha256'],
                  audio_dataset=dict(repo=cfg['audio_dataset'].get('repo'), revision=cfg['audio_dataset'].get('revision'),
                                     files=S['audio_files']),
                  encoders=dict(audio=cfg['audio_encoders'], text=cfg['text_encoder']),
                  selected_audio=S['selected'], feature_sha256=S['feature_sha256'], control=BASE, winner=st['winner'],
                  stage1_mean_dev_macro_f1=st['means'],
                  gate=dict(gain=st['gain'], threshold=cfg['gate_min_gain'], proceed=st['proceed']),
                  stage2_seeds=cfg['stage2_seeds'], epochs=cfg['epochs'],
                  checkpoint_rule='per run, the epoch with the best dev macro-F1 (earliest on ties)',
                  bootstrap=dict(B=cfg['bootstrap_B'], seed=cfg['boot_seed'],
                                 design='hierarchical: seeds and test dialogues resampled; seed-level t-intervals also reported'),
                  multiplicity=dict(test_looks=cfg['test_looks'],
                                    note='the verdict uses the 95% rule; a Bonferroni-adjusted interval is also reported'),
                  criteria=CRITERIA, device=S['device'], env=S['env'])
    path = os.path.join(S['out'], 'frozen_config_audio.json')
    P.write_json(path, frozen)
    S['frozen_path'], S['frozen_sha256'] = path, P.sha256_file(path)
    P.log_event(S, 'FREEZE', sha256=S['frozen_sha256'][:16], winner=st['winner']['name'], proceed=st['proceed'])


def _load_frozen(S):
    if not S.get('frozen_path') or P.sha256_file(S['frozen_path']) != S['frozen_sha256']:
        raise P.CheckFailed('frozen_config_audio.json is missing or was modified: run the freeze cell first')
    with open(S['frozen_path']) as f:
        F_ = json.load(f)
    if any(P.sha256_file(os.path.join(S['code_dir'], n)) != h for n, h in F_['code_sha256'].items()):
        raise P.CheckFailed('code changed after the freeze')
    if F_['feature_sha256'] != S['feature_sha256']:
        raise P.CheckFailed('input features changed after the freeze')
    return F_


def stage_confirm(S):
    F_ = _load_frozen(S)
    S['stage2'] = None
    if not F_['gate']['proceed']:
        print('Stage 2 skipped by the pre-registered gate: the dev winner did not beat A0 by the required margin.')
        return
    arms = [('control', F_['control']), ('winner', F_['winner'])]
    runs, t0 = {a: [] for a, _ in arms}, time.time()
    for sd in F_['stage2_seeds']:
        for arm, v in arms:
            r = train_run(S, v, sd, F_['epochs'])
            print(LB._run_line('stage 2', r), flush=True)
            path = os.path.join(S['cache'], 'checkpoints', f'{arm}_seed{sd}.pt')
            torch.save(r.pop('state'), path)
            r['checkpoint'] = path
            runs[arm].append(r)
    S['stage2'] = runs
    rows = [dict(arm=arm, candidate=r['variant'], seed=r['seed'], best_epoch=r['epoch'], dev_macro_f1=r['dev_macro_f1'],
                 dev_weighted_f1=r['dev_weighted_f1'], dev_acc=r['dev_acc']) for arm, rs in runs.items() for r in rs]
    P.write_csv(os.path.join(S['out'], 'tables', 'stage2_dev_runs.csv'), rows)
    P.log_event(S, 'stage 2 training done', minutes=round((time.time() - t0) / 60, 1))


def _t_interval(x):
    n = len(x)
    if n < 2:
        return [float('nan'), float('nan')]
    try:
        from scipy.stats import t as student_t
        q = float(student_t.ppf(0.975, n - 1))
    except Exception:
        q = 2.093 if n == 20 else 2.262
    m, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    return [m - q * sd / math.sqrt(n), m + q * sd / math.sqrt(n)]


def stage_test(S):
    F_ = _load_frozen(S)
    if S.get('stage2') is None:
        S['results'] = None
        _write_summary(S, F_)
        return
    P.log_event(S, 'TEST evaluation starts (test dialogues are run and test labels read only from here on)')
    seeds, arms = F_['stage2_seeds'], [('control', F_['control']), ('winner', F_['winner'])]
    n_te = len(S['keys']['test'])
    cms = np.zeros((n_te, len(seeds), 2, C * C), dtype=np.int64)
    for a, (arm, v) in enumerate(arms):
        A = S['arm_data'][v['name']]
        test_ds = P.make_dataset(dict(S, D=A['D']), 'test')
        for i, r in enumerate(S['stage2'][arm]):
            model = build_model(S, A['d_a'], A['d_v'])
            model.load_state_dict(torch.load(r['checkpoint'], map_location=S['device'], weights_only=True), strict=True)
            model.to(S['device']).eval()
            d = P.run_inference(S, model, test_ds)
            cms[:, i, a, :] = P.per_dialogue_cms(d['y'], d['logp'].argmax(1)[None, :], d['dlg'], n_te)[:, 0, :]
            np.savez_compressed(os.path.join(S['out'], f'test_{arm}_seed{seeds[i]}.npz'), logp=d['logp'].astype(np.float32),
                                y=d['y'], dlg=d['dlg'])
    sc = P.scores(cms.sum(0).reshape(len(seeds), 2, C, C))
    diff = {k: 100 * (sc[k][:, 1] - sc[k][:, 0]) for k in ('mf1', 'acc', 'wf1', 'f1', 'prec', 'rec')}
    dl = {k: v.mean(0) for k, v in diff.items()}
    per_seed = diff['mf1']
    bt = LB.hierarchical_bootstrap(cms, F_['bootstrap']['B'], F_['bootstrap']['seed'])
    ci = lambda x, q=(2.5, 97.5): np.percentile(x, q, axis=0)  # noqa: E731
    lo, hi = ci(bt['mf1'])
    alpha = 0.05 / F_['multiplicity']['test_looks']
    b_lo, b_hi = ci(bt['mf1'], (100 * alpha / 2, 100 * (1 - alpha / 2)))
    verdict, conditions = LB.decide(float(dl['mf1']), lo, hi, dl['f1'], float(dl['acc']), float(dl['wf1']),
                                    float((per_seed > 0).mean()))
    consistency = {}
    for name, x in (('macro_f1', diff['mf1']), ('weighted_f1', diff['wf1']), ('accuracy', diff['acc']),
                    ('fear_f1', diff['f1'][:, FEA]), ('disgust_f1', diff['f1'][:, DIS])):
        consistency[name] = dict(mean=float(x.mean()), seed_t_ci95=_t_interval(x), seeds_positive=int((x > 0).sum()),
                                 seeds=len(seeds))
    arm_means = {arm: dict(acc=100 * sc['acc'][:, a].mean(), macro_f1=100 * sc['mf1'][:, a].mean(),
                           weighted_f1=100 * sc['wf1'][:, a].mean(), f1=100 * sc['f1'][:, a].mean(0),
                           precision=100 * sc['prec'][:, a].mean(0), recall=100 * sc['rec'][:, a].mean(0),
                           macro_f1_sd=100 * sc['mf1'][:, a].std(ddof=1) if len(seeds) > 1 else None)
                 for a, (arm, _) in enumerate(arms)}
    S['results'] = dict(
        verdict=verdict, conditions=conditions, winner=F_['winner'], selected_audio=F_['selected_audio'], seeds=seeds,
        delta_macro_f1=float(dl['mf1']), ci95_hierarchical=[lo, hi], ci95_seed_t=_t_interval(per_seed),
        per_seed_delta=per_seed, seeds_positive=int((per_seed > 0).sum()),
        multiplicity=dict(test_looks=F_['multiplicity']['test_looks'], level=100 * (1 - alpha),
                          ci_hierarchical=[b_lo, b_hi], lower_bound_above_0=bool(b_lo > 0)),
        consistency=consistency,
        per_class=dict(delta_f1=dl['f1'], delta_f1_ci95=ci(bt['f1']).T, delta_precision=dl['prec'], delta_recall=dl['rec']),
        guardrails=dict(delta_acc=float(dl['acc']), delta_acc_ci95=ci(bt['acc']), delta_weighted_f1=float(dl['wf1']),
                        delta_weighted_f1_ci95=ci(bt['wf1'])),
        arms=arm_means, support=sc['support'][0, 0],
        fidelity=dict(control_weighted_f1=float(arm_means['control']['weighted_f1']), paper_weighted_f1=P.PAPER_TEST_WF1))
    rows = []
    for i, sd in enumerate(seeds):
        for a, (arm, v) in enumerate(arms):
            row = dict(seed=sd, arm=arm, candidate=v['name'], acc=100 * sc['acc'][i, a], macro_f1=100 * sc['mf1'][i, a],
                       weighted_f1=100 * sc['wf1'][i, a])
            row.update({f'F1_{CLASSES[c]}': 100 * sc['f1'][i, a, c] for c in range(C)})
            rows.append(row)
    P.write_csv(os.path.join(S['out'], 'tables', 'stage2_test_per_seed.csv'), rows)
    P.write_json(os.path.join(S['out'], 'test_results_audio.json'), S['results'])
    _write_summary(S, F_)
    P.log_event(S, 'TEST done', verdict=verdict, delta=round(float(dl['mf1']), 2),
                ci=[round(float(lo), 2), round(float(hi), 2)])


def _write_summary(S, F_):
    st, env, sel = S['stage1'], S['env'], S['selected']
    L = ['# ECERC Limitation 2, Experiment 6: self-supervised audio evidence', '',
         f'Run {time.strftime("%Y-%m-%d %H:%M")} | {env["device"]} ({env["gpu"]}) | torch {env["torch"]} | '
         f'transformers {env["transformers"]} | data sha256 {S["data_sha256"][:16]}...', '',
         '## Features (train split only)', '',
         f'- audio clips aligned to the feature file; utterances without usable audio: '
         f'{int((~S["has_audio"]).sum())} (training mean used)',
         f'- selected speech features: **{sel["name"]}** (out-of-fold train probe macro-F1 {sel["oof_train_macro_f1"]:.2f}); '
         'see tables/probe_table.csv', '',
         '## Stage 1: dev screen (mean best-epoch dev macro-F1 over seeds; selection only, not a result)', '',
         '| arm | dev macro-F1 | vs A0 |', '|---|---|---|']
    base = st['means'][BASE['name']]
    for name, m in st['means'].items():
        L.append(f'| {name} | {m:.2f} | {m - base:+.2f} |')
    L += ['', f'Winner: **{st["winner"]["name"]}** ({st["gain"]:+.2f} on dev); gate {S["cfg"]["gate_min_gain"]:+.2f} -> '
              + ('stage 2 ran' if st['proceed'] else 'stage 2 skipped, test not evaluated'), '']
    res = S.get('results')
    if res:
        L += [f'## Verdict: {res["verdict"]}', '', '| condition | met |', '|---|---|']
        L += [f'| {k} | {"yes" if v else "no"} |' for k, v in res['conditions'].items()]
        a, w = res['arms']['control'], res['arms']['winner']
        L += ['', f'## Test, {len(res["seeds"])} seeds per arm (mean)', '',
              '| arm | acc | weighted F1 | macro-F1 (SD) | ' + ' | '.join(f'F1 {c}' for c in CLASSES) + ' |',
              '|---' * (4 + C) + '|']
        for name, m in (('control (A0)', a), (f'winner ({res["winner"]["name"]})', w)):
            L.append(f'| {name} | {m["acc"]:.2f} | {m["weighted_f1"]:.2f} | {m["macro_f1"]:.2f} ({m["macro_f1_sd"]:.2f}) | '
                     + ' | '.join(f'{x:.2f}' for x in m['f1']) + ' |')
        g, mu = res['guardrails'], res['multiplicity']
        L += ['', f'- Delta macro-F1: **{P._fmt(res["delta_macro_f1"], *res["ci95_hierarchical"])}** (hierarchical '
                  f'bootstrap); seed-level t-interval [{res["ci95_seed_t"][0]:+.2f}, {res["ci95_seed_t"][1]:+.2f}]',
              f'- third-look adjustment: {mu["level"]:.2f}% interval [{mu["ci_hierarchical"][0]:+.2f}, '
              f'{mu["ci_hierarchical"][1]:+.2f}], lower bound above 0: {"yes" if mu["lower_bound_above_0"] else "no"}',
              f'- positive in {res["seeds_positive"]}/{len(res["seeds"])} seeds: '
              + ', '.join(f'{x:+.2f}' for x in res['per_seed_delta']),
              f'- Delta accuracy {P._fmt(g["delta_acc"], *g["delta_acc_ci95"])}; '
              f'Delta weighted F1 {P._fmt(g["delta_weighted_f1"], *g["delta_weighted_f1_ci95"])}',
              '', '| metric (winner - A0, paired by seed) | mean | seed-level 95% t-interval | seeds positive |',
              '|---|---|---|---|']
        for name, c in res['consistency'].items():
            L.append(f'| {name} | {c["mean"]:+.2f} | [{c["seed_t_ci95"][0]:+.2f}, {c["seed_t_ci95"][1]:+.2f}] | '
                     f'{c["seeds_positive"]}/{c["seeds"]} |')
        L += ['', '| class | Delta F1 [95% CI] | Delta precision | Delta recall | support |', '|---|---|---|---|---|']
        pc = res['per_class']
        for c in range(C):
            L.append(f'| {CLASSES[c]} | {P._fmt(pc["delta_f1"][c], *pc["delta_f1_ci95"][c])} | '
                     f'{pc["delta_precision"][c]:+.2f} | {pc["delta_recall"][c]:+.2f} | {int(res["support"][c])} |')
        L += ['', f'- control weighted F1 {res["fidelity"]["control_weighted_f1"]:.2f} vs paper {P.PAPER_TEST_WF1}']
    warn = [c for c in S['checks'] if not c['passed']]
    L += ['', f'Checks: {sum(c["passed"] for c in S["checks"])} passed, {len(warn)} warnings'
          + ('' if not warn else ' -- ' + '; '.join(f'{c["name"]} ({c["detail"]})' for c in warn))]
    with open(os.path.join(S['out'], 'SUMMARY_audio.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')


def stage_package(S):
    zpath = os.path.join(S['cfg']['work_dir'], 'ecerc_l2_audio_results.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(S['out']):
            for name in files:
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, os.path.dirname(S['out'])))
    with open(os.path.join(S['out'], 'SUMMARY_audio.md')) as f:
        print(f.read())
    print(f'Results archive: {zpath} ({os.path.getsize(zpath) / 2 ** 20:.1f} MB)')
    return zpath
