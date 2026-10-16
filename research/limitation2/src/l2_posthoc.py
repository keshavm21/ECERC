"""
ECERC Limitation 2: post-hoc logit adjustment on the released MELD checkpoints.

New code for research/limitation2. The original ECERC files (model.py,
dataloader.py, inference.py, loss.py) are imported unchanged from verbatim
copies.

The notebook runs the stages in this order:
    stage_setup -> stage_download -> stage_load_data -> stage_fidelity
    -> stage_dev_analysis -> stage_freeze -> stage_test -> stage_package

Test-set rule: nothing before stage_freeze runs the model on test dialogues or
reads test labels, and stage_test takes every decision rule from
frozen_config.json.
"""
import csv
import hashlib
import importlib
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import time
import urllib.request
import zipfile
from types import SimpleNamespace

import numpy as np

# ----------------------------------------------------------------------------- constants
CLASSES = ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang']  # label order used by the MELD code
C = len(CLASSES)
NEU, SUR, FEA, SAD, JOY, DIS, ANG = range(C)
OFFICIAL_TRAIN_COUNTS = [4710, 1205, 268, 683, 1743, 271, 1109]  # MELD paper (Poria et al., 2019), train split
PAPER_TEST_WF1 = 66.46  # ECERC paper, Table 3 (3-run mean); used only for a post-freeze fidelity flag
FIELDS = ['speakers', 'emotion_labels', 'sentiment_labels',
          'eroberta1', 'eroberta2', 'eroberta3', 'eroberta4',
          'sroberta1', 'sroberta2', 'sroberta3', 'sroberta4',
          'videoAudio', 'videoVisual', 'sentences', 'train_ids', 'test_ids', 'valid_ids']
TAU_GRID = [round(0.1 * i, 2) for i in range(21)]     # 0.0 ... 2.0
DELTA_GRID = [round(0.15 * i, 2) for i in range(21)]  # 0.00 ... 3.00
RULES = [('tau', v) for v in TAU_GRID] + [('delta', v) for v in DELTA_GRID]
R = len(RULES)
TAU_IDX = np.arange(len(TAU_GRID))
DELTA_IDX = np.arange(len(TAU_GRID), R)
BASE_RULE = 0  # tau = 0 is the plain argmax of the released model
LA1_RULE = TAU_GRID.index(1.0)
BRANCHES = ['self_contagion', 'cross_emotion', 'self_event', 'cross_event']  # smax_fc input blocks, in forward() order
PAIRS = [(FEA, NEU), (FEA, SUR), (FEA, SAD), (FEA, ANG), (DIS, NEU), (DIS, ANG), (SAD, NEU)]
CRITERIA = dict(
    primary='mean over included checkpoints of test macro-F1(LA*) - macro-F1(BASE), in points',
    success=['delta >= +1.0',
             'procedure-level 95% CI lower bound > 0',
             'mean F1 change > 0 for Fear and for Disgust',
             'accuracy change >= -1.0',
             'weighted-F1 change >= -0.5',
             'every class: mean F1 change >= -2.0',
             'delta > 0 for every included checkpoint'],
    verdict_order=['SUCCESS', 'TRADE-OFF (efficacy met, a guardrail fails)',
                   'PARTIAL (macro-F1 target met, Fear/Disgust/consistency not)',
                   'FAILURE (CI upper bound < +1.0)', 'INCONCLUSIVE'],
    guardrails_judged_on='point estimates; 95% CIs reported',
)
TRUST_HINT = ('If you trust the source (the official ECERC Google Drive link), set TRUST_DATA_PICKLE = True in the '
              'first code cell and run all cells again. The Colab VM is isolated and Google Drive is not mounted.')
DATA_FALLBACK = ('Automatic download from Google Drive failed. Manual fallback: open '
                 'https://drive.google.com/file/d/{id}/view , download the file, upload it with the Colab Files '
                 'panel into {dir} (keep the name {name}), then run all cells again.')


# ----------------------------------------------------------------------------- utilities
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(_jsonable(obj), f, indent=2)


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in _jsonable(r).items()})


def log_event(S, stage, **info):
    rec = dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), stage=stage, **info)
    with open(os.path.join(S['out'], 'run_log.jsonl'), 'a') as f:
        f.write(json.dumps(_jsonable(rec)) + '\n')
    print(f'== {stage} ' + ' '.join(f'{k}={v}' for k, v in info.items()))


class CheckFailed(RuntimeError):
    """A pre-registered hard check failed; the run stops before any test evaluation."""


def record_check(S, name, passed, hard, detail=''):
    S['checks'].append(dict(name=name, passed=bool(passed), hard=hard, detail=str(detail)))
    status = 'PASS' if passed else ('FAIL' if hard else 'WARN')
    print(f'  [{status}] {name}' + (f' -- {detail}' if detail != '' else ''))
    if hard and not passed:
        _save_checks(S)
        raise CheckFailed(f'{name}: {detail}')


def _save_checks(S):
    write_json(os.path.join(S['out'], 'checks.json'), S['checks'])


def _div(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    shape = np.broadcast_shapes(a.shape, b.shape)
    return np.divide(np.broadcast_to(a, shape), np.broadcast_to(b, shape), out=np.zeros(shape),
                     where=np.broadcast_to(b, shape) > 0)


def _fmt(est, lo=None, hi=None):
    if lo is None:
        return f'{est:+.2f}'
    return f'{est:+.2f} [{lo:+.2f}, {hi:+.2f}]'


# ----------------------------------------------------------------------------- safe data loading
_SAFE_GLOBALS = {
    ('collections', 'OrderedDict'), ('collections', 'defaultdict'),
    ('builtins', 'set'), ('builtins', 'frozenset'), ('_codecs', 'encode'),
    ('numpy', 'ndarray'), ('numpy', 'dtype'),
    ('numpy.core.multiarray', '_reconstruct'), ('numpy.core.multiarray', 'scalar'),
    ('numpy._core.multiarray', '_reconstruct'), ('numpy._core.multiarray', 'scalar'),
    ('numpy.core.numeric', '_frombuffer'), ('numpy._core.numeric', '_frombuffer'),
}


class _SafeUnpickler(pickle.Unpickler):
    """Only NumPy arrays and plain containers can be rebuilt, so the data file cannot run code."""

    def find_class(self, module, name):
        if (module, name) in _SAFE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'blocked global {module}.{name}')


def load_pickle(path, trust=False):
    with open(path, 'rb') as f:
        if trust:
            return pickle.load(f, encoding='latin1')
        return _SafeUnpickler(f, encoding='latin1').load()


def _dim(x):
    return int(np.asarray(x).shape[-1])


# ----------------------------------------------------------------------------- stage 1: setup
def install_cpu_cuda_shim(torch):
    """ECERC.forward builds its attention masks only when cuda_flag=True and then calls .cuda() on them.
    Without a GPU, make Tensor.cuda() return the tensor itself so the unchanged forward runs on CPU."""
    torch.Tensor.cuda = lambda self, *args, **kwargs: self


def stage_setup(CONFIG, code_dir):
    out = os.path.join(CONFIG['work_dir'], 'outputs')
    for sub in ('', 'tables', 'figures', 'dumps'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    S = dict(cfg=CONFIG, out=out, code_dir=code_dir, checks=[])
    import pandas
    import sklearn
    import torch
    S['torch'] = torch
    cuda = torch.cuda.is_available()
    if not cuda:
        install_cpu_cuda_shim(torch)
    S['device'] = 'cuda' if cuda else 'cpu'
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    for name in ('model', 'dataloader', 'loss', 'inference'):  # the unchanged original ECERC modules
        S['m_' + name] = importlib.import_module(name)
    S['env'] = dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                    numpy=np.__version__, pandas=pandas.__version__, sklearn=sklearn.__version__,
                    device=S['device'], gpu=torch.cuda.get_device_name(0) if cuda else None,
                    cpu_cuda_shim=not cuda,
                    code_sha256={f: sha256_file(os.path.join(code_dir, f))
                                 for f in sorted(os.listdir(code_dir)) if f.endswith('.py')})
    write_json(os.path.join(out, 'environment.json'), S['env'])
    log_event(S, 'setup', device=S['device'], gpu=S['env']['gpu'], torch=torch.__version__)
    return S


# ----------------------------------------------------------------------------- stage 2: downloads
def _fetch(url, dest, expected_sha256, retries=3):
    if os.path.exists(dest):
        got = sha256_file(dest)
        if got != expected_sha256:
            raise CheckFailed(f'{dest} exists with an unexpected SHA-256 ({got}); not overwriting it.')
        return 'already present'
    tmp = dest + '.part'
    for attempt in range(1, retries + 1):
        try:
            urllib.request.urlretrieve(url, tmp)
            break
        except Exception as e:  # transient network error
            if attempt == retries:
                raise
            print(f'  retry {attempt} after error: {e}')
            time.sleep(5)
    got = sha256_file(tmp)
    if got != expected_sha256:
        os.remove(tmp)
        raise CheckFailed(f'SHA-256 mismatch for {url}: {got}')
    os.replace(tmp, dest)
    return 'downloaded'


def stage_download(S):
    cfg = S['cfg']
    ck_dir = os.path.join(cfg['work_dir'], 'checkpoints')
    os.makedirs(ck_dir, exist_ok=True)
    S['ckpts'] = []
    for ck in cfg['checkpoints']:
        if cfg.get('offline'):
            path, status = ck['local_path'], 'local file'
        else:
            path = os.path.join(ck_dir, os.path.basename(ck['url']))
            status = _fetch(ck['url'], path, ck['sha256'])
        got = sha256_file(path)
        record_check(S, f'{ck["name"]} SHA-256 matches the HuggingFace release', got == ck['sha256'], hard=True,
                     detail=status)
        with zipfile.ZipFile(path) as z:
            archive = z.namelist()[0].split('/')[0]
        S['ckpts'].append(dict(name=ck['name'], path=path, sha256=got, archive=archive,
                               flagged_test_selection='test' in archive.lower()))
        print(f'  {ck["name"]}: embedded archive name "{archive}"')

    data_dir = os.path.join(cfg['work_dir'], 'data')
    os.makedirs(data_dir, exist_ok=True)
    path = cfg.get('data_local_path') or os.path.join(data_dir, cfg['data_filename'])
    if os.path.exists(path) and os.path.getsize(path) < 1_000_000:  # leftover HTML error page from a failed download
        os.remove(path)
    if not os.path.exists(path):
        if cfg.get('offline'):
            raise CheckFailed(f'offline run but {path} is missing')
        try:
            import gdown
        except ImportError:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)
            import gdown
        try:
            gdown.download(id=cfg['data_file_id'], output=path, quiet=False)
        except Exception as e:
            print('  gdown error:', e)
        if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
            raise CheckFailed(DATA_FALLBACK.format(id=cfg['data_file_id'], dir=data_dir, name=cfg['data_filename']))
    S['data_path'] = path
    S['data_bytes'] = os.path.getsize(path)
    S['data_sha256'] = sha256_file(path)
    write_json(os.path.join(S['out'], 'provenance.json'),
               dict(checkpoints=[{k: v for k, v in c.items() if k != 'path'} for c in S['ckpts']],
                    data=dict(file=cfg['data_filename'], google_drive_id=cfg['data_file_id'],
                              bytes=S['data_bytes'], sha256=S['data_sha256'])))
    _save_checks(S)
    log_event(S, 'downloads verified', data_MB=round(S['data_bytes'] / 2 ** 20, 1),
              data_sha256=S['data_sha256'][:16])


# ----------------------------------------------------------------------------- stage 3: data gate
def stage_load_data(S):
    cfg = S['cfg']
    t0 = time.time()
    try:
        obj = load_pickle(S['data_path'], cfg.get('trust_data_pickle', False))
    except pickle.UnpicklingError as e:
        raise CheckFailed(f'Safe loader refused the data file ({e}). {TRUST_HINT}') from None
    ok = isinstance(obj, (tuple, list)) and len(obj) == len(FIELDS)
    record_check(S, 'data file has the 17 fields read by MELD/dataloader.py', ok, hard=True,
                 detail=f'type={type(obj).__name__}')
    D = dict(zip(FIELDS, obj))
    S['D'] = D
    keys = {'train': list(D['train_ids']), 'valid': list(D['valid_ids']), 'test': list(D['test_ids'])}
    S['keys'] = keys
    tr, dv, te = (set(keys[s]) for s in ('train', 'valid', 'test'))
    record_check(S, 'dialogue IDs are disjoint across splits', not (tr & dv or tr & te or dv & te), hard=True)
    # Sizes come from feature lengths: test labels are not read before the freeze.
    n_utt = {s: int(sum(len(D['videoAudio'][v]) for v in ks)) for s, ks in keys.items()}
    n_dlg = {s: len(ks) for s, ks in keys.items()}
    exp_u, exp_d = cfg.get('expected_split_utterances'), cfg.get('expected_split_dialogues')
    if exp_u:
        record_check(S, 'utterance counts per split match MELD', n_utt == exp_u, hard=False, detail=n_utt)
    if exp_d:
        ok = all(n_dlg[s] in (exp_d[s] if isinstance(exp_d[s], (list, tuple)) else [exp_d[s]]) for s in n_dlg)
        record_check(S, 'dialogue counts per split match MELD', ok, hard=False, detail=n_dlg)
    v0 = keys['train'][0]
    dims = {f: _dim(D[f][v0]) for f in ('eroberta1', 'sroberta1', 'videoVisual', 'videoAudio', 'speakers')}
    ok = (dims['eroberta1'] == 1024 and dims['sroberta1'] == 1024 and dims['speakers'] == 9
          and dims['videoVisual'] + dims['videoAudio'] == 642)
    record_check(S, 'feature dimensions fit the released model (text 1024, audio+visual 642, 9 speakers)', ok,
                 hard=True, detail=dims)
    y_tr = np.concatenate([np.asarray(D['emotion_labels'][v], dtype=np.int64) for v in keys['train']])
    y_dv = np.concatenate([np.asarray(D['emotion_labels'][v], dtype=np.int64) for v in keys['valid']])
    record_check(S, 'train/dev labels are in 0..6',
                 y_tr.min() >= 0 and y_tr.max() < C and y_dv.min() >= 0 and y_dv.max() < C, hard=True)
    record_check(S, 'train/dev label lengths match feature lengths',
                 all(len(D['emotion_labels'][v]) == len(D['videoAudio'][v]) for s in ('train', 'valid')
                     for v in keys[s]), hard=True)
    counts = np.bincount(y_tr, minlength=C)
    record_check(S, 'train label counts match the MELD paper', counts.tolist() == OFFICIAL_TRAIN_COUNTS,
                 hard=False, detail=counts.tolist())
    S['train_counts'] = counts
    S['prior'] = counts / counts.sum()
    S['log_prior'] = np.log(S['prior'])
    S['dev_counts'] = np.bincount(y_dv, minlength=C)
    write_json(os.path.join(S['out'], 'data_gate.json'),
               dict(data_sha256=S['data_sha256'], data_bytes=S['data_bytes'], n_utterances=n_utt, n_dialogues=n_dlg,
                    feature_dims=dims,
                    modality_note=('MELD/dataloader.py returns (videoVisual, videoAudio) and MELD/inference.py '
                                   'concatenates them as [text, audio, vision], i.e. [text, videoVisual, videoAudio]. '
                                   'Kept unchanged because the released weights were trained this way.'),
                    train_counts=counts, prior_train=S['prior'], dev_counts=S['dev_counts'],
                    load_seconds=round(time.time() - t0, 1)))
    _save_checks(S)
    log_event(S, 'data loaded', seconds=round(time.time() - t0, 1), dialogues=n_dlg)


# ----------------------------------------------------------------------------- model and inference
def make_dataset(S, split):
    """The original MELDRobertaDataset with only __init__ replaced: the original re-reads the pickle from
    '../data/meld/...' for every split, here the already-loaded data is attached. __getitem__ and
    collate_fn are the unchanged originals."""
    D = S['D']

    class MELDSplit(S['m_dataloader'].MELDRobertaDataset):
        def __init__(self, split_name):
            for field in FIELDS:
                setattr(self, field, D[field])
            ids = {'train': self.train_ids, 'valid': self.valid_ids, 'test': self.test_ids}[split_name]
            self.keys = [x for x in ids]
            self.len = len(self.keys)

    return MELDSplit(split)


def build_model(S):
    """Same constructor arguments as MELD/inference.py (MELD, feature_type='multi', base_layer=1).
    cuda_flag=True so the unchanged forward builds every mask (see install_cpu_cuda_shim for CPU runs)."""
    args = SimpleNamespace(feature_type='multi', Dataset='MELD', base_layer=1)
    return S['m_model'].ECERC(args, d_t=1024, d_a=300, d_v=342, base_layer=1, input_size=1024 + 300 + 342,
                              hidden_size=128, n_speakers=9, n_classes=C, cuda_flag=True)


def run_inference(S, model, dataset, capture=False, batch_size=32):
    """Mirror of MELD/inference.py::train_or_eval_model (eval mode, feature_type='multi') that also keeps the
    per-utterance log-probabilities, dialogue index, speaker and, optionally, the smax_fc input features."""
    torch = S['torch']
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn, num_workers=0)
    store, handle = {}, None
    if capture:
        handle = model.smax_fc.register_forward_hook(lambda mod, inp, out: store.__setitem__('x', inp[0].detach()))
    logps, ys, dlgs, poss, spks, feats, vids = [], [], [], [], [], [], []
    base = 0
    model.eval()
    try:
        with torch.no_grad():
            for data in loader:
                emo, sem, audio, vision, qmask, umask, label = [d.to(S['device']) for d in data[:-1]]
                seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
                emo = torch.cat([emo, audio, vision], dim=-1)  # as in MELD/inference.py
                sem = torch.cat([sem], dim=-1)
                log_prob = model(emo, sem, qmask, umask, seq_lengths)
                lab = torch.cat([label[j][:seq_lengths[j]] for j in range(len(label))])
                logps.append(log_prob.float().cpu().numpy())
                ys.append(lab.cpu().numpy())
                for j, n in enumerate(seq_lengths):
                    dlgs.append(np.full(n, base + j))
                    poss.append(np.arange(n))
                    spks.append(qmask[:n, j, :].argmax(-1).cpu().numpy())
                    if capture:
                        feats.append(store['x'][j, :n, :].float().cpu().numpy())
                vids.extend(list(data[-1]))
                base += len(seq_lengths)
    finally:
        if handle is not None:
            handle.remove()
    out = dict(logp=np.concatenate(logps).astype(np.float64), y=np.concatenate(ys).astype(np.int64),
               dlg=np.concatenate(dlgs), pos=np.concatenate(poss), spk=np.concatenate(spks), vids=vids)
    if capture:
        out['feat'] = np.concatenate(feats)
    assert vids == list(dataset.keys) and len(out['y']) == len(out['logp']) == len(out['dlg'])
    return out


def original_function_eval(S, model, dataset):
    """Run the unchanged MELD/inference.py::train_or_eval_model on the same dataset."""
    from torch.utils.data import DataLoader
    inf = S['m_inference']
    inf.args = SimpleNamespace(feature_type='multi')  # the original function reads a module-level `args`
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=dataset.collate_fn, num_workers=0)
    res = inf.train_or_eval_model(model=model, loss_f=S['m_loss'].Loss(alpha=None), dataloader=loader,
                                  cuda_flag=(S['device'] == 'cuda'), feature_type='multi', target_names=CLASSES)
    return float(res[1]), float(res[2]), np.asarray(res[3][1])


def save_dump(S, split, name, d):
    np.savez_compressed(os.path.join(S['out'], 'dumps', f'{split}_{name}.npz'), logp=d['logp'].astype(np.float32),
                        y=d['y'], dlg=d['dlg'], pos=d['pos'], spk=d['spk'], vids=np.array([str(v) for v in d['vids']]))


# ----------------------------------------------------------------------------- metrics and decision rules
def scores(cm):
    """Metrics from confusion counts cm[..., true, pred] with any leading batch dimensions."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    sup, prd = cm.sum(-1), cm.sum(-2)
    n = sup.sum(-1)
    f1, prec, rec = _div(2 * tp, sup + prd), _div(tp, prd), _div(tp, sup)
    return dict(acc=_div(tp.sum(-1), n), mf1=f1.mean(-1), wf1=_div((f1 * sup).sum(-1), n), bacc=rec.mean(-1),
                f1=f1, prec=prec, rec=rec, support=sup, pred=prd)


def rule_scores(logp, log_prior, rule):
    kind, v = rule
    if kind == 'tau':  # logit adjustment: subtract tau * log(train prior)
        return logp - v * log_prior[None, :]
    s = logp.copy()  # Neutral-only offset
    s[:, NEU] -= v
    return s


def all_rule_preds(logp, log_prior):
    return np.stack([rule_scores(logp, log_prior, r).argmax(1) for r in RULES])  # (R, N)


def per_dialogue_cms(y, preds, dlg, n_dlg):
    """(n_dlg, R, 49) confusion counts for every dialogue and decision rule."""
    nr = preds.shape[0]
    idx = ((dlg[None, :] * nr + np.arange(nr)[:, None]) * (C * C) + y[None, :] * C + preds).ravel()
    return np.bincount(idx, minlength=n_dlg * nr * C * C).reshape(n_dlg, nr, C * C)


def split_cms(dumps, log_prior, n_dlg):
    """(n_dlg, K, R, 49) for per-checkpoint dumps of one split."""
    return np.stack([per_dialogue_cms(d['y'], all_rule_preds(d['logp'], log_prior), d['dlg'], n_dlg)
                     for d in dumps], axis=1)


def first_argmax(x, axis=-1, tol=1e-12):
    """Index of the first maximum along `axis`, so ties go to the smaller grid value."""
    return np.argmax(x >= x.max(axis=axis, keepdims=True) - tol, axis=axis)


def select_rule(tot, idx, include, row_weights=None):
    """Rule in `idx` with the highest mean macro-F1 over included checkpoints. tot: (K, R, 49) counts."""
    cm = tot.reshape(tot.shape[0], tot.shape[1], C, C).astype(float)
    if row_weights is not None:
        cm = cm * row_weights[None, None, :, None]
    mean = scores(cm)['mf1'][include].mean(0)
    return int(idx[int(first_argmax(mean[idx]))]), mean


def _boot_weights(rng, n, b):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(float)


def crossfit_forecast(dev_cms, include, n_folds, seed, B, n_test_dlg):
    """Honest dev estimate of the LA gain (tau chosen without the fold it is applied to) and a projected
    test standard error from a dev bootstrap of that cross-fitted gain."""
    n_dev, K = dev_cms.shape[:2]
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n_dev), n_folds)
    cf = np.zeros((n_dev, K, C * C))
    chosen = []
    for f in folds:
        keep = np.ones(n_dev, bool)
        keep[f] = False
        r, _ = select_rule(dev_cms[keep].sum(0), TAU_IDX, include)
        chosen.append(RULES[r][1])
        cf[f] = dev_cms[f][:, :, r, :]
    base = dev_cms[:, :, BASE_RULE, :].astype(float)

    def delta(w):
        a = scores((w @ cf.reshape(n_dev, -1)).reshape(-1, K, C, C))['mf1']
        b = scores((w @ base.reshape(n_dev, -1)).reshape(-1, K, C, C))['mf1']
        return 100 * (a - b)[:, include].mean(1)

    point = float(delta(np.ones((1, n_dev)))[0])
    boots = delta(_boot_weights(rng, n_dev, B))
    se_dev = float(boots.std(ddof=1))
    se_test = se_dev * math.sqrt(n_dev / n_test_dlg)
    return dict(fold_tau=chosen, crossfit_delta_macro_f1=point, dev_bootstrap_ci95=np.percentile(boots, [2.5, 97.5]),
                se_dev=se_dev, se_test_projected=se_test, mde80_projected=2.8 * se_test,
                note='projected test SE = dev SE * sqrt(dev dialogues / test dialogues); MDE80 = 2.8 * SE')


def procedure_bootstrap(dev_cms, test_cms, include, B, seed, chunk=500):
    """Two-sample dialogue bootstrap: resample dev dialogues and re-select tau / delta with the frozen rule,
    resample test dialogues, recompute every difference against BASE. Returns per-replicate arrays."""
    n_dev, K = dev_cms.shape[:2]
    n_te = test_cms.shape[0]
    rng = np.random.default_rng(seed)
    dflat = dev_cms.reshape(n_dev, -1).astype(float)
    tflat = test_cms.reshape(n_te, -1).astype(float)
    keys = ['d_mf1', 'per_ckpt', 'd_acc', 'd_wf1', 'd_f1', 'd_prec', 'd_rec', 'd_mf1_no', 'mech', 'd_mf1_la1',
            'tau_sel', 'delta_sel']
    out = {k: [] for k in keys}
    taus, deltas = np.array([v for _, v in RULES])[TAU_IDX], np.array([v for _, v in RULES])[DELTA_IDX]
    done = 0
    while done < B:
        b = min(chunk, B - done)
        dsc = scores((_boot_weights(rng, n_dev, b) @ dflat).reshape(b, K, R, C, C))
        tsc = scores((_boot_weights(rng, n_te, b) @ tflat).reshape(b, K, R, C, C))
        dmean = dsc['mf1'][:, include, :].mean(1)  # (b, R)
        ti = first_argmax(dmean[:, TAU_IDX], axis=1)
        di = first_argmax(dmean[:, DELTA_IDX], axis=1)
        tsel, dsel = TAU_IDX[ti], DELTA_IDX[di]
        rows, cols = np.arange(b)[:, None], np.arange(K)[None, :]

        def at(x, sel):
            return x[rows, cols, sel[:, None]]

        base = {k: tsc[k][:, :, BASE_RULE] for k in ('mf1', 'acc', 'wf1', 'f1', 'prec', 'rec')}
        la_mf1 = at(tsc['mf1'], tsel)
        d_mf1 = 100 * (la_mf1 - base['mf1'])[:, include].mean(1)
        d_no = 100 * (at(tsc['mf1'], dsel) - base['mf1'])[:, include].mean(1)
        out['d_mf1'].append(d_mf1)
        out['per_ckpt'].append(100 * (la_mf1 - base['mf1']))
        out['d_acc'].append(100 * (at(tsc['acc'], tsel) - base['acc'])[:, include].mean(1))
        out['d_wf1'].append(100 * (at(tsc['wf1'], tsel) - base['wf1'])[:, include].mean(1))
        for key, src in (('d_f1', 'f1'), ('d_prec', 'prec'), ('d_rec', 'rec')):
            out[key].append(100 * (at(tsc[src], tsel) - base[src])[:, include, :].mean(1))
        out['d_mf1_no'].append(d_no)
        out['mech'].append(d_mf1 - d_no)
        out['d_mf1_la1'].append(100 * (tsc['mf1'][:, :, LA1_RULE] - base['mf1'])[:, include].mean(1))
        out['tau_sel'].append(taus[ti])
        out['delta_sel'].append(deltas[di])
        done += b
    return {k: np.concatenate(v) for k, v in out.items()}


def fixed_rule_bootstrap(test_cms, include, rules_per_ckpt, B, seed, chunk=1000):
    """Test-only dialogue bootstrap of the macro-F1 difference for fixed (already frozen) rules."""
    n, K = test_cms.shape[:2]
    rng = np.random.default_rng(seed)
    flat = test_cms.reshape(n, -1).astype(float)
    sel = np.asarray(rules_per_ckpt)
    res, done = [], 0
    while done < B:
        b = min(chunk, B - done)
        mf1 = scores((_boot_weights(rng, n, b) @ flat).reshape(b, K, R, C, C))['mf1']
        res.append(100 * (mf1[:, np.arange(K), sel] - mf1[:, :, BASE_RULE])[:, include].mean(1))
        done += b
    return np.concatenate(res)


def frontier_rows(tot, include, split):
    sc = scores(tot.reshape(tot.shape[0], R, C, C).astype(float))
    rows = []
    for r in TAU_IDX:
        row = dict(split=split, tau=RULES[r][1], acc=100 * sc['acc'][include, r].mean(),
                   macro_f1=100 * sc['mf1'][include, r].mean(), weighted_f1=100 * sc['wf1'][include, r].mean())
        for c in range(C):
            row[f'F1_{CLASSES[c]}'] = 100 * sc['f1'][include, r, c].mean()
            row[f'P_{CLASSES[c]}'] = 100 * sc['prec'][include, r, c].mean()
            row[f'R_{CLASSES[c]}'] = 100 * sc['rec'][include, r, c].mean()
        rows.append(row)
    return rows


def plot_frontier(S, rows, split, tau_star):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:  # plotting is optional
        print('  (plot skipped:', e, ')')
        return
    taus = [r['tau'] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for c in CLASSES:
        ax[0].plot(taus, [r[f'F1_{c}'] for r in rows], marker='.', label=c)
    ax[0].plot(taus, [r['macro_f1'] for r in rows], 'k-', lw=2.5, label='macro-F1')
    ax[0].axvline(tau_star, color='gray', ls='--')
    ax[0].set(xlabel='tau', ylabel='F1 (%)', title=f'{split}: F1 vs adjustment strength (mean over checkpoints)')
    ax[0].legend(fontsize=8, ncol=2)
    for c in ('fea', 'dis', 'sad'):
        ax[1].plot([r[f'R_{c}'] for r in rows], [r[f'P_{c}'] for r in rows], marker='o', ms=3, label=c)
    ax[1].set(xlabel='recall (%)', ylabel='precision (%)', title=f'{split}: precision vs recall as tau grows')
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(S['out'], 'figures', f'{split}_frontier.png'), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- diagnostics (dev only)
def _report(y, p):
    sc = scores(np.bincount(y * C + p, minlength=C * C).reshape(C, C))
    return dict(acc=100 * sc['acc'], macro_f1=100 * sc['mf1'], recall=100 * sc['rec'], f1=100 * sc['f1'])


def _recall(y, p):
    return np.array([((p == c) & (y == c)).sum() / max((y == c).sum(), 1) for c in range(C)])


def utter_matrix(S, split, field='eroberta1'):
    assert split in ('train', 'valid'), 'diagnostics never read test labels'
    D, keys = S['D'], S['keys'][split]
    X = np.concatenate([np.asarray(D[field][v], dtype=np.float32) for v in keys])
    y = np.concatenate([np.asarray(D['emotion_labels'][v], dtype=np.int64) for v in keys])
    return X, y


def d1_separability(S, include):
    from sklearn.metrics import average_precision_score, roc_auc_score
    per = []
    for d, inc in zip(S['dev_dump'], include):
        if not inc:
            continue
        y, lp = d['y'], d['logp']
        p = np.exp(lp)
        auroc = [roc_auc_score(y == c, p[:, c]) if 0 < (y == c).sum() < len(y) else np.nan for c in range(C)]
        ap = [average_precision_score(y == c, p[:, c]) if (y == c).any() else np.nan for c in range(C)]
        pw = []
        for a, b in PAIRS:
            m = (y == a) | (y == b)
            ok = (y[m] == a).any() and (y[m] == b).any()
            pw.append(roc_auc_score(y[m] == a, lp[m, a] - lp[m, b]) if ok else np.nan)
        per.append(dict(auroc=auroc, ap=ap, pairwise=pw))
    y = S['dev_dump'][0]['y']
    return dict(classes=CLASSES, prevalence=np.bincount(y, minlength=C) / len(y),
                ovr_auroc=np.nanmean([q['auroc'] for q in per], 0), ovr_ap=np.nanmean([q['ap'] for q in per], 0),
                pairs=[f'{CLASSES[a]} vs {CLASSES[b]}' for a, b in PAIRS],
                pairwise_auroc=np.nanmean([q['pairwise'] for q in per], 0), per_checkpoint=per)


def d3_train_vs_dev(S, include):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ck = [dict(name=c['name'], train=_report(dtr['y'], dtr['logp'].argmax(1)),
               dev=_report(ddv['y'], ddv['logp'].argmax(1)))
          for c, dtr, ddv, inc in zip(S['ckpts'], S['train_dump'], S['dev_dump'], include) if inc]
    Xtr, ytr = utter_matrix(S, 'train')
    Xdv, ydv = utter_matrix(S, 'valid')
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    t0 = time.time()
    clf.fit(scaler.transform(Xtr), ytr)
    probe = dict(features='eroberta1 only (text emotion features, no context, no audio/visual)', C=1.0,
                 n_iter=int(np.max(clf.n_iter_)), seconds=round(time.time() - t0, 1),
                 train=_report(ytr, clf.predict(scaler.transform(Xtr))),
                 dev=_report(ydv, clf.predict(scaler.transform(Xdv))))
    return dict(checkpoints=ck, roberta_probe=probe)


def branch_decomposition(S, include):
    """Exact split of a logit margin into the four cause branches: smax_fc is linear over their concatenation."""
    out = {}
    for c in (FEA, DIS, SAD, ANG, SUR, JOY):
        per = []
        for d, model, inc in zip(S['dev_dump'], S['models'], include):
            if not inc:
                continue
            W = model.smax_fc.weight.detach().cpu().double().numpy()
            bias = model.smax_fc.bias.detach().cpu().double().numpy()
            x = d['feat'].astype(np.float64)
            pred = d['logp'].argmax(1)
            blk = W.shape[1] // len(BRANCHES)
            for group, sel, hi, lo in (('missed_as_neutral', (d['y'] == c) & (pred == NEU), NEU, c),
                                       ('correct', (d['y'] == c) & (pred == c), c, NEU)):
                if not sel.any():
                    continue
                contrib = [float(((W[hi, j * blk:(j + 1) * blk] - W[lo, j * blk:(j + 1) * blk])
                                  * x[sel, j * blk:(j + 1) * blk]).sum(1).mean()) for j in range(len(BRANCHES))]
                actual = float((d['logp'][sel, hi] - d['logp'][sel, lo]).mean())
                b = float(bias[hi] - bias[lo])
                per.append(dict(group=group, n=int(sel.sum()), margin=actual, bias=b,
                                reconstruction_error=abs(sum(contrib) + b - actual), **dict(zip(BRANCHES, contrib))))
        agg = {}
        for group in ('missed_as_neutral', 'correct'):
            rows = [p for p in per if p['group'] == group]
            if rows:
                agg[group] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != 'group'}
        out[CLASSES[c]] = agg
    return dict(note=('margin = logit(first class) - logit(second class), averaged per checkpoint then over '
                      'checkpoints. missed_as_neutral: Neutral minus true class for items predicted Neutral; '
                      'correct: true class minus Neutral for correctly predicted items.'), classes=out)


def d5_context(S, include):
    d0 = S['dev_dump'][0]
    y, dlg, spk = d0['y'], d0['dlg'], d0['spk']
    n = len(y)
    stratum = np.empty(n, dtype=object)
    prev_any = np.full(n, -1)
    last_spk, last_dlg = {}, {}
    for i in range(n):
        key = (int(dlg[i]), int(spk[i]))
        stratum[i] = 'first' if key not in last_spk else ('same' if y[last_spk[key]] == y[i] else 'shift')
        if int(dlg[i]) in last_dlg:
            prev_any[i] = y[last_dlg[int(dlg[i])]]
        last_spk[key] = i
        last_dlg[int(dlg[i])] = i
    preds = [d['logp'].argmax(1) for d, inc in zip(S['dev_dump'], include) if inc]
    strata = {}
    for st in ('first', 'same', 'shift'):
        m = stratum == st
        strata[st] = dict(n=int(m.sum()), class_counts=np.bincount(y[m], minlength=C),
                          recall=100 * np.mean([_recall(y[m], p[m]) for p in preds], 0) if m.any() else None,
                          acc=100 * float(np.mean([(p[m] == y[m]).mean() for p in preds])) if m.any() else None)
    has_prev = prev_any >= 0
    copying = []
    for p in preds:
        wrong = has_prev & (p != y)
        copying.append(dict(
            errors_with_a_previous_utterance=int(wrong.sum()),
            pct_errors_predicting_previous_label=100 * float((p[wrong] == prev_any[wrong]).mean()) if wrong.any() else None,
            pct_items_whose_label_equals_previous=100 * float((y[has_prev] == prev_any[has_prev]).mean())))
    return dict(shift_definition="label differs from the same speaker's previous label in the dialogue",
                strata=strata, context_copying=copying, branch_margins=branch_decomposition(S, include))


def d6_consistency(S, include):
    y = S['dev_dump'][0]['y']
    preds = np.stack([d['logp'].argmax(1) for d, inc in zip(S['dev_dump'], include) if inc])
    k = preds.shape[0]
    wrong = (preds != y[None, :]).sum(0)
    return dict(n_checkpoints=k, by_class={
        CLASSES[c]: dict(n=int((y == c).sum()), **{f'wrong_in_{j}_of_{k}': int(((wrong == j) & (y == c)).sum())
                                                   for j in range(k + 1)}) for c in range(C)})


def _log_softmax(z):
    m = z.max(1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(1, keepdims=True))


def _nll(lp, y, T):
    return float(-_log_softmax(lp / T)[np.arange(len(y)), y].mean())


def calib_stats(lp, y, T=1.0, bins=15):
    z = _log_softmax(lp / T)
    p = np.exp(z)
    conf, correct = p.max(1), p.argmax(1) == y
    ece = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return dict(nll=_nll(lp, y, T), ece_pct=100 * float(ece))


def calibration_dev(S):
    grid = np.round(np.arange(0.5, 5.0001, 0.01), 2)
    out, S['temperature'] = {}, {}
    for c, d in zip(S['ckpts'], S['dev_dump']):
        T = float(grid[int(np.argmin([_nll(d['logp'], d['y'], t) for t in grid]))])
        S['temperature'][c['name']] = T
        out[c['name']] = dict(T=T, raw=calib_stats(d['logp'], d['y']),
                              scaled_in_sample=calib_stats(d['logp'], d['y'], T))
    return out


# ----------------------------------------------------------------------------- stage 4: fidelity (dev only)
def stage_fidelity(S):
    from sklearn.metrics import confusion_matrix
    torch = S['torch']
    dev_ds = make_dataset(S, 'valid')
    S['models'], S['dev_dump'], S['train_dump'] = [], [], []
    report = []
    for ck in S['ckpts']:
        model = build_model(S)
        sd = torch.load(ck['path'], map_location=S['device'], weights_only=True)
        try:
            model.load_state_dict(sd, strict=True)
            err = ''
        except RuntimeError as e:
            err = str(e)[:300]
        record_check(S, f'{ck["name"]}: loads into the unchanged ECERC class (strict)', err == '', hard=True,
                     detail=err)
        model.to(S['device']).eval()
        d1 = run_inference(S, model, dev_ds, capture=True)
        d2 = run_inference(S, model, dev_ds)
        maxdiff = float(np.abs(d1['logp'] - d2['logp']).max())
        same = np.array_equal(d1['logp'].argmax(1), d2['logp'].argmax(1))
        record_check(S, f'{ck["name"]}: two dev passes agree', same and maxdiff <= 1e-5, hard=True,
                     detail=f'max |diff| = {maxdiff:.1e}')
        acc_o, wf1_o, cm_o = original_function_eval(S, model, dev_ds)
        cm_ours = confusion_matrix(d1['y'], d1['logp'].argmax(1))
        diff = int(np.abs(cm_o - cm_ours).sum()) if cm_o.shape == cm_ours.shape else -1
        record_check(S, f'{ck["name"]}: our loop reproduces the original train_or_eval_model on dev',
                     0 <= diff <= 4, hard=True,
                     detail=f'sum |confusion diff| = {diff}; original dev acc {acc_o}, weighted F1 {wf1_o}')
        acc = float((d1['logp'].argmax(1) == d1['y']).mean())
        record_check(S, f'{ck["name"]}: dev accuracy >= {100 * S["cfg"]["min_dev_acc"]:.0f}%',
                     acc >= S['cfg']['min_dev_acc'], hard=True, detail=f'{100 * acc:.2f}%')
        report.append(dict(name=ck['name'], archive=ck['archive'], dev_acc=100 * acc, original_dev_acc=acc_o,
                           original_dev_weighted_f1=wf1_o, determinism_max_diff=maxdiff, confusion_diff=diff))
        S['models'].append(model)
        S['dev_dump'].append(d1)
    for d in S['dev_dump'][1:]:
        assert np.array_equal(d['y'], S['dev_dump'][0]['y']) and np.array_equal(d['dlg'], S['dev_dump'][0]['dlg'])
    train_ds = make_dataset(S, 'train')  # eval-mode pass for diagnostic D3
    S['train_dump'] = [run_inference(S, m, train_ds) for m in S['models']]
    for ck, dv, tr in zip(S['ckpts'], S['dev_dump'], S['train_dump']):
        save_dump(S, 'dev', ck['name'], dv)
        save_dump(S, 'train', ck['name'], tr)
    write_json(os.path.join(S['out'], 'fidelity.json'), report)
    _save_checks(S)
    log_event(S, 'fidelity checks passed', dev_acc=[round(r['dev_acc'], 2) for r in report])


# ----------------------------------------------------------------------------- stage 5: dev analysis
def stage_dev_analysis(S):
    cfg = S['cfg']
    K = len(S['ckpts'])
    include = np.array([not c['flagged_test_selection'] for c in S['ckpts']])
    if not include.any():
        include[:] = True
    S['include'] = include
    n_dev = len(S['keys']['valid'])
    S['dev_cms'] = split_cms(S['dev_dump'], S['log_prior'], n_dev)
    tot = S['dev_cms'].sum(0)
    tau_rule, mean_by_rule = select_rule(tot, TAU_IDX, include)
    delta_rule, _ = select_rule(tot, DELTA_IDX, include)
    s1 = [select_rule(tot[k:k + 1], TAU_IDX, np.array([True]))[0] for k in range(K)]
    pi_dev = S['dev_counts'] / S['dev_counts'].sum()
    s2_rule, _ = select_rule(tot, TAU_IDX, include, row_weights=_div(S['prior'], pi_dev))
    fc = crossfit_forecast(S['dev_cms'], include, cfg['n_folds'], cfg['fold_seed'], cfg['power_B'],
                           len(S['keys']['test']))
    names = [c['name'] for c in S['ckpts']]
    sel = dict(include=[n for n, i in zip(names, include) if i],
               excluded_flagged=[n for n, i in zip(names, include) if not i],
               tau_star=RULES[tau_rule][1], delta_star=RULES[delta_rule][1],
               tau_per_checkpoint={n: RULES[r][1] for n, r in zip(names, s1)},
               tau_prior_matched=RULES[s2_rule][1],
               dev_mean_macro_f1_by_tau={str(RULES[r][1]): 100 * mean_by_rule[r] for r in TAU_IDX},
               dev_mean_macro_f1_by_delta={str(RULES[r][1]): 100 * mean_by_rule[r] for r in DELTA_IDX},
               forecast=fc)
    S['selection'] = sel
    rows = frontier_rows(tot, include, 'dev')
    write_csv(os.path.join(S['out'], 'tables', 'dev_frontier.csv'), rows)
    plot_frontier(S, rows, 'dev', sel['tau_star'])
    print('  running dev diagnostics (the RoBERTa probe takes a minute or two)...')
    S['dev_diag'] = dict(D1_separability=d1_separability(S, include), D3_train_vs_dev=d3_train_vs_dev(S, include),
                         D5_context=d5_context(S, include), D6_error_consistency=d6_consistency(S, include),
                         calibration_dev=calibration_dev(S))
    write_json(os.path.join(S['out'], 'dev_selection.json'), sel)
    write_json(os.path.join(S['out'], 'dev_diagnostics.json'), S['dev_diag'])
    print(dev_summary_text(S))
    log_event(S, 'dev analysis done', tau_star=sel['tau_star'], delta_star=sel['delta_star'],
              forecast=round(fc['crossfit_delta_macro_f1'], 2), mde80=round(fc['mde80_projected'], 2))


def dev_summary_text(S):
    sel, dg = S['selection'], S['dev_diag']
    by_tau = sel['dev_mean_macro_f1_by_tau']
    fc = sel['forecast']
    d1, d3 = dg['D1_separability'], dg['D3_train_vs_dev']
    lines = ['', 'DEV-ONLY RESULTS (frozen before the test set is touched)',
             f'  tau* = {sel["tau_star"]}  (dev mean macro-F1 {by_tau[str(sel["tau_star"])]:.2f} vs {by_tau["0.0"]:.2f} at tau=0)',
             f'  delta* (Neutral-only offset) = {sel["delta_star"]}; per-checkpoint tau = {sel["tau_per_checkpoint"]}; '
             f'prior-matched tau = {sel["tau_prior_matched"]}',
             f'  cross-fitted dev forecast of the macro-F1 gain: {fc["crossfit_delta_macro_f1"]:+.2f} '
             f'(fold taus {fc["fold_tau"]})',
             f'  projected test SE {fc["se_test_projected"]:.2f} -> effect detectable with 80% power ~ '
             f'{fc["mde80_projected"]:.2f} points',
             '  pairwise dev AUROC: ' + ', '.join(f'{p} {v:.3f}' for p, v in zip(d1['pairs'], d1['pairwise_auroc']))]
    tr = np.mean([c['train']['recall'] for c in d3['checkpoints']], 0)
    dv = np.mean([c['dev']['recall'] for c in d3['checkpoints']], 0)
    lines.append('  recall train -> dev (checkpoints): ' + ', '.join(f'{CLASSES[c]} {tr[c]:.0f}->{dv[c]:.0f}'
                                                                     for c in range(C)))
    pr = d3['roberta_probe']
    lines.append(f'  RoBERTa-only probe macro-F1 train {pr["train"]["macro_f1"]:.1f} -> dev {pr["dev"]["macro_f1"]:.1f}')
    return '\n'.join(lines)


# ----------------------------------------------------------------------------- stage 6: freeze
def stage_freeze(S):
    sel = S['selection']
    frozen = dict(
        created=time.strftime('%Y-%m-%d %H:%M:%S'),
        note='Written before any test-set access. stage_test reads every decision rule from this file.',
        code_sha256=S['env']['code_sha256'], data_sha256=S['data_sha256'], data_bytes=S['data_bytes'],
        checkpoints=[dict(name=c['name'], sha256=c['sha256'], archive=c['archive'],
                          flagged_test_selection=c['flagged_test_selection']) for c in S['ckpts']],
        include=sel['include'], train_counts=S['train_counts'], prior_train=S['prior'],
        log_prior_train=S['log_prior'], tau_grid=TAU_GRID, delta_grid=DELTA_GRID,
        selection_rule='highest mean dev macro-F1 over included checkpoints; ties go to the smaller value',
        tau_star=sel['tau_star'], delta_star=sel['delta_star'], tau_per_checkpoint=sel['tau_per_checkpoint'],
        tau_prior_matched=sel['tau_prior_matched'], la1_tau=1.0, temperature_dev=S['temperature'],
        forecast=sel['forecast'],
        bootstrap=dict(B=S['cfg']['bootstrap_B'], seed=S['cfg']['boot_seed'], unit='dialogue',
                       design='dev and test resampled independently; tau and delta re-selected in every replicate'),
        criteria=CRITERIA, device=S['device'], env=S['env'])
    path = os.path.join(S['out'], 'frozen_config.json')
    write_json(path, frozen)
    S['frozen_path'], S['frozen_sha256'] = path, sha256_file(path)
    log_event(S, 'FREEZE', frozen_config_sha256=S['frozen_sha256'][:16], tau_star=sel['tau_star'])


# ----------------------------------------------------------------------------- stage 7: test (once)
def decide(est, lo, hi, d_f1, d_acc, d_wf1, per_ckpt):
    c = dict(delta_ge_1=est >= 1.0, ci_lower_gt_0=lo > 0, fear_improves=d_f1[FEA] > 0,
             disgust_improves=d_f1[DIS] > 0, accuracy_guardrail=d_acc >= -1.0, weighted_f1_guardrail=d_wf1 >= -0.5,
             class_guardrail=bool(np.all(np.asarray(d_f1) >= -2.0)),
             every_checkpoint_positive=all(x > 0 for x in per_ckpt))
    efficacy = c['delta_ge_1'] and c['ci_lower_gt_0']
    minority = c['fear_improves'] and c['disgust_improves'] and c['every_checkpoint_positive']
    guard = c['accuracy_guardrail'] and c['weighted_f1_guardrail'] and c['class_guardrail']
    if efficacy and minority and guard:
        v = 'SUCCESS'
    elif efficacy and minority:
        v = 'TRADE-OFF'
    elif efficacy:
        v = 'PARTIAL'
    elif hi < 1.0:
        kind = 'harmful' if hi < 0 else ('real but small' if lo > 0 else 'no detectable effect')
        v = f'FAILURE ({kind}; the +1.0 target is excluded)'
    else:
        v = 'INCONCLUSIVE'
    return v, c


def stage_test(S):
    if not S.get('frozen_path') or sha256_file(S['frozen_path']) != S['frozen_sha256']:
        raise CheckFailed('frozen_config.json is missing or was modified: run the freeze cell first')
    with open(S['frozen_path']) as f:
        F = json.load(f)
    if sha256_file(os.path.join(S['code_dir'], 'l2_posthoc.py')) != F['code_sha256']['l2_posthoc.py']:
        raise CheckFailed('analysis code changed after the freeze')
    marker = os.path.join(S['out'], 'TEST_EVALUATED.json')
    rerun = os.path.exists(marker)
    if rerun:
        print('NOTE: the test set was already evaluated in this session. The first complete run is the result.')
    log_event(S, 'TEST evaluation starts (test dialogues are run and test labels read only from here on)',
              rerun=rerun)
    names = [c['name'] for c in S['ckpts']]
    K = len(names)
    include = np.array([n in F['include'] for n in names])
    log_prior = np.asarray(F['log_prior_train'], dtype=float)
    rule = dict(LA=RULES.index(('tau', F['tau_star'])), NO=RULES.index(('delta', F['delta_star'])),
                S2=RULES.index(('tau', F['tau_prior_matched'])))
    s1 = [RULES.index(('tau', F['tau_per_checkpoint'][n])) for n in names]
    test_ds = make_dataset(S, 'test')
    test_dump = [run_inference(S, m, test_ds) for m in S['models']]
    for n, d in zip(names, test_dump):
        save_dump(S, 'test', n, d)
    test_cms = split_cms(test_dump, log_prior, len(S['keys']['test']))
    tot = test_cms.sum(0)
    sc = scores(tot.reshape(K, R, C, C))
    arms = {'BASE': [BASE_RULE] * K, 'LA*': [rule['LA']] * K, 'NO*': [rule['NO']] * K, 'LA1': [LA1_RULE] * K,
            'S1_per_checkpoint_tau': s1, 'S2_prior_matched_tau': [rule['S2']] * K}
    ar = np.arange(K)
    point, rows = {}, []
    for arm, rr in arms.items():
        rr = np.asarray(rr)
        m = {k: sc[k][ar, rr] for k in ('acc', 'mf1', 'wf1', 'bacc')}
        pc = {k: sc[k][ar, rr, :] for k in ('f1', 'prec', 'rec', 'pred', 'support')}
        point[arm] = dict(
            per_checkpoint={n: dict(acc=100 * m['acc'][k], macro_f1=100 * m['mf1'][k],
                                    weighted_f1=100 * m['wf1'][k], f1=100 * pc['f1'][k]) for k, n in enumerate(names)},
            mean=dict(acc=100 * m['acc'][include].mean(), macro_f1=100 * m['mf1'][include].mean(),
                      weighted_f1=100 * m['wf1'][include].mean(), balanced_acc=100 * m['bacc'][include].mean(),
                      f1=100 * pc['f1'][include].mean(0), precision=100 * pc['prec'][include].mean(0),
                      recall=100 * pc['rec'][include].mean(0),
                      pred_over_true=_div(pc['pred'][include].mean(0), pc['support'][include].mean(0)),
                      support=pc['support'][0]))
        for k, n in enumerate(names):
            row = dict(arm=arm, checkpoint=n, included=bool(include[k]), acc=100 * m['acc'][k],
                       macro_f1=100 * m['mf1'][k], weighted_f1=100 * m['wf1'][k], balanced_acc=100 * m['bacc'][k])
            for c in range(C):
                row[f'F1_{CLASSES[c]}'] = 100 * pc['f1'][k, c]
                row[f'P_{CLASSES[c]}'] = 100 * pc['prec'][k, c]
                row[f'R_{CLASSES[c]}'] = 100 * pc['rec'][k, c]
            rows.append(row)
    write_csv(os.path.join(S['out'], 'tables', 'test_arms_per_checkpoint.csv'), rows)

    def dlt(arm, key):
        return np.asarray(point[arm]['mean'][key]) - np.asarray(point['BASE']['mean'][key])

    B, seed = F['bootstrap']['B'], F['bootstrap']['seed']
    print(f'  bootstrap: {B} replicates...')
    bt = procedure_bootstrap(S['dev_cms'], test_cms, include, B, seed)
    ci = lambda x: np.percentile(x, [2.5, 97.5])  # noqa: E731
    est = float(dlt('LA*', 'macro_f1'))
    lo, hi = ci(bt['d_mf1'])
    d_f1, d_acc, d_wf1 = dlt('LA*', 'f1'), float(dlt('LA*', 'acc')), float(dlt('LA*', 'weighted_f1'))
    per_ck = {n: point['LA*']['per_checkpoint'][n]['macro_f1'] - point['BASE']['per_checkpoint'][n]['macro_f1']
              for n in names}
    verdict, conditions = decide(est, lo, hi, d_f1, d_acc, d_wf1, [per_ck[n] for n in F['include']])
    s3 = fixed_rule_bootstrap(test_cms, include, arms['LA*'], B, seed + 1)
    s1b = fixed_rule_bootstrap(test_cms, include, s1, B, seed + 2)
    s2b = fixed_rule_bootstrap(test_cms, include, arms['S2_prior_matched_tau'], B, seed + 3)
    base_wf1 = float(point['BASE']['mean']['weighted_f1'])
    results = dict(
        verdict=verdict, conditions=conditions, rerun=rerun,
        primary=dict(delta_macro_f1=est, ci95=[lo, hi], tau_star=F['tau_star'], per_checkpoint=per_ck,
                     ci_kind='procedure-level two-sample dialogue bootstrap (tau re-selected per replicate)'),
        per_class_delta=dict(f1=d_f1, f1_ci95=[ci(bt['d_f1'][:, c]) for c in range(C)],
                             precision=dlt('LA*', 'precision'), precision_ci95=[ci(bt['d_prec'][:, c]) for c in range(C)],
                             recall=dlt('LA*', 'recall'), recall_ci95=[ci(bt['d_rec'][:, c]) for c in range(C)]),
        guardrails=dict(delta_acc=d_acc, delta_acc_ci95=ci(bt['d_acc']), delta_weighted_f1=d_wf1,
                        delta_weighted_f1_ci95=ci(bt['d_wf1'])),
        mechanism=dict(delta_NO=float(dlt('NO*', 'macro_f1')), delta_NO_ci95=ci(bt['d_mf1_no']),
                       LA_minus_NO=est - float(dlt('NO*', 'macro_f1')), LA_minus_NO_ci95=ci(bt['mech']),
                       delta_LA1=float(dlt('LA1', 'macro_f1')), delta_LA1_ci95=ci(bt['d_mf1_la1'])),
        sensitivity=dict(
            S1_per_checkpoint_tau=dict(delta=float(dlt('S1_per_checkpoint_tau', 'macro_f1')), test_only_ci95=ci(s1b)),
            S2_prior_matched_tau=dict(tau=F['tau_prior_matched'], delta=float(dlt('S2_prior_matched_tau', 'macro_f1')),
                                      test_only_ci95=ci(s2b)),
            S3_fixed_tau_star_test_only_ci95=ci(s3),
            S4_all_checkpoints_included=bool(include.all())),
        bootstrap_tau_selected=dict(zip(*np.unique(bt['tau_sel'], return_counts=True))),
        fidelity=dict(base_weighted_f1=base_wf1, paper_weighted_f1=PAPER_TEST_WF1,
                      flag=abs(base_wf1 - PAPER_TEST_WF1) > 1.5),
        arms=point)
    for arm in ('BASE', 'LA*', 'NO*'):
        cm = tot[include][:, arms[arm][0]].sum(0).reshape(C, C)
        write_csv(os.path.join(S['out'], 'tables', f'test_confusion_{arm.strip("*")}.csv'),
                  [dict(true=CLASSES[i], **{CLASSES[j]: int(cm[i, j]) for j in range(C)}) for i in range(C)])
    trows = frontier_rows(tot, include, 'test')
    write_csv(os.path.join(S['out'], 'tables', 'test_frontier.csv'), trows)
    plot_frontier(S, trows, 'test', F['tau_star'])
    results['calibration_test'] = {n: dict(raw=calib_stats(d['logp'], d['y']),
                                           temperature_scaled=calib_stats(d['logp'], d['y'], F['temperature_dev'][n]))
                                   for n, d in zip(names, test_dump)}
    write_json(os.path.join(S['out'], 'test_results.json'), results)
    S['results'], S['frozen'] = results, F
    with open(os.path.join(S['out'], 'SUMMARY.md'), 'w') as f:
        f.write(summary_markdown(S))
    write_json(marker, dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), frozen_config_sha256=S['frozen_sha256']))
    log_event(S, 'TEST done', verdict=verdict, delta=round(est, 2), ci=[round(lo, 2), round(hi, 2)])


# ----------------------------------------------------------------------------- summary and packaging
def summary_markdown(S):
    F, Rz, env = S['frozen'], S['results'], S['env']
    sel, fc = S['selection'], F['forecast']
    P = Rz['primary']
    L = [f'# ECERC Limitation 2: post-hoc logit adjustment (MELD, released checkpoints)', '',
         f'Run {time.strftime("%Y-%m-%d %H:%M")} | device {env["device"]} ({env["gpu"]}) | torch {env["torch"]} | '
         f'numpy {env["numpy"]} | sklearn {env["sklearn"]}',
         f'Data sha256 {S["data_sha256"][:16]}... ({S["data_bytes"] / 2 ** 20:.0f} MB) | checkpoints: '
         + ', '.join(f'{c["name"]} ({c["archive"]})' for c in S['ckpts']) + f' | included: {F["include"]}', '']
    warn = [c for c in S['checks'] if not c['passed']]
    L.append(f'Checks: {sum(c["passed"] for c in S["checks"])} passed, {len(warn)} warnings'
             + ('' if not warn else ' -- ' + '; '.join(f'{c["name"]} ({c["detail"]})' for c in warn)))
    L += ['', '## Frozen on dev before the test set was touched',
          f'- tau* = {F["tau_star"]} (dev mean macro-F1 {sel["dev_mean_macro_f1_by_tau"][str(F["tau_star"])]:.2f} '
          f'vs {sel["dev_mean_macro_f1_by_tau"]["0.0"]:.2f} at tau = 0); Neutral-only delta* = {F["delta_star"]}',
          f'- per-checkpoint tau {F["tau_per_checkpoint"]}; prior-matched tau {F["tau_prior_matched"]}',
          f'- cross-fitted dev forecast of the gain {fc["crossfit_delta_macro_f1"]:+.2f}; projected test SE '
          f'{fc["se_test_projected"]:.2f}, so ~{fc["mde80_projected"]:.2f} points detectable with 80% power',
          '', f'## Verdict: {Rz["verdict"]}', '', '| condition | met |', '|---|---|']
    L += [f'| {k} | {"yes" if v else "no"} |' for k, v in Rz['conditions'].items()]
    L += ['', '## Test results (mean over included checkpoints)', '',
          '| arm | acc | weighted F1 | macro-F1 | ' + ' | '.join(f'F1 {c}' for c in CLASSES) + ' |',
          '|---' * (4 + C) + '|']
    for arm, v in Rz['arms'].items():
        m = v['mean']
        L.append(f'| {arm} | {m["acc"]:.2f} | {m["weighted_f1"]:.2f} | {m["macro_f1"]:.2f} | '
                 + ' | '.join(f'{x:.2f}' for x in m['f1']) + ' |')
    pc = Rz['per_class_delta']
    g, mech, sens = Rz['guardrails'], Rz['mechanism'], Rz['sensitivity']
    L += ['', f'- Delta macro-F1 (LA* - BASE): **{_fmt(P["delta_macro_f1"], *P["ci95"])}** ({P["ci_kind"]})',
          f'- test-only CI with tau* fixed: [{sens["S3_fixed_tau_star_test_only_ci95"][0]:+.2f}, '
          f'{sens["S3_fixed_tau_star_test_only_ci95"][1]:+.2f}]',
          '- per checkpoint: ' + ', '.join(f'{n} {v:+.2f}' for n, v in P['per_checkpoint'].items()),
          f'- Delta accuracy {_fmt(g["delta_acc"], *g["delta_acc_ci95"])}; '
          f'Delta weighted F1 {_fmt(g["delta_weighted_f1"], *g["delta_weighted_f1_ci95"])}',
          '', '| class | Delta F1 [95% CI] | Delta precision | Delta recall | support |', '|---|---|---|---|---|']
    sup = Rz['arms']['BASE']['mean']['support']
    for c in range(C):
        L.append(f'| {CLASSES[c]} | {_fmt(pc["f1"][c], *pc["f1_ci95"][c])} | {pc["precision"][c]:+.2f} | '
                 f'{pc["recall"][c]:+.2f} | {int(sup[c])} |')
    L += ['', f'- Neutral-only offset NO*: {_fmt(mech["delta_NO"], *mech["delta_NO_ci95"])}; '
              f'LA* - NO*: {_fmt(mech["LA_minus_NO"], *mech["LA_minus_NO_ci95"])}',
          f'- LA1 (tau = 1, untuned): {_fmt(mech["delta_LA1"], *mech["delta_LA1_ci95"])}',
          f'- S1 per-checkpoint tau: {_fmt(sens["S1_per_checkpoint_tau"]["delta"], *sens["S1_per_checkpoint_tau"]["test_only_ci95"])}'
          f' | S2 prior-matched tau ({sens["S2_prior_matched_tau"]["tau"]}): '
          f'{_fmt(sens["S2_prior_matched_tau"]["delta"], *sens["S2_prior_matched_tau"]["test_only_ci95"])}',
          '- predicted / true count, BASE -> LA*: ' + ', '.join(
              f'{CLASSES[c]} {Rz["arms"]["BASE"]["mean"]["pred_over_true"][c]:.2f}->'
              f'{Rz["arms"]["LA*"]["mean"]["pred_over_true"][c]:.2f}' for c in range(C)),
          f'- fidelity: BASE weighted F1 {Rz["fidelity"]["base_weighted_f1"]:.2f} vs paper {PAPER_TEST_WF1} '
          f'({"FLAG: differs by > 1.5" if Rz["fidelity"]["flag"] else "ok"})']
    dg = S['dev_diag']
    d1, d3, d5, d6 = (dg['D1_separability'], dg['D3_train_vs_dev'], dg['D5_context'], dg['D6_error_consistency'])
    L += ['', '## Dev diagnostics', '',
          '- pairwise AUROC: ' + ', '.join(f'{p} {v:.3f}' for p, v in zip(d1['pairs'], d1['pairwise_auroc'])),
          '- one-vs-rest AUROC / AP: ' + ', '.join(f'{CLASSES[c]} {d1["ovr_auroc"][c]:.3f}/{d1["ovr_ap"][c]:.3f}'
                                                   for c in range(C))]
    tr = np.mean([c['train']['recall'] for c in d3['checkpoints']], 0)
    dv = np.mean([c['dev']['recall'] for c in d3['checkpoints']], 0)
    pr = d3['roberta_probe']
    L += ['- checkpoint recall train -> dev: ' + ', '.join(f'{CLASSES[c]} {tr[c]:.0f}->{dv[c]:.0f}' for c in range(C)),
          f'- RoBERTa-only probe: macro-F1 train {pr["train"]["macro_f1"]:.1f} -> dev {pr["dev"]["macro_f1"]:.1f}; recall '
          + ', '.join(f'{CLASSES[c]} {pr["train"]["recall"][c]:.0f}->{pr["dev"]["recall"][c]:.0f}' for c in range(C))]
    for st, v in d5['strata'].items():
        if v['recall'] is not None:
            L.append(f'- dev recall, {st} (n={v["n"]}): ' + ', '.join(
                f'{CLASSES[c]} {v["recall"][c]:.0f} (n={int(v["class_counts"][c])})' for c in range(C)))
    for cls, groups in d5['branch_margins']['classes'].items():
        for group, v in groups.items():
            L.append(f'- branch margins, {cls} {group} (n~{v["n"]:.0f}): total {v["margin"]:+.2f} = '
                     + ' + '.join(f'{b} {v[b]:+.2f}' for b in BRANCHES) + f' + bias {v["bias"]:+.2f}')
    k = d6['n_checkpoints']
    L.append('- dev errors made by all checkpoints: ' + ', '.join(
        f'{c} {v[f"wrong_in_{k}_of_{k}"]}/{v["n"] - v[f"wrong_in_0_of_{k}"]}' for c, v in d6['by_class'].items()))
    return '\n'.join(L) + '\n'


def stage_package(S):
    zpath = os.path.join(S['cfg']['work_dir'], 'ecerc_l2_results.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(S['out']):
            for name in files:
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, os.path.dirname(S['out'])))
    summary = os.path.join(S['out'], 'SUMMARY.md')
    if os.path.exists(summary):
        with open(summary) as f:
            print(f.read())
    print(f'Results archive: {zpath} ({os.path.getsize(zpath) / 2 ** 20:.1f} MB)')
    return zpath
