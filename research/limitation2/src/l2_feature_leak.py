"""
ECERC Limitation 2, follow-up: do the precomputed text emotion features carry the training labels?

New code for research/limitation2. It reads only the train and dev splits of the authors' MELD feature
file; the test split is never read.

Why: the text emotion features (eroberta1) come from a RoBERTa model fine-tuned on the MELD training
utterances (COSMIC protocol). If that fine-tuning baked the training labels into the training-set features,
any model trained on them sees unrealistically easy rare-class examples, and its held-out behaviour differs.

Test: a linear probe is fitted on 4/5 of the training dialogues and scored on (a) the held-out 1/5 of the
training dialogues and (b) the dev set. Both are unseen by the probe, so without a leak their scores should
be close. The held-out training dialogues were, however, seen by the RoBERTa fine-tuning. The plain RoBERTa
features (sroberta1, not fine-tuned) are the control; audio and visual features are reported descriptively.
A 1-nearest-neighbour check (other dialogues only) gives a model-free second view.

Pre-registered rule (RULE_TEXT): LEAK CONFIRMED if gap(eroberta1) >= 15 macro-F1 points and
gap(eroberta1) - gap(sroberta1) >= 10 with a 95% bootstrap CI lower bound > 0, where gap = macro-F1 on
held-out training dialogues - mean macro-F1 of the same fold models on dev. Otherwise NOT CONFIRMED.
"""
import os
import pickle
import platform
import subprocess
import sys
import time
import zipfile

import numpy as np

import l2_posthoc as P  # safe loader, metrics and helpers from the first experiment (unchanged)

FEATURES = {
    'eroberta1': 'text emotion features: RoBERTa-large fine-tuned on MELD training labels (ECERC evidence input)',
    'sroberta1': 'text semantic features: RoBERTa-large, not fine-tuned (ECERC event input); control',
    'videoAudio': 'acoustic features (descriptive)',
    'videoVisual': 'visual features (descriptive)',
}
PRIMARY, CONTROL = 'eroberta1', 'sroberta1'
RULE = dict(min_gap_primary=15.0, min_gap_difference=10.0)
RULE_TEXT = ('LEAK CONFIRMED if gap(eroberta1) >= 15 macro-F1 points and gap(eroberta1) - gap(sroberta1) >= 10 '
             'with a 95% bootstrap CI lower bound > 0; otherwise NOT CONFIRMED. gap = macro-F1 on held-out '
             'training dialogues (out-of-fold) minus the mean dev macro-F1 of the same fold models.')


def _cm(y, p):
    return np.bincount(y * P.C + p, minlength=P.C * P.C).reshape(P.C, P.C)


# ----------------------------------------------------------------------------- stages
def stage_setup(CONFIG, code_dir):
    out = os.path.join(CONFIG['work_dir'], 'outputs_leak')
    for sub in ('', 'tables', 'figures'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    import joblib
    import sklearn
    S = dict(cfg=CONFIG, out=out, code_dir=code_dir, checks=[])
    S['env'] = dict(python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
                    sklearn=sklearn.__version__, joblib=joblib.__version__, cpus=os.cpu_count(),
                    code_sha256={f: P.sha256_file(os.path.join(code_dir, f))
                                 for f in sorted(os.listdir(code_dir)) if f.endswith('.py')})
    P.write_json(os.path.join(out, 'environment.json'), S['env'])
    P.write_json(os.path.join(out, 'rule.json'),  # written before any data is loaded
                 dict(rule=RULE_TEXT, primary=PRIMARY, control=CONTROL, features=FEATURES,
                      n_folds=CONFIG['n_folds'], fold_seed=CONFIG['fold_seed'], C=CONFIG['C'],
                      max_iter=CONFIG['max_iter'], bootstrap_B=CONFIG['bootstrap_B'], boot_seed=CONFIG['boot_seed']))
    P.log_event(S, 'setup', cpus=os.cpu_count(), sklearn=sklearn.__version__)
    return S


def stage_download(S):
    cfg = S['cfg']
    data_dir = os.path.join(cfg['work_dir'], 'data')
    os.makedirs(data_dir, exist_ok=True)
    path = cfg.get('data_local_path') or os.path.join(data_dir, cfg['data_filename'])
    if os.path.exists(path) and os.path.getsize(path) < 1_000_000:  # leftover HTML error page
        os.remove(path)
    if not os.path.exists(path):
        if cfg.get('offline'):
            raise P.CheckFailed(f'offline run but {path} is missing')
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
            raise P.CheckFailed(P.DATA_FALLBACK.format(id=cfg['data_file_id'], dir=data_dir,
                                                       name=cfg['data_filename']))
    S['data_path'], S['data_bytes'], S['data_sha256'] = path, os.path.getsize(path), P.sha256_file(path)
    if cfg.get('expected_data_sha256'):
        P.record_check(S, 'data file is identical to the one used in run_colab_1',
                       S['data_sha256'] == cfg['expected_data_sha256'], hard=False, detail=S['data_sha256'][:16])
    P._save_checks(S)
    P.log_event(S, 'data ready', MB=round(S['data_bytes'] / 2 ** 20, 1), sha256=S['data_sha256'][:16])


def stage_load(S):
    cfg = S['cfg']
    t0 = time.time()
    try:
        obj = P.load_pickle(S['data_path'], cfg.get('trust_data_pickle', False))
    except pickle.UnpicklingError as e:
        raise P.CheckFailed(f'Safe loader refused the data file ({e}). {P.TRUST_HINT}') from None
    P.record_check(S, 'data file has the 17 fields read by MELD/dataloader.py',
                   isinstance(obj, (tuple, list)) and len(obj) == len(P.FIELDS), hard=True)
    D = dict(zip(P.FIELDS, obj))
    keys = {'train': list(D['train_ids']), 'valid': list(D['valid_ids'])}  # the test split is never read
    X, y, dlg = {}, {}, {}
    for split, ks in keys.items():
        lens = [len(D['emotion_labels'][v]) for v in ks]
        y[split] = np.concatenate([np.asarray(D['emotion_labels'][v], dtype=np.int64) for v in ks])
        dlg[split] = np.concatenate([np.full(n, i) for i, n in enumerate(lens)])
        X[split] = {f: np.concatenate([np.asarray(D[f][v], dtype=np.float32).reshape(n, -1)
                                       for v, n in zip(ks, lens)]) for f in FEATURES}
    del D, obj
    S.update(X=X, y=y, dlg=dlg, n_dlg={s: len(k) for s, k in keys.items()})
    n_utt = {s: int(len(y[s])) for s in y}
    if cfg.get('expected_utterances'):
        P.record_check(S, 'utterance counts match MELD (train 9989, dev 1109)', n_utt == cfg['expected_utterances'],
                       hard=False, detail=n_utt)
    counts = np.bincount(y['train'], minlength=P.C)
    P.record_check(S, 'train label counts match the MELD paper', counts.tolist() == P.OFFICIAL_TRAIN_COUNTS,
                   hard=False, detail=counts.tolist())
    P._save_checks(S)
    P.log_event(S, 'train/dev features loaded (test split not read)', seconds=round(time.time() - t0, 1),
                utterances=n_utt, dims={f: int(X['train'][f].shape[1]) for f in FEATURES})


def _fit_predict(X_fit, y_fit, evals, C_reg, max_iter):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_fit)
    clf = LogisticRegression(max_iter=max_iter, C=C_reg)  # same probe as diagnostic D3 of the first run
    clf.fit(scaler.transform(X_fit), y_fit)
    return [clf.predict(scaler.transform(X)) for X in evals], int(np.max(clf.n_iter_))


def stage_probes(S):
    from joblib import Parallel, delayed
    cfg = S['cfg']
    K = cfg['n_folds']
    rng = np.random.default_rng(cfg['fold_seed'])
    fold_of_dlg = np.empty(S['n_dlg']['train'], dtype=int)
    for k, idx in enumerate(np.array_split(rng.permutation(S['n_dlg']['train']), K)):
        fold_of_dlg[idx] = k
    fold = fold_of_dlg[S['dlg']['train']]
    Xtr, Xdv, ytr = S['X']['train'], S['X']['valid'], S['y']['train']
    tasks = [(f, k) for f in FEATURES for k in range(K)]
    print(f'  fitting {len(tasks)} linear probes on {os.cpu_count()} CPU cores; this takes a few minutes...')
    t0 = time.time()
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_fit_predict)(Xtr[f][fold != k], ytr[fold != k], [Xtr[f][fold == k], Xdv[f], Xtr[f][fold != k]],
                              cfg['C'], cfg['max_iter']) for f, k in tasks)
    S['fold'], S['probe'] = fold, {}
    for (f, k), (preds, n_iter) in zip(tasks, results):
        r = S['probe'].setdefault(f, dict(oof=np.full(len(ytr), -1), dev=[None] * K, insample_mf1=[None] * K,
                                          n_iter=[None] * K))
        r['oof'][fold == k] = preds[0]
        r['dev'][k] = preds[1]
        r['insample_mf1'][k] = 100 * float(P.scores(_cm(ytr[fold != k], preds[2]))['mf1'])
        r['n_iter'][k] = n_iter
    assert all((S['probe'][f]['oof'] >= 0).all() for f in FEATURES)
    P.log_event(S, 'probes fitted', seconds=round(time.time() - t0, 1),
                hit_max_iter=[f for f in FEATURES if max(S['probe'][f]['n_iter']) >= cfg['max_iter']])


def _nn_predict(Xq, Xref, yref, dq=None, dref=None, chunk=2048):
    """1-nearest-neighbour labels by cosine similarity; neighbours from the query's own dialogue are excluded."""
    preds = np.empty(len(Xq), dtype=np.int64)
    for i in range(0, len(Xq), chunk):
        sims = Xq[i:i + chunk] @ Xref.T
        if dq is not None:
            sims[dq[i:i + chunk, None] == dref[None, :]] = -np.inf
        preds[i:i + chunk] = yref[sims.argmax(1)]
    return preds


def stage_nearest_neighbour(S):
    t0 = time.time()
    S['nn'] = {}
    for f in FEATURES:
        Xtr, Xdv = S['X']['train'][f], S['X']['valid'][f]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        A = (Xtr - mu) / sd
        A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
        Bq = (Xdv - mu) / sd
        Bq /= np.linalg.norm(Bq, axis=1, keepdims=True) + 1e-12
        S['nn'][f] = dict(train=_nn_predict(A, A, S['y']['train'], S['dlg']['train'], S['dlg']['train']),
                          dev=_nn_predict(Bq, A, S['y']['train']))
    P.log_event(S, 'nearest-neighbour check done', seconds=round(time.time() - t0, 1))


def stage_decide(S):
    cfg = S['cfg']
    K, feats = cfg['n_folds'], list(FEATURES)
    ytr, ydv = S['y']['train'], S['y']['valid']
    ntr, ndv = S['n_dlg']['train'], S['n_dlg']['valid']
    oof_cms = np.stack([P.per_dialogue_cms(ytr, S['probe'][f]['oof'][None, :], S['dlg']['train'], ntr)[:, 0, :]
                        for f in feats], axis=1)  # (ntr, F, 49)
    dev_cms = np.stack([P.per_dialogue_cms(ydv, np.stack(S['probe'][f]['dev']), S['dlg']['valid'], ndv)
                        for f in feats], axis=1)  # (ndv, F, K, 49)

    def stats(wt, wd):
        o = P.scores((wt @ oof_cms.reshape(ntr, -1)).reshape(-1, len(feats), P.C, P.C))
        d = P.scores((wd @ dev_cms.reshape(ndv, -1)).reshape(-1, len(feats), K, P.C, P.C))
        return o, d

    o, d = stats(np.ones((1, ntr)), np.ones((1, ndv)))
    oof_mf1, dev_mf1 = 100 * o['mf1'][0], 100 * d['mf1'][0].mean(-1)
    gap = oof_mf1 - dev_mf1
    rng = np.random.default_rng(cfg['boot_seed'])
    ob, db = stats(P._boot_weights(rng, ntr, cfg['bootstrap_B']), P._boot_weights(rng, ndv, cfg['bootstrap_B']))
    gap_b = 100 * ob['mf1'] - 100 * db['mf1'].mean(-1)  # (B, F)
    ip, ic = feats.index(PRIMARY), feats.index(CONTROL)
    did = float(gap[ip] - gap[ic])
    did_ci = np.percentile(gap_b[:, ip] - gap_b[:, ic], [2.5, 97.5])
    gap_ci = np.percentile(gap_b, [2.5, 97.5], axis=0)
    confirmed = bool(gap[ip] >= RULE['min_gap_primary'] and did >= RULE['min_gap_difference'] and did_ci[0] > 0)
    table = {}
    for i, f in enumerate(feats):
        nn_tr, nn_dv = _cm(ytr, S['nn'][f]['train']), _cm(ydv, S['nn'][f]['dev'])
        table[f] = dict(
            description=FEATURES[f],
            insample_macro_f1=float(np.mean(S['probe'][f]['insample_mf1'])),
            heldout_train_macro_f1=float(oof_mf1[i]), dev_macro_f1=float(dev_mf1[i]),
            gap=float(gap[i]), gap_ci95=gap_ci[:, i],
            heldout_train_recall=100 * o['rec'][0, i], dev_recall=100 * d['rec'][0, i].mean(0),
            heldout_train_acc=100 * float(o['acc'][0, i]), dev_acc=100 * float(d['acc'][0, i].mean()),
            nn_train_macro_f1=100 * float(P.scores(nn_tr)['mf1']), nn_dev_macro_f1=100 * float(P.scores(nn_dv)['mf1']),
            nn_train_recall=100 * P.scores(nn_tr)['rec'], nn_dev_recall=100 * P.scores(nn_dv)['rec'],
            probe_iterations=S['probe'][f]['n_iter'])
    res = dict(verdict='LEAK CONFIRMED' if confirmed else 'NOT CONFIRMED', rule=RULE_TEXT,
               gap_primary=float(gap[ip]), gap_control=float(gap[ic]), difference=did, difference_ci95=did_ci,
               features=table, folds=K, dialogues=S['n_dlg'])
    S['res'] = res
    P.write_json(os.path.join(S['out'], 'leak_results.json'), res)
    rows = []
    for f, t in table.items():
        row = dict(features=f, insample=t['insample_macro_f1'], heldout_train=t['heldout_train_macro_f1'],
                   dev=t['dev_macro_f1'], gap=t['gap'], gap_lo=t['gap_ci95'][0], gap_hi=t['gap_ci95'][1],
                   nn_train=t['nn_train_macro_f1'], nn_dev=t['nn_dev_macro_f1'])
        for c in range(P.C):
            row[f'R_{P.CLASSES[c]}_heldout_train'] = t['heldout_train_recall'][c]
            row[f'R_{P.CLASSES[c]}_dev'] = t['dev_recall'][c]
        rows.append(row)
    P.write_csv(os.path.join(S['out'], 'tables', 'leak_probe_table.csv'), rows)
    _plot(S, table)
    with open(os.path.join(S['out'], 'SUMMARY_leak.md'), 'w') as f:
        f.write(summary_markdown(S))
    P.log_event(S, 'decision', verdict=res['verdict'], gap_primary=round(res['gap_primary'], 1),
                gap_control=round(res['gap_control'], 1), difference=round(did, 1))


def _plot(S, table):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print('  (plot skipped:', e, ')')
        return
    feats = list(table)
    x = np.arange(len(feats))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for j, (key, lab) in enumerate((('insample_macro_f1', 'fitted training dialogues'),
                                    ('heldout_train_macro_f1', 'held-out training dialogues'),
                                    ('dev_macro_f1', 'dev'))):
        ax[0].bar(x + (j - 1) * 0.27, [table[f][key] for f in feats], 0.27, label=lab)
    ax[0].set_xticks(x, feats)
    ax[0].set(ylabel='macro-F1 (%)', title='Linear probe: same models, three kinds of utterances')
    ax[0].legend(fontsize=8)
    t = table[PRIMARY]
    xc = np.arange(P.C)
    ax[1].bar(xc - 0.2, t['heldout_train_recall'], 0.4, label='held-out training dialogues')
    ax[1].bar(xc + 0.2, t['dev_recall'], 0.4, label='dev')
    ax[1].set_xticks(xc, P.CLASSES)
    ax[1].set(ylabel='recall (%)', title=f'{PRIMARY} probe: per-class recall')
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(S['out'], 'figures', 'leak_probe.png'), dpi=130)
    plt.close(fig)


def summary_markdown(S):
    res, env = S['res'], S['env']
    L = ['# ECERC Limitation 2: do the text emotion features carry the training labels?', '',
         f'Run {time.strftime("%Y-%m-%d %H:%M")} | {env["cpus"]} CPU cores | sklearn {env["sklearn"]} | '
         f'data sha256 {S["data_sha256"][:16]}... | train/dev only (test split never read)', '',
         f'## Verdict: {res["verdict"]}', '', f'Rule: {res["rule"]}', '',
         f'- gap, emotion text features ({PRIMARY}): **{res["gap_primary"]:+.1f}** points',
         f'- gap, plain text features ({CONTROL}): {res["gap_control"]:+.1f} points',
         f'- difference: **{res["difference"]:+.1f} [95% CI {res["difference_ci95"][0]:+.1f}, '
         f'{res["difference_ci95"][1]:+.1f}]**', '',
         '| features | fitted train | held-out train | dev | gap [95% CI] | 1-NN train | 1-NN dev |',
         '|---|---|---|---|---|---|---|']
    for f, t in res['features'].items():
        L.append(f'| {f} | {t["insample_macro_f1"]:.1f} | {t["heldout_train_macro_f1"]:.1f} | {t["dev_macro_f1"]:.1f} | '
                 f'{t["gap"]:+.1f} [{t["gap_ci95"][0]:+.1f}, {t["gap_ci95"][1]:+.1f}] | '
                 f'{t["nn_train_macro_f1"]:.1f} | {t["nn_dev_macro_f1"]:.1f} |')
    L += ['', 'Macro-F1 (%) of linear probes fitted on 4/5 of the training dialogues (5 folds). "Held-out train" = '
              'the remaining training dialogues; 1-NN = nearest neighbour in other dialogues.', '',
          '| class | ' + ' | '.join(f'{f} held-out train / dev recall' for f in (PRIMARY, CONTROL)) + ' |',
          '|---|---|---|']
    for c in range(P.C):
        L.append(f'| {P.CLASSES[c]} | ' + ' | '.join(
            f'{res["features"][f]["heldout_train_recall"][c]:.0f} / {res["features"][f]["dev_recall"][c]:.0f}'
            for f in (PRIMARY, CONTROL)) + ' |')
    warn = [c for c in S['checks'] if not c['passed']]
    L += ['', f'Checks: {sum(c["passed"] for c in S["checks"])} passed, {len(warn)} warnings'
          + ('' if not warn else ' -- ' + '; '.join(f'{c["name"]} ({c["detail"]})' for c in warn))]
    return '\n'.join(L) + '\n'


def stage_package(S):
    zpath = os.path.join(S['cfg']['work_dir'], 'ecerc_l2_leak_results.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(S['out']):
            for name in files:
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, os.path.dirname(S['out'])))
    with open(os.path.join(S['out'], 'SUMMARY_leak.md')) as f:
        print(f.read())
    print(f'Results archive: {zpath} ({os.path.getsize(zpath) / 2 ** 20:.1f} MB)')
    return zpath
