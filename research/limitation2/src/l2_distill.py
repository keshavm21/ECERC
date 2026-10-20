"""
ECERC Limitation 2, Experiment 8b: distilling the zero-shot LLM into ECERC.

New code for research/limitation2. Experiment 7 raised test macro-F1 by fusing ECERC with a frozen zero-shot LLM at
inference. Here a single ECERC model (unchanged architecture, features, slot routing and recipe) is trained with an
added KL term towards soft labels, and evaluated alone; no LLM runs at test time:

    L = (1 - alpha) * L_orig(y) + alpha * T^2 * mean KL(q_T || p_T)

Teachers (fixed before any student is trained; label-free for the item they describe):
  T1  the zero-shot LLM's training-split log-probabilities from notebook 8a, prior-matched with Experiment 7's frozen q
  T2  0.8 * log p_ECERC-oof + 0.2 * log T1 (Experiment 7's w), where p_ECERC-oof comes from ECERC models trained on the
      other four of five folds of training dialogues (last epoch, three seeds), so dev is not used
Stage 1, dev only: D0 (control) and D1-D5 (teacher x alpha x T) and two label-smoothing controls, 5 seeds each. The
winner (D1-D5) has the highest mean best-epoch dev macro-F1; stage 2 runs only if the CROSS-FITTED dev gain (winner
and every epoch chosen without the fold they are scored on) is at least CONFIG['gate_min_gain'].
Freeze: teachers (SHA-256), winner, seeds and criteria go to frozen_config_distill.json before any test access.
Stage 2, confirmatory: fresh seeds x {D0, winner}; each run keeps its best-dev-macro-F1 epoch; the test set is
evaluated once at the end. This is the fourth test look: efficacy and FAILURE are judged on the Bonferroni-adjusted
interval (PROTOCOL.md, Experiment 8 addendum).
"""
import hashlib
import json
import math
import os
import platform
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

import l2_posthoc as P   # safe loader, data gate, dataset/model builders, inference and metrics (unchanged)
import l2_pathb as LB    # data download, seeding, hierarchical bootstrap and verdict rules (unchanged)

C, CLASSES = P.C, P.CLASSES
NEU, SUR, FEA, SAD, JOY, DIS, ANG = range(C)
BASE = dict(name='D0_original', teacher=None, alpha=0.0, T=1.0)
CANDIDATES = [
    BASE,
    dict(name='D1_T1_a0.3', teacher='T1', alpha=0.3, T=1.0),
    dict(name='D2_T1_a0.7', teacher='T1', alpha=0.7, T=1.0),
    dict(name='D3_T2_a0.3', teacher='T2', alpha=0.3, T=1.0),
    dict(name='D4_T2_a0.7', teacher='T2', alpha=0.7, T=1.0),
    dict(name='D5_T2_a0.7_T2', teacher='T2', alpha=0.7, T=2.0),
    dict(name='L1_uniform_a0.3', teacher='uniform', alpha=0.3, T=1.0),
    dict(name='L2_uniform_a0.7', teacher='uniform', alpha=0.7, T=1.0),
]
ELIGIBLE = [c['name'] for c in CANDIDATES if c['name'][0] == 'D' and c is not BASE]
CONTROLS_LS = [c['name'] for c in CANDIDATES if c['name'][0] == 'L']
CRITERIA = dict(
    LB.CRITERIA,
    primary='mean over stage-2 seeds of test macro-F1(winner) - macro-F1(D0), paired by seed, in points',
    success=['delta >= +1.0', 'Bonferroni-adjusted (test_looks) hierarchical-bootstrap interval lower bound > 0',
             'mean F1 change > 0 for Fear and for Disgust', 'delta > 0 in at least 70% of seeds',
             'accuracy change >= -1.0', 'weighted-F1 change >= -0.5', 'every class: mean F1 change >= -2.0'],
    failure='Bonferroni-adjusted interval upper bound < +1.0',
    note='fourth test look: efficacy and FAILURE use the adjusted interval; the 95% interval is reported')


def _metrics(y, logp):
    m = P.scores(LB._cm(y, logp.argmax(1)))
    return dict(acc=100 * float(m['acc']), weighted_f1=100 * float(m['wf1']), macro_f1=100 * float(m['mf1']),
                f1=100 * m['f1'], predicted=np.bincount(logp.argmax(1), minlength=C))


def _renorm(s):
    return s - np.logaddexp.reduce(s, axis=1, keepdims=True)


def prior_match(lp, log_q, log_prior):  # Experiment 7's label-free calibration (l2_zeroshot.prior_match)
    return _renorm(lp - log_q[None, :] + log_prior[None, :])


def _flat(S, split, per_vid):
    return np.concatenate([per_vid[v] for v in S['keys'][split]])


def _per_vid(S, split, X):
    out, k = {}, 0
    for v in S['keys'][split]:
        n = len(S['D']['videoAudio'][v])
        out[v] = X[k:k + n]
        k += n
    assert k == len(X)
    return out


def _subset(S, split, keys):
    ds = P.make_dataset(S, split)
    ds.keys, ds.len = list(keys), len(keys)
    return ds


# ----------------------------------------------------------------------------- setup and data
def stage_setup(CONFIG, code_dir):
    out = os.path.join(CONFIG['work_dir'], 'outputs_distill')
    for sub in ('', 'tables', 'dumps'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    os.makedirs(os.path.join(CONFIG['cache_dir'], 'checkpoints'), exist_ok=True)
    import pandas
    import sklearn
    S = dict(cfg=CONFIG, out=out, code_dir=code_dir, checks=[], torch=torch)
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
                    numpy=np.__version__, pandas=pandas.__version__, sklearn=sklearn.__version__, device=S['device'],
                    gpu=torch.cuda.get_device_name(0) if cuda else None,
                    code_sha256={f: P.sha256_file(os.path.join(code_dir, f))
                                 for f in sorted(os.listdir(code_dir)) if f.endswith('.py')})
    P.write_json(os.path.join(out, 'environment.json'), S['env'])
    P.write_json(os.path.join(out, 'design.json'),  # written before any data is loaded
                 dict(candidates=CANDIDATES, eligible=ELIGIBLE, label_smoothing_controls=CONTROLS_LS,
                      loss='(1 - alpha) * L_orig(y) + alpha * T^2 * mean KL(q_T || p_T)',
                      teachers=dict(T1='prior-matched zero-shot LLM (notebook 8a), q = Experiment 7 frozen log_q_dev',
                                    T2=f'{1 - CONFIG["t2_w"]} * log p_ECERC-oof + {CONFIG["t2_w"]} * log T1',
                                    oof=dict(folds=CONFIG['oof_folds'], fold_seed=CONFIG['oof_fold_seed'],
                                             seeds=CONFIG['oof_seeds'], epochs=CONFIG['epochs'],
                                             checkpoint='last epoch (dev not used)')),
                      epochs=CONFIG['epochs'], stage1_seeds=CONFIG['stage1_seeds'], stage2_seeds=CONFIG['stage2_seeds'],
                      gate=dict(statistic='cross-fitted dev gain (winner and epochs chosen without the scored fold)',
                                folds=CONFIG['n_folds'], fold_seed=CONFIG['fold_seed'], threshold=CONFIG['gate_min_gain']),
                      test_looks=CONFIG['test_looks'], criteria=CRITERIA))
    if not cuda:
        print('WARNING: no GPU. Training will be very slow; switch the accelerator to GPU and run again.')
    P.log_event(S, 'setup', device=S['device'], gpu=S['env']['gpu'], torch=torch.__version__)
    return S


def stage_prepare(S):
    """MELD features and the data gate (train/dev labels only); train and dev datasets."""
    LB.stage_download(S)
    P.stage_load_data(S)
    S['datasets'] = {s: P.make_dataset(S, s) for s in ('train', 'valid')}
    S['y'] = {s: np.concatenate([np.asarray(S['D']['emotion_labels'][v], dtype=np.int64) for v in S['keys'][s]])
              for s in ('train', 'valid')}
    P._save_checks(S)
    P.log_event(S, 'datasets ready', train_utterances=len(S['y']['train']), dev_utterances=len(S['y']['valid']))


# ----------------------------------------------------------------------------- teachers
def _file(S, key_path, key_sha, embedded):
    cfg = S['cfg']
    path = cfg.get(key_path) or os.path.join(S['code_dir'], embedded)
    if not os.path.exists(path):
        raise P.CheckFailed(f'{embedded} is missing: run notebook 8a, add its results to the repository and rebuild')
    got = P.sha256_file(path)
    if cfg.get(key_sha):
        P.record_check(S, f'{embedded}: SHA-256 matches the pre-recorded file', got == cfg[key_sha], hard=True,
                       detail=got[:16])
    else:
        P.record_check(S, f'{embedded}: SHA-256 pinned', bool(cfg.get('offline')), hard=True,
                       detail='no expected SHA-256 (allowed only in the offline smoke test)')
    return path, got


def stage_teacher(S):
    """T1 from notebook 8a's training-split log-probabilities; diagnostics of T1 on train and dev."""
    cfg = S['cfg']
    path, S['teacher_file_sha256'] = _file(S, 'teacher_local_path', 'teacher_sha256', 'train_llm.npz')
    z = np.load(path)
    lengths = np.array([len(S['D']['videoAudio'][v]) for v in S['keys']['train']])
    P.record_check(S, 'teacher file was computed from the same data file', str(z['data_sha256']) == S['data_sha256'],
                   hard=True, detail=str(z['data_sha256'])[:16])
    P.record_check(S, 'teacher rows align with the training dialogues (keys, order and lengths)',
                   [str(k) for k in z['keys']] == [str(k) for k in S['keys']['train']]
                   and np.array_equal(z['lengths'], lengths) and len(z['logp']) == int(lengths.sum()), hard=True)
    lp_tr = z['logp'].astype(np.float64)
    P.record_check(S, 'teacher log-probabilities are finite and normalised',
                   np.isfinite(lp_tr).all() and np.allclose(np.logaddexp.reduce(lp_tr, axis=1), 0, atol=1e-3), hard=True)
    dpath, S['dev_llm_sha256'] = _file(S, 'dev_llm_local_path', 'dev_llm_sha256', 'dev_llm.npz')
    zd = np.load(dpath)
    lp_dv = zd['logp'].astype(np.float64)
    P.record_check(S, 'Experiment 7 dev log-probabilities align with the dev utterances (count and labels)',
                   len(lp_dv) == len(S['y']['valid']) and np.array_equal(zd['y'], S['y']['valid']), hard=True,
                   detail=len(lp_dv))
    log_q = np.log(np.exp(lp_dv).mean(0))
    if cfg.get('log_q_dev') is not None:
        diff = float(np.abs(log_q - np.asarray(cfg['log_q_dev'])).max())
        P.record_check(S, "recomputed q equals Experiment 7's frozen log_q_dev", diff < 1e-4, hard=True,
                       detail=f'max |diff| {diff:.2e}')
    S['log_q'] = log_q
    t1_tr, t1_dv = prior_match(lp_tr, log_q, S['log_prior']), prior_match(lp_dv, log_q, S['log_prior'])
    S['t1'] = dict(train=t1_tr, valid=t1_dv)
    q_train = np.log(np.exp(lp_tr).mean(0))
    S['teacher_diag'] = dict(
        llm_raw=dict(train=_metrics(S['y']['train'], lp_tr), dev=_metrics(S['y']['valid'], lp_dv)),
        T1=dict(train=_metrics(S['y']['train'], t1_tr), dev=_metrics(S['y']['valid'], t1_dv)),
        log_q_dev=log_q, log_q_train_label_free=q_train)
    d = S['teacher_diag']['T1']
    print(f'  T1 (calibrated LLM) macro-F1: train {d["train"]["macro_f1"]:.2f}, dev {d["dev"]["macro_f1"]:.2f} | '
          'train per class ' + ' '.join(f'{c} {f:.1f}' for c, f in zip(CLASSES, d['train']['f1'])))
    P._save_checks(S)
    P.log_event(S, 'teacher T1 ready', train_macro_f1=round(d['train']['macro_f1'], 2),
                dev_macro_f1=round(d['dev']['macro_f1'], 2))


# ----------------------------------------------------------------------------- training (train/dev only)
def train_run(S, cand, seed, epochs, train_ds=None, track_dev=True):
    """One run of the unchanged ECERC model and recipe (Path B's C0 code path) plus the candidate's KL term.
    With track_dev, keeps the epoch with the best dev macro-F1 (earliest on ties) and every epoch's dev predictions;
    without, keeps the last epoch and never touches dev. Never touches the test split."""
    from torch.utils.data import DataLoader
    dev = S['device']
    LB.seed_everything(seed)
    model = P.build_model(S).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=2e-4)  # MELD/train.py: Adam, l2 = 0.0002
    loss_f = S['m_loss'].Loss(alpha=None)  # original MELD objective (focal-style, gamma = 1, no class weights)
    ds = train_ds if train_ds is not None else S['datasets']['train']
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=ds.collate_fn, num_workers=0)
    teach = S['teachers'][cand['teacher']] if cand['alpha'] > 0 and cand['teacher'] != 'uniform' else None
    alpha, T = cand['alpha'], cand['T']
    curve, preds, best, t0 = [], [], None, time.time()
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
            if alpha > 0:
                if teach is None:  # label smoothing: a uniform target
                    tq = torch.zeros_like(log_prob)
                else:
                    tq = torch.as_tensor(np.concatenate([teach[v][:n] for v, n in zip(data[-1], seq)]),
                                         dtype=log_prob.dtype, device=dev)
                lq, lp = F.log_softmax(tq / T, dim=1), F.log_softmax(log_prob / T, dim=1)
                kl = (lq.exp() * (lq - lp)).sum(1).mean()
                loss = (1 - alpha) * loss + alpha * T * T * kl
            loss.backward()
            opt.step()
        if not track_dev:
            continue
        d = P.run_inference(S, model, S['datasets']['valid'])
        p = d['logp'].argmax(1)
        preds.append(p.astype(np.uint8))
        m = P.scores(LB._cm(d['y'], p))
        curve.append(dict(epoch=epoch, macro_f1=100 * float(m['mf1']), weighted_f1=100 * float(m['wf1']),
                          acc=100 * float(m['acc'])))
        if best is None or 100 * float(m['mf1']) > best['dev_macro_f1'] + 1e-9:
            best = dict(epoch=epoch, dev_macro_f1=100 * float(m['mf1']), dev_weighted_f1=100 * float(m['wf1']),
                        dev_acc=100 * float(m['acc']), dev_f1=100 * m['f1'], dev_logp=d['logp'],
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    if not track_dev:
        return dict(model=model.eval(), seed=seed, seconds=round(time.time() - t0, 1))
    best.update(variant=cand['name'], seed=seed, seconds=round(time.time() - t0, 1), curve=curve,
                dev_preds=np.stack(preds))
    return best


def stage_oof(S):
    """T2: cross-fitted ECERC probabilities on the training split, fused with T1 by Experiment 7's weight."""
    cfg, D, t0 = S['cfg'], S['D'], time.time()
    keys = list(S['keys']['train'])
    rng = np.random.default_rng(cfg['oof_fold_seed'])
    parts = np.array_split(rng.permutation(len(keys)), cfg['oof_folds'])
    oof, dev_probs = {}, []
    for k, part in enumerate(parts):
        held = [keys[i] for i in sorted(part)]
        held_set = set(held)
        fit = [v for v in keys if v not in held_set]
        probs = []
        for sd in cfg['oof_seeds']:
            r = train_run(S, BASE, sd, cfg['epochs'], train_ds=_subset(S, 'train', fit), track_dev=False)
            d = P.run_inference(S, r['model'], _subset(S, 'train', held))
            probs.append(np.exp(d['logp']))
            dev_probs.append(np.exp(P.run_inference(S, r['model'], S['datasets']['valid'])['logp']))
            print(f'  [teacher T2] fold {k + 1}/{len(parts)} seed {sd}: trained on {len(fit)} dialogues, '
                  f'{r["seconds"]:.0f}s', flush=True)
        lp = np.log(np.mean(probs, 0))
        for v, x in _per_vid(dict(S, keys=dict(train=held)), 'train', lp).items():
            oof[v] = x
    lp_oof = _flat(S, 'train', oof)
    w = cfg['t2_w']
    t2_tr = _renorm((1 - w) * lp_oof + w * S['t1']['train'])
    ecerc_dev = np.log(np.mean(dev_probs, 0))
    t2_dv = _renorm((1 - w) * ecerc_dev + w * S['t1']['valid'])
    S['t2'] = dict(train=t2_tr, valid=t2_dv)
    S['teachers'] = dict(T1=_per_vid(S, 'train', S['t1']['train']), T2=_per_vid(S, 'train', t2_tr))
    S['teacher_arrays_sha256'] = hashlib.sha256(np.concatenate([S['t1']['train'], t2_tr], 1)
                                                .astype(np.float32).tobytes()).hexdigest()
    ytr, ydv = S['y']['train'], S['y']['valid']
    S['teacher_diag'].update(
        ecerc_oof=dict(train=_metrics(ytr, lp_oof)), T2=dict(train=_metrics(ytr, t2_tr), dev=_metrics(ydv, t2_dv)),
        ecerc_fold_models_dev=_metrics(ydv, ecerc_dev), teacher_arrays_sha256=S['teacher_arrays_sha256'])
    P.write_json(os.path.join(S['out'], 'teachers.json'), S['teacher_diag'])
    np.savez_compressed(os.path.join(S['out'], 'dumps', 'teachers_train.npz'), t1=S['t1']['train'].astype(np.float32),
                        t2=t2_tr.astype(np.float32), ecerc_oof=lp_oof.astype(np.float32), y=ytr)
    td = S['teacher_diag']
    print('\nTEACHERS (train and dev only), macro-F1:')
    for name, split in (('T1 calibrated LLM', 'train'), ('T1 calibrated LLM', 'dev'), ('ECERC cross-fitted', 'train'),
                        ('T2', 'train'), ('T2', 'dev')):
        m = td[{'T1 calibrated LLM': 'T1', 'ECERC cross-fitted': 'ecerc_oof', 'T2': 'T2'}[name]][split]
        print(f'  {name:<20s} {split:<5s} {m["macro_f1"]:6.2f} | ' + ' '.join(f'{c} {f:.1f}' for c, f in zip(CLASSES, m['f1'])))
    P._save_checks(S)
    P.log_event(S, 'teacher T2 ready', minutes=round((time.time() - t0) / 60, 1),
                t2_train_macro_f1=round(td['T2']['train']['macro_f1'], 2), sha256=S['teacher_arrays_sha256'][:16])


def check_eval_path(S, r):
    """Once: our evaluation loop must reproduce the original train_or_eval_model on dev for a trained model."""
    from sklearn.metrics import confusion_matrix
    model = P.build_model(S)
    model.load_state_dict(r['state'], strict=True)
    model.to(S['device']).eval()
    ours = P.run_inference(S, model, S['datasets']['valid'])
    _, _, cm_o = P.original_function_eval(S, model, S['datasets']['valid'])
    cm = confusion_matrix(ours['y'], ours['logp'].argmax(1))
    diff = int(np.abs(cm_o - cm).sum()) if cm_o.shape == cm.shape else -1
    P.record_check(S, 'evaluation loop reproduces the original train_or_eval_model on dev', 0 <= diff <= 4,
                   hard=True, detail=f'sum |confusion diff| = {diff}')
    return ours


def _kl_dev(teacher_lp, student_lp):
    return float((np.exp(teacher_lp) * (teacher_lp - student_lp)).sum(1).mean())


def crossfit_gate(S, results, n_folds, seed, power_B):
    """Cross-fitted dev gain of the whole stage-1 procedure (winner among ELIGIBLE and every run's epoch chosen on the
    other folds) over D0 (epoch chosen the same way), pooled per seed; plus a projected test SE."""
    y, dlg = S['dev_y'], S['dev_dlg']
    n_dlg = int(dlg.max()) + 1
    cms = {c: [P.per_dialogue_cms(y, r['dev_preds'].astype(np.int64), dlg, n_dlg) for r in rs]
           for c, rs in results.items()}  # name -> per seed (n_dlg, epochs, 49)
    ns = len(results[BASE['name']])
    folds = np.array_split(np.random.default_rng(seed).permutation(n_dlg), n_folds)
    contrib = np.zeros((n_dlg, ns, 2, C * C))  # per dialogue: held-out counts of (control, procedure)
    chosen = []
    for f in folds:
        keep = np.ones(n_dlg, bool)
        keep[f] = False

        def pick(name):
            ep, sc = [], []
            for x in cms[name]:
                m = P.scores(x[keep].sum(0).reshape(-1, C, C))['mf1']
                e = int(P.first_argmax(m))
                ep.append(e)
                sc.append(m[e])
            return ep, float(np.mean(sc))

        picks = {c: pick(c) for c in ELIGIBLE + [BASE['name']]}
        win = ELIGIBLE[0]
        for c in ELIGIBLE[1:]:
            if picks[c][1] > picks[win][1] + 1e-12:
                win = c
        chosen.append(win)
        for s in range(ns):
            contrib[f, s, 0] = cms[BASE['name']][s][f, picks[BASE['name']][0][s]]
            contrib[f, s, 1] = cms[win][s][f, picks[win][0][s]]

    def gain(w=None):
        x = contrib if w is None else contrib * w[:, None, None, None]
        m = P.scores(x.sum(0).reshape(ns, 2, C, C))['mf1']
        return 100 * float((m[:, 1] - m[:, 0]).mean())

    rng = np.random.default_rng(seed + 1)
    boot = np.array([gain(P._boot_weights(rng, n_dlg, 1)[0]) for _ in range(power_B)])
    se = float(boot.std(ddof=1)) * math.sqrt(n_dlg / S['n_test_dlg'])
    z_adj = 2.4977  # two-sided 98.75% (Bonferroni, four looks)
    return dict(crossfit_delta_macro_f1=gain(), fold_winner=chosen, dev_bootstrap_ci95=np.percentile(boot, [2.5, 97.5]),
                se_test_projected=se, mde80_projected_95=(1.96 + 0.8416) * se, mde80_projected_adjusted=(z_adj + 0.8416) * se)


def stage_screen(S):
    cfg, t0 = S['cfg'], time.time()
    S['n_test_dlg'] = len(S['keys']['test'])  # a count of dialogue keys only; no test label is read
    teacher_dev = dict(T1=S['t1']['valid'], T2=S['t2']['valid'])
    results, means = {}, {}
    for cand in CANDIDATES:
        res = []
        for sd in cfg['stage1_seeds']:
            r = train_run(S, cand, sd, cfg['epochs'])
            print(LB._run_line('stage 1', r), flush=True)
            if cand is BASE and sd == cfg['stage1_seeds'][0]:
                ref = check_eval_path(S, r)
                S['dev_y'], S['dev_dlg'] = ref['y'], ref['dlg']
            r['kl_dev'] = {t: _kl_dev(x, r['dev_logp']) for t, x in teacher_dev.items()}
            r.pop('state')
            r.pop('dev_logp')
            res.append(r)
        results[cand['name']] = res
        means[cand['name']] = float(np.mean([r['dev_macro_f1'] for r in res]))
        P.log_event(S, 'stage 1 candidate done', candidate=cand['name'], mean_dev_macro_f1=round(means[cand['name']], 2))
    ref = cfg.get('a0_dev_macro_f1') or {}
    pairs = [(r['seed'], r['dev_macro_f1'], ref[r['seed']]) for r in results[BASE['name']] if r['seed'] in ref]
    if pairs:
        P.record_check(S, "D0 reproduces Experiment 6's A0 on dev for the shared seeds (same code path; library versions "
                          "may differ)", all(abs(a - b) < 0.01 for _, a, b in pairs), hard=False,
                       detail=', '.join(f'seed {s}: {a:.2f} vs {b:.2f}' for s, a, b in pairs))
    base = means[BASE['name']]
    winner = ELIGIBLE[0]
    for c in ELIGIBLE[1:]:
        if means[c] > means[winner] + 1e-12:
            winner = c
    gate = crossfit_gate(S, results, cfg['n_folds'], cfg['fold_seed'], cfg['power_B'])
    proceed = gate['crossfit_delta_macro_f1'] >= cfg['gate_min_gain']
    best_ls = max(CONTROLS_LS, key=lambda c: means[c])
    S['stage1'] = dict(results=results, means=means, winner=next(c for c in CANDIDATES if c['name'] == winner),
                       in_sample_gain=means[winner] - base, gate=gate, proceed=proceed,
                       winner_vs_best_ls=dict(control=best_ls, delta=means[winner] - means[best_ls]))
    rows = []
    for name, rs in results.items():
        for r in rs:
            row = dict(candidate=name, seed=r['seed'], best_epoch=r['epoch'], dev_macro_f1=r['dev_macro_f1'],
                       dev_weighted_f1=r['dev_weighted_f1'], dev_acc=r['dev_acc'], kl_dev_T1=r['kl_dev']['T1'],
                       kl_dev_T2=r['kl_dev']['T2'], seconds=r['seconds'])
            row.update({f'F1_{CLASSES[c]}': r['dev_f1'][c] for c in range(C)})
            rows.append(row)
    P.write_csv(os.path.join(S['out'], 'tables', 'stage1_runs.csv'), rows)
    np.savez_compressed(os.path.join(S['out'], 'dumps', 'stage1_dev_preds.npz'), y=S['dev_y'], dlg=S['dev_dlg'],
                        **{f'{n}__seed{r["seed"]}': r['dev_preds'] for n, rs in results.items() for r in rs})
    st = S['stage1']
    P.write_json(os.path.join(S['out'], 'stage1_summary.json'),
                 dict(mean_dev_macro_f1=means, winner=st['winner'], in_sample_gain_over_D0=st['in_sample_gain'],
                      gate=dict(gate, threshold=cfg['gate_min_gain'], proceed=proceed),
                      winner_vs_best_label_smoothing=st['winner_vs_best_ls'],
                      mean_kl_dev={n: {t: float(np.mean([r['kl_dev'][t] for r in rs])) for t in ('T1', 'T2')}
                                   for n, rs in results.items()},
                      mean_dev_f1={n: np.mean([r['dev_f1'] for r in rs], 0) for n, rs in results.items()},
                      curves={n: [r['curve'] for r in rs] for n, rs in results.items()}))
    print('\nSTAGE 1 (dev only): mean best-epoch dev macro-F1 over seeds')
    for name in results:
        f1 = np.mean([r['dev_f1'] for r in results[name]], 0)
        print(f'  {name:<18s} {means[name]:6.2f}  ({means[name] - base:+.2f} vs D0) | Fear {f1[FEA]:.1f} Disgust '
              f'{f1[DIS]:.1f} Sadness {f1[SAD]:.1f} Joy {f1[JOY]:.1f}')
    print(f'  winner (D1-D5): {winner} ({st["in_sample_gain"]:+.2f} in sample); vs best label-smoothing control '
          f'{best_ls}: {st["winner_vs_best_ls"]["delta"]:+.2f}')
    print(f'  cross-fitted dev gain {gate["crossfit_delta_macro_f1"]:+.2f} (fold winners {gate["fold_winner"]}); '
          f'projected test SE {gate["se_test_projected"]:.2f}, MDE80 at 98.75% {gate["mde80_projected_adjusted"]:.2f}; '
          f'gate {cfg["gate_min_gain"]:+.2f} -> ' + ('stage 2 will run' if proceed else 'stage 2 skipped, test stays sealed'))
    P._save_checks(S)
    P.log_event(S, 'stage 1 done', minutes=round((time.time() - t0) / 60, 1), winner=winner,
                crossfit_gain=round(gate['crossfit_delta_macro_f1'], 2), proceed=proceed)


# ----------------------------------------------------------------------------- freeze, confirmation, test
def stage_freeze(S):
    st, cfg = S['stage1'], S['cfg']
    frozen = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),
                  note='Written before any test-set access. Stage 2 and the test evaluation read their rules from here.',
                  code_sha256=S['env']['code_sha256'], data_sha256=S['data_sha256'],
                  teacher_file_sha256=S['teacher_file_sha256'], dev_llm_sha256=S['dev_llm_sha256'],
                  teacher_arrays_sha256=S['teacher_arrays_sha256'], log_q_dev=S['log_q'], t2_w=cfg['t2_w'],
                  control=BASE, winner=st['winner'], stage1_mean_dev_macro_f1=st['means'],
                  gate=dict(st['gate'], threshold=cfg['gate_min_gain'], proceed=st['proceed']),
                  stage2_seeds=cfg['stage2_seeds'], epochs=cfg['epochs'],
                  checkpoint_rule='per run, the epoch with the best dev macro-F1 (earliest on ties)',
                  bootstrap=dict(B=cfg['bootstrap_B'], seed=cfg['boot_seed'],
                                 design='hierarchical: seeds and test dialogues resampled; seed-level t-intervals also reported'),
                  multiplicity=dict(test_looks=cfg['test_looks'],
                                    note='efficacy (lower bound > 0) and FAILURE (upper bound < +1.0) use the '
                                         'Bonferroni-adjusted interval; the 95% interval is reported'),
                  criteria=CRITERIA, device=S['device'], env=S['env'])
    path = os.path.join(S['out'], 'frozen_config_distill.json')
    P.write_json(path, frozen)
    S['frozen_path'], S['frozen_sha256'] = path, P.sha256_file(path)
    P.log_event(S, 'FREEZE', sha256=S['frozen_sha256'][:16], winner=st['winner']['name'], proceed=st['proceed'])


def _load_frozen(S):
    if not S.get('frozen_path') or P.sha256_file(S['frozen_path']) != S['frozen_sha256']:
        raise P.CheckFailed('frozen_config_distill.json is missing or was modified: run the freeze cell first')
    with open(S['frozen_path']) as f:
        F_ = json.load(f)
    if any(P.sha256_file(os.path.join(S['code_dir'], n)) != h for n, h in F_['code_sha256'].items()):
        raise P.CheckFailed('code changed after the freeze')
    sha = hashlib.sha256(np.concatenate([_flat(S, 'train', S['teachers']['T1']), _flat(S, 'train', S['teachers']['T2'])],
                                        1).astype(np.float32).tobytes()).hexdigest()
    if sha != F_['teacher_arrays_sha256']:
        raise P.CheckFailed('teachers changed after the freeze')
    return F_


def stage_confirm(S):
    F_ = _load_frozen(S)
    S['stage2'] = None
    if not F_['gate']['proceed']:
        print('Stage 2 skipped by the pre-registered gate: the cross-fitted dev gain is below the threshold.')
        return
    arms = [('control', F_['control']), ('winner', F_['winner'])]
    runs, t0 = {a: [] for a, _ in arms}, time.time()
    for sd in F_['stage2_seeds']:
        for arm, v in arms:
            r = train_run(S, v, sd, F_['epochs'])
            print(LB._run_line('stage 2', r), flush=True)
            path = os.path.join(S['cfg']['cache_dir'], 'checkpoints', f'{arm}_seed{sd}.pt')
            torch.save(r.pop('state'), path)
            r.pop('dev_logp')
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
    test_ds = P.make_dataset(S, 'test')
    n_te = len(S['keys']['test'])
    cms = np.zeros((n_te, len(seeds), 2, C * C), dtype=np.int64)
    for a, (arm, v) in enumerate(arms):
        for i, r in enumerate(S['stage2'][arm]):
            model = P.build_model(S)
            model.load_state_dict(torch.load(r['checkpoint'], map_location=S['device'], weights_only=True), strict=True)
            model.to(S['device']).eval()
            d = P.run_inference(S, model, test_ds)
            cms[:, i, a, :] = P.per_dialogue_cms(d['y'], d['logp'].argmax(1)[None, :], d['dlg'], n_te)[:, 0, :]
            np.savez_compressed(os.path.join(S['out'], 'dumps', f'test_{arm}_seed{seeds[i]}.npz'),
                                logp=d['logp'].astype(np.float32), y=d['y'], dlg=d['dlg'])
    sc = P.scores(cms.sum(0).reshape(len(seeds), 2, C, C))
    diff = {k: 100 * (sc[k][:, 1] - sc[k][:, 0]) for k in ('mf1', 'acc', 'wf1', 'f1', 'prec', 'rec')}
    dl = {k: v.mean(0) for k, v in diff.items()}
    per_seed = diff['mf1']
    bt = LB.hierarchical_bootstrap(cms, F_['bootstrap']['B'], F_['bootstrap']['seed'])
    ci = lambda x, q=(2.5, 97.5): np.percentile(x, q, axis=0)  # noqa: E731
    lo, hi = ci(bt['mf1'])
    alpha = 0.05 / F_['multiplicity']['test_looks']
    b_lo, b_hi = ci(bt['mf1'], (100 * alpha / 2, 100 * (1 - alpha / 2)))
    # pre-registered for the fourth look: efficacy and FAILURE are judged on the adjusted interval
    verdict, conditions = LB.decide(float(dl['mf1']), b_lo, b_hi, dl['f1'], float(dl['acc']), float(dl['wf1']),
                                    float((per_seed > 0).mean()))
    conditions['adjusted_ci_lower_gt_0'] = conditions.pop('ci_lower_gt_0')
    verdict_95, _ = LB.decide(float(dl['mf1']), lo, hi, dl['f1'], float(dl['acc']), float(dl['wf1']),
                              float((per_seed > 0).mean()))
    consistency = {}
    for name, x in (('macro_f1', diff['mf1']), ('weighted_f1', diff['wf1']), ('accuracy', diff['acc']),
                    ('fear_f1', diff['f1'][:, FEA]), ('disgust_f1', diff['f1'][:, DIS]), ('joy_f1', diff['f1'][:, JOY])):
        consistency[name] = dict(mean=float(x.mean()), seed_t_ci95=_t_interval(x), seeds_positive=int((x > 0).sum()),
                                 seeds=len(seeds))
    arm_means = {arm: dict(acc=100 * sc['acc'][:, a].mean(), macro_f1=100 * sc['mf1'][:, a].mean(),
                           weighted_f1=100 * sc['wf1'][:, a].mean(), f1=100 * sc['f1'][:, a].mean(0),
                           precision=100 * sc['prec'][:, a].mean(0), recall=100 * sc['rec'][:, a].mean(0),
                           macro_f1_sd=100 * sc['mf1'][:, a].std(ddof=1) if len(seeds) > 1 else None)
                 for a, (arm, _) in enumerate(arms)}
    S['results'] = dict(
        verdict=verdict, conditions=conditions, winner=F_['winner'], seeds=seeds,
        delta_macro_f1=float(dl['mf1']), ci95_hierarchical=[lo, hi], ci95_seed_t=_t_interval(per_seed),
        multiplicity=dict(test_looks=F_['multiplicity']['test_looks'], level=100 * (1 - alpha),
                          ci_hierarchical=[b_lo, b_hi], lower_bound_above_0=bool(b_lo > 0),
                          verdict_if_judged_at_95=verdict_95),
        per_seed_delta=per_seed, seeds_positive=int((per_seed > 0).sum()), consistency=consistency,
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
    P.write_json(os.path.join(S['out'], 'test_results_distill.json'), S['results'])
    _write_summary(S, F_)
    P.log_event(S, 'TEST done', verdict=verdict, delta=round(float(dl['mf1']), 2),
                ci_adjusted=[round(float(b_lo), 2), round(float(b_hi), 2)])


def _write_summary(S, F_):
    st, env, td = S['stage1'], S['env'], S['teacher_diag']
    g = st['gate']
    L = ['# ECERC Limitation 2, Experiment 8: distilling the zero-shot LLM into ECERC', '',
         f'Run {time.strftime("%Y-%m-%d %H:%M")} | {env["device"]} ({env["gpu"]}) | torch {env["torch"]} | '
         f'data sha256 {S["data_sha256"][:16]}... | teacher file sha256 {S["teacher_file_sha256"][:16]}...', '',
         '## Teachers (train and dev only)', '', '| teacher | split | macro-F1 | ' + ' | '.join(f'F1 {c}' for c in CLASSES) + ' |',
         '|---' * (3 + C) + '|']
    for name, key, split in (('T1, calibrated LLM', 'T1', 'train'), ('T1, calibrated LLM', 'T1', 'dev'),
                             ('ECERC cross-fitted', 'ecerc_oof', 'train'), ('T2', 'T2', 'train'), ('T2', 'T2', 'dev')):
        m = td[key][split]
        L.append(f'| {name} | {split} | {m["macro_f1"]:.2f} | ' + ' | '.join(f'{x:.1f}' for x in m['f1']) + ' |')
    L += ['', '## Stage 1: dev screen (mean best-epoch dev macro-F1 over seeds; selection only, not a result)', '',
          '| candidate | dev macro-F1 | vs D0 |', '|---|---|---|']
    base = st['means'][BASE['name']]
    for name, m in st['means'].items():
        L.append(f'| {name} | {m:.2f} | {m - base:+.2f} |')
    L += ['', f'Winner (D1-D5): **{st["winner"]["name"]}** ({st["in_sample_gain"]:+.2f} in sample; '
              f'{st["winner_vs_best_ls"]["delta"]:+.2f} vs {st["winner_vs_best_ls"]["control"]})',
          f'Cross-fitted dev gain **{g["crossfit_delta_macro_f1"]:+.2f}** (gate {S["cfg"]["gate_min_gain"]:+.2f}: '
          + ('passed, stage 2 ran' if st['proceed'] else 'not passed, stage 2 skipped and the test not evaluated')
          + f'); projected test SE {g["se_test_projected"]:.2f}, MDE80 at the adjusted level {g["mde80_projected_adjusted"]:.2f}', '']
    res = S.get('results')
    if res:
        mu = res['multiplicity']
        L += [f'## Verdict: {res["verdict"]}', '', f'(judged on the {mu["level"]:.2f}% interval, as pre-registered for '
              f'test look {mu["test_looks"]}; at 95% it would be: {mu["verdict_if_judged_at_95"]})', '',
              '| condition | met |', '|---|---|']
        L += [f'| {k} | {"yes" if v else "no"} |' for k, v in res['conditions'].items()]
        a, w = res['arms']['control'], res['arms']['winner']
        L += ['', f'## Test, {len(res["seeds"])} seeds per arm (mean)', '',
              '| arm | acc | weighted F1 | macro-F1 (SD) | ' + ' | '.join(f'F1 {c}' for c in CLASSES) + ' |',
              '|---' * (4 + C) + '|']
        for name, m in (('control (D0)', a), (f'winner ({res["winner"]["name"]})', w)):
            L.append(f'| {name} | {m["acc"]:.2f} | {m["weighted_f1"]:.2f} | {m["macro_f1"]:.2f} ({m["macro_f1_sd"]:.2f}) | '
                     + ' | '.join(f'{x:.2f}' for x in m['f1']) + ' |')
        gr = res['guardrails']
        L += ['', f'- Delta macro-F1: **{res["delta_macro_f1"]:+.2f}**; {mu["level"]:.2f}% interval '
                  f'[{mu["ci_hierarchical"][0]:+.2f}, {mu["ci_hierarchical"][1]:+.2f}]; 95% interval '
                  f'[{res["ci95_hierarchical"][0]:+.2f}, {res["ci95_hierarchical"][1]:+.2f}] (hierarchical bootstrap); '
                  f'seed-level t-interval [{res["ci95_seed_t"][0]:+.2f}, {res["ci95_seed_t"][1]:+.2f}]',
              f'- positive in {res["seeds_positive"]}/{len(res["seeds"])} seeds: '
              + ', '.join(f'{x:+.2f}' for x in res['per_seed_delta']),
              f'- Delta accuracy {P._fmt(gr["delta_acc"], *gr["delta_acc_ci95"])}; '
              f'Delta weighted F1 {P._fmt(gr["delta_weighted_f1"], *gr["delta_weighted_f1_ci95"])}',
              '', '| metric (winner - D0, paired by seed) | mean | seed-level 95% t-interval | seeds positive |',
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
    with open(os.path.join(S['out'], 'SUMMARY_distill.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')


def stage_package(S):
    zpath = os.path.join(S['cfg']['work_dir'], 'ecerc_l2_distill_results.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(S['out']):
            for name in files:
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, os.path.dirname(S['out'])))
    with open(os.path.join(S['out'], 'SUMMARY_distill.md')) as f:
        print(f.read())
    print(f'Results archive: {zpath} ({os.path.getsize(zpath) / 2 ** 20:.1f} MB)')
    return zpath
