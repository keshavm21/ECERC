"""
ECERC Limitation 2, Path B: can training-time representation changes improve the rare emotions?

New code for research/limitation2. The original ECERC model, dataset class and loss are used unchanged.
The extra heads below exist only during training and are dropped afterwards, so every evaluated model is
the original architecture.

Stage 1, dev only, 3 seeds per candidate, against the original recipe (C0):
  C1  evidence and event inputs = mean of the four provided RoBERTa layers (same 1024-d shape) instead of layer 1
  C2  + confusion-aware prototype contrastive loss on the fused representation (weight 0.1 / 0.3 / 1.0)
  C3  + sentiment auxiliary head trained on MELD's own training sentiment labels (weight 0.3 / 1.0)
  C4  learning rate 3e-5 instead of 1e-5 (the released models under-fit their own training data)
  C5  the families (best member each) that beat C0 on dev, combined
The winner is the candidate with the highest mean dev macro-F1. Stage 2 runs only if it beats C0 by at least
CONFIG['gate_min_gain'] points.
Freeze: winner, seeds and criteria go to frozen_config_pathb.json before any test access.
Stage 2, confirmatory: fresh seeds x {C0, winner}, same initialisation and data order per seed; each run keeps
its best-dev-macro-F1 epoch; the test set is evaluated once at the end.
"""
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import l2_posthoc as P  # safe loader, data gate, dataset/model builders, inference and metrics (unchanged)

C, CLASSES = P.C, P.CLASSES
NEU, SUR, FEA, SAD, JOY, DIS, ANG = range(C)
CONFUSABLE = [(FEA, SAD), (FEA, ANG), (FEA, SUR), (DIS, ANG)]  # the confusions seen in run_colab_1
BASE = dict(name='C0_original', family='C0', features='layer1', lr=1e-5, lam_c=0.0, lam_s=0.0)
CANDIDATES = [
    BASE,
    dict(BASE, name='C1_mean4_layers', family='C1', features='mean4'),
    dict(BASE, name='C2_contrast_0.1', family='C2', lam_c=0.1),
    dict(BASE, name='C2_contrast_0.3', family='C2', lam_c=0.3),
    dict(BASE, name='C2_contrast_1.0', family='C2', lam_c=1.0),
    dict(BASE, name='C3_sentiment_0.3', family='C3', lam_s=0.3),
    dict(BASE, name='C3_sentiment_1.0', family='C3', lam_s=1.0),
    dict(BASE, name='C4_lr_3e-5', family='C4', lr=3e-5),
]
CRITERIA = dict(
    primary='mean over stage-2 seeds of test macro-F1(winner) - macro-F1(C0), paired by seed, in points',
    success=['delta >= +1.0', 'hierarchical-bootstrap 95% CI lower bound > 0',
             'mean F1 change > 0 for Fear and for Disgust', 'delta > 0 in at least 70% of seeds',
             'accuracy change >= -1.0', 'weighted-F1 change >= -0.5', 'every class: mean F1 change >= -2.0'],
    verdict_order=['SUCCESS', 'TRADE-OFF (efficacy met, a guardrail fails)',
                   'PARTIAL (macro-F1 target met, Fear/Disgust/consistency not)',
                   'FAILURE (CI upper bound < +1.0)', 'INCONCLUSIVE'],
    guardrails_judged_on='point estimates; 95% CIs reported')


def _cm(y, p):
    return np.bincount(y * C + p, minlength=C * C).reshape(C, C)


def seed_everything(seed):  # same as seed_everything() in MELD/train.py
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AuxHeads(nn.Module):
    """Training-only heads on the 1536-d fused representation that feeds ECERC's classifier (smax_fc).

    Contrastive part (SPCL-style): each utterance is pulled towards same-class utterances in the batch and
    towards an EMA prototype of its class, so rare classes always have a positive. Negatives from the
    confusable classes count conf_weight times in the denominator. Sentiment part: cross-entropy on MELD's
    training sentiment labels."""

    def __init__(self, lam_c, lam_s, n_sent, dim_in=1536, proj_dim=128, temperature=0.1, conf_weight=2.0,
                 momentum=0.9):
        super().__init__()
        self.lam_c, self.lam_s, self.t, self.momentum = lam_c, lam_s, temperature, momentum
        self.proj = nn.Sequential(nn.Linear(dim_in, 256), nn.ReLU(), nn.Linear(256, proj_dim)) if lam_c > 0 else None
        self.sent = nn.Linear(dim_in, n_sent) if lam_s > 0 else None
        self.register_buffer('protos', torch.zeros(C, proj_dim))
        self.register_buffer('proto_ready', torch.zeros(C, dtype=torch.bool))
        logw = torch.zeros(C, C)
        for a, b in CONFUSABLE:
            logw[a, b] = logw[b, a] = math.log(conf_weight)
        self.register_buffer('logw', logw)

    def contrast(self, R, y):
        z = F.normalize(self.proj(R), dim=1)
        n, dev = z.shape[0], z.device
        protos = self.protos.clone()  # a copy: the buffer is updated in place below, before backward runs
        logits = torch.cat([z @ z.t(), z @ protos.t()], dim=1) / self.t  # (n, n + C)
        cand = torch.cat([y, torch.arange(C, device=dev)])
        valid = torch.cat([torch.ones(n, dtype=torch.bool, device=dev), self.proto_ready])[None, :].repeat(n, 1)
        valid[:, :n] &= ~torch.eye(n, dtype=torch.bool, device=dev)
        pos = (cand[None, :] == y[:, None]) & valid
        log_den = torch.logsumexp((logits + self.logw[y][:, cand]).masked_fill(~valid, float('-inf')), 1, keepdim=True)
        npos = pos.sum(1)
        keep = npos > 0
        loss = (-((logits - log_den) * pos).sum(1)[keep] / npos[keep]).mean() if keep.any() else z.sum() * 0.0
        with torch.no_grad():  # EMA prototypes, updated after the loss
            for k in y.unique().tolist():
                m = F.normalize(z[y == k].mean(0), dim=0)
                if bool(self.proto_ready[k]):
                    self.protos[k] = F.normalize(self.momentum * self.protos[k] + (1 - self.momentum) * m, dim=0)
                else:
                    self.protos[k] = m
                    self.proto_ready[k] = True
        return loss

    def forward(self, R, y, s):
        loss = R.new_zeros(())
        if self.proj is not None:
            loss = loss + self.lam_c * self.contrast(R, y)
        if self.sent is not None:
            loss = loss + self.lam_s * F.cross_entropy(self.sent(R), s)
        return loss


# ----------------------------------------------------------------------------- stages
def stage_setup(CONFIG, code_dir):
    out = os.path.join(CONFIG['work_dir'], 'outputs_pathb')
    for sub in ('', 'tables', 'figures'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    os.makedirs(os.path.join(CONFIG['work_dir'], 'checkpoints_pathb'), exist_ok=True)
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
                 dict(candidates=CANDIDATES, confusable_pairs=[(CLASSES[a], CLASSES[b]) for a, b in CONFUSABLE],
                      epochs=CONFIG['epochs'], stage1_seeds=CONFIG['stage1_seeds'],
                      stage2_seeds=CONFIG['stage2_seeds'], gate_min_gain=CONFIG['gate_min_gain'], criteria=CRITERIA))
    if not cuda:
        print('WARNING: no GPU. Training will be very slow; switch the accelerator to GPU and run again.')
    P.log_event(S, 'setup', device=S['device'], gpu=S['env']['gpu'], torch=torch.__version__)
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


def stage_prepare(S):
    """Data gate (from the first experiment), the two input-feature variants and the train/dev datasets."""
    P.stage_load_data(S)  # reads train/dev labels only
    D = S['D']
    s_tr = np.concatenate([np.asarray(D['sentiment_labels'][v], dtype=np.int64) for v in S['keys']['train']])
    P.record_check(S, 'training sentiment labels are small non-negative integers', s_tr.min() >= 0 and s_tr.max() < 10,
                   hard=True, detail=np.bincount(s_tr).tolist())
    S['n_sent'] = int(s_tr.max()) + 1
    mean4 = dict(D)
    for prefix in ('eroberta', 'sroberta'):  # features only; labels untouched
        mean4[prefix + '1'] = {v: np.mean([np.asarray(D[f'{prefix}{k}'][v], dtype=np.float32) for k in range(1, 5)], 0)
                               for v in D[prefix + '1']}
    S['D_variants'] = {'layer1': D, 'mean4': mean4}
    S['datasets'] = {name: {split: P.make_dataset(dict(S, D=Dv), split) for split in ('train', 'valid')}
                     for name, Dv in S['D_variants'].items()}
    P._save_checks(S)
    P.log_event(S, 'datasets ready', sentiment_classes=S['n_sent'])


def train_run(S, variant, seed, epochs):
    """One training run of the unchanged ECERC model with the original recipe plus the variant's changes.
    Keeps the epoch with the best dev macro-F1 (ties: earliest). Never touches the test split."""
    from torch.utils.data import DataLoader
    dev = S['device']
    seed_everything(seed)
    model = P.build_model(S).to(dev)
    heads = None
    if variant['lam_c'] > 0 or variant['lam_s'] > 0:
        with torch.random.fork_rng(devices=[]):  # own initialisation, global random stream untouched
            torch.manual_seed(seed + 100_000)
            heads = AuxHeads(variant['lam_c'], variant['lam_s'], S['n_sent'])
        heads.to(dev)
    params = list(model.parameters()) + (list(heads.parameters()) if heads is not None else [])
    opt = torch.optim.Adam(params, lr=variant['lr'], weight_decay=2e-4)  # MELD/train.py: Adam, l2 = 0.0002
    loss_f = S['m_loss'].Loss(alpha=None)  # original MELD objective (focal-style, gamma = 1, no class weights)
    ds = S['datasets'][variant['features']]
    loader = DataLoader(ds['train'], batch_size=32, shuffle=True, collate_fn=ds['train'].collate_fn, num_workers=0)
    sent = S['D']['sentiment_labels']
    store = {}
    hook = model.smax_fc.register_forward_hook(lambda mod, inp, out: store.__setitem__('x', inp[0]))
    curve, best, t0 = [], None, time.time()
    try:
        for epoch in range(epochs):
            model.train()
            if heads is not None:
                heads.train()
            for data in loader:
                opt.zero_grad()
                emo, sem, audio, vision, qmask, umask, label = [d.to(dev) for d in data[:-1]]
                seq = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
                emo = torch.cat([emo, audio, vision], dim=-1)  # as in MELD/train.py
                sem = torch.cat([sem], dim=-1)
                log_prob = model(emo, sem, qmask, umask, seq)
                y = torch.cat([label[j][:seq[j]] for j in range(len(label))])
                loss = loss_f(log_prob, y)
                if heads is not None:
                    rep = torch.cat([store['x'][j, :seq[j], :] for j in range(len(seq))])
                    s = torch.as_tensor(np.concatenate([np.asarray(sent[v], dtype=np.int64)[:n]
                                                        for v, n in zip(data[-1], seq)]), device=dev)
                    loss = loss + heads(rep, y, s)
                loss.backward()
                opt.step()
            d = P.run_inference(S, model, ds['valid'])
            m = P.scores(_cm(d['y'], d['logp'].argmax(1)))
            curve.append(dict(epoch=epoch, macro_f1=100 * float(m['mf1']), weighted_f1=100 * float(m['wf1']),
                              acc=100 * float(m['acc'])))
            if best is None or 100 * float(m['mf1']) > best['dev_macro_f1'] + 1e-9:
                best = dict(epoch=epoch, dev_macro_f1=100 * float(m['mf1']), dev_weighted_f1=100 * float(m['wf1']),
                            dev_acc=100 * float(m['acc']), dev_f1=100 * m['f1'], dev_logp=d['logp'],
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    finally:
        hook.remove()
    best.update(variant=variant['name'], seed=seed, seconds=round(time.time() - t0, 1), curve=curve)
    return best


def _run_line(stage, r):
    return (f'  [{stage}] {r["variant"]:<20s} seed {r["seed"]}: best dev macro-F1 {r["dev_macro_f1"]:.2f} '
            f'(epoch {r["epoch"]}), weighted F1 {r["dev_weighted_f1"]:.2f}, Fear F1 {r["dev_f1"][FEA]:.1f}, '
            f'Disgust F1 {r["dev_f1"][DIS]:.1f}, {r["seconds"]:.0f}s')


def stage_screen(S):
    cfg = S['cfg']
    results, means = {}, {}

    def run_variant(v):
        res = []
        for sd in cfg['stage1_seeds']:
            r = train_run(S, v, sd, cfg['epochs'])
            print(_run_line('stage 1', r), flush=True)
            if v is BASE and sd == cfg['stage1_seeds'][0]:
                check_eval_path(S, r)
            r.pop('state')
            r.pop('dev_logp')
            res.append(r)
        results[v['name']] = res
        means[v['name']] = float(np.mean([r['dev_macro_f1'] for r in res]))
        P.log_event(S, 'stage 1 candidate done', candidate=v['name'], mean_dev_macro_f1=round(means[v['name']], 2))

    t0 = time.time()
    for v in CANDIDATES:
        run_variant(v)
    base = means[BASE['name']]
    fam_best = {}
    for v in CANDIDATES[1:]:
        if v['family'] not in fam_best or means[v['name']] > means[fam_best[v['family']]['name']] + 1e-12:
            fam_best[v['family']] = v
    helpful = [v for _, v in sorted(fam_best.items()) if means[v['name']] > base]
    combo = None
    if len(helpful) >= 2:
        combo = dict(BASE, name='C5_combined', family='C5', components=[v['name'] for v in helpful])
        for v in helpful:
            for key in ('features', 'lr', 'lam_c', 'lam_s'):
                if v[key] != BASE[key]:
                    combo[key] = v[key]
        run_variant(combo)
    pool = CANDIDATES[1:] + ([combo] if combo else [])
    winner = pool[0]
    for v in pool[1:]:
        if means[v['name']] > means[winner['name']] + 1e-12:
            winner = v
    gain = means[winner['name']] - base
    proceed = gain >= cfg['gate_min_gain']
    S['stage1'] = dict(results=results, means=means, winner=winner, gain=gain, proceed=proceed, combo=combo)
    rows = []
    for name, rs in results.items():
        for r in rs:
            row = dict(candidate=name, seed=r['seed'], best_epoch=r['epoch'], dev_macro_f1=r['dev_macro_f1'],
                       dev_weighted_f1=r['dev_weighted_f1'], dev_acc=r['dev_acc'], seconds=r['seconds'])
            row.update({f'F1_{CLASSES[c]}': r['dev_f1'][c] for c in range(C)})
            rows.append(row)
    P.write_csv(os.path.join(S['out'], 'tables', 'stage1_runs.csv'), rows)
    P.write_json(os.path.join(S['out'], 'stage1_summary.json'),
                 dict(mean_dev_macro_f1=means, family_best={f: v['name'] for f, v in fam_best.items()},
                      combination=combo, winner=winner, gain_over_control=gain, gate=cfg['gate_min_gain'],
                      proceed_to_stage2=proceed, curves={n: [r['curve'] for r in rs] for n, rs in results.items()}))
    print('\nSTAGE 1 (dev only): mean best-epoch dev macro-F1 over seeds')
    for name in results:
        print(f'  {name:<20s} {means[name]:6.2f}  ({means[name] - base:+.2f} vs control)')
    print(f'  winner: {winner["name"]} ({gain:+.2f}); gate {cfg["gate_min_gain"]:+.2f} -> '
          + ('stage 2 will run' if proceed else 'stage 2 skipped (the winner did not beat the control by the required margin)'))
    P.log_event(S, 'stage 1 done', minutes=round((time.time() - t0) / 60, 1), winner=winner['name'],
                gain=round(gain, 2), proceed=proceed)


def check_eval_path(S, r):
    """Once: the evaluation loop must reproduce the original train_or_eval_model on dev for a trained model."""
    from sklearn.metrics import confusion_matrix
    model = P.build_model(S)
    model.load_state_dict(r['state'], strict=True)
    model.to(S['device']).eval()
    ds = S['datasets']['layer1']['valid']
    ours = P.run_inference(S, model, ds)
    _, _, cm_o = P.original_function_eval(S, model, ds)
    cm = confusion_matrix(ours['y'], ours['logp'].argmax(1))
    diff = int(np.abs(cm_o - cm).sum()) if cm_o.shape == cm.shape else -1
    P.record_check(S, 'evaluation loop reproduces the original train_or_eval_model on dev', 0 <= diff <= 4,
                   hard=True, detail=f'sum |confusion diff| = {diff}')
    P._save_checks(S)


def stage_freeze(S):
    st, cfg = S['stage1'], S['cfg']
    frozen = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),
                  note='Written before any test-set access. Stage 2 and the test evaluation read their rules from here.',
                  code_sha256=S['env']['code_sha256'], data_sha256=S['data_sha256'], control=BASE,
                  winner=st['winner'], stage1_mean_dev_macro_f1=st['means'],
                  gate=dict(gain=st['gain'], threshold=cfg['gate_min_gain'], proceed=st['proceed']),
                  stage2_seeds=cfg['stage2_seeds'], epochs=cfg['epochs'],
                  checkpoint_rule='per run, the epoch with the best dev macro-F1 (earliest on ties)',
                  bootstrap=dict(B=cfg['bootstrap_B'], seed=cfg['boot_seed'],
                                 design='hierarchical: seeds and test dialogues resampled; seed-level t-interval also reported'),
                  criteria=CRITERIA, device=S['device'], env=S['env'])
    path = os.path.join(S['out'], 'frozen_config_pathb.json')
    P.write_json(path, frozen)
    S['frozen_path'], S['frozen_sha256'] = path, P.sha256_file(path)
    P.log_event(S, 'FREEZE', sha256=S['frozen_sha256'][:16], winner=st['winner']['name'], proceed=st['proceed'])


def _load_frozen(S):
    if not S.get('frozen_path') or P.sha256_file(S['frozen_path']) != S['frozen_sha256']:
        raise P.CheckFailed('frozen_config_pathb.json is missing or was modified: run the freeze cell first')
    with open(S['frozen_path']) as f:
        F_ = json.load(f)
    if any(P.sha256_file(os.path.join(S['code_dir'], n)) != h for n, h in F_['code_sha256'].items()):
        raise P.CheckFailed('code changed after the freeze')
    return F_


def stage_confirm(S):
    F_ = _load_frozen(S)
    S['stage2'] = None
    if not F_['gate']['proceed']:
        print('Stage 2 skipped by the pre-registered gate: the dev winner did not beat the original recipe by the required margin.')
        return
    arms = [('control', F_['control']), ('winner', F_['winner'])]
    runs, t0 = {a: [] for a, _ in arms}, time.time()
    for sd in F_['stage2_seeds']:
        for arm, v in arms:
            r = train_run(S, v, sd, F_['epochs'])
            print(_run_line('stage 2', r), flush=True)
            path = os.path.join(S['cfg']['work_dir'], 'checkpoints_pathb', f'{arm}_seed{sd}.pt')
            torch.save(r.pop('state'), path)
            r['checkpoint'] = path
            runs[arm].append(r)
    S['stage2'] = runs
    rows = []
    for arm, rs in runs.items():
        for r in rs:
            rows.append(dict(arm=arm, candidate=r['variant'], seed=r['seed'], best_epoch=r['epoch'],
                             dev_macro_f1=r['dev_macro_f1'], dev_weighted_f1=r['dev_weighted_f1'], dev_acc=r['dev_acc']))
    P.write_csv(os.path.join(S['out'], 'tables', 'stage2_dev_runs.csv'), rows)
    P.log_event(S, 'stage 2 training done', minutes=round((time.time() - t0) / 60, 1))


def hierarchical_bootstrap(cms, B, seed, chunk=1000):
    """cms: (n_dialogues, n_seeds, 2 arms, 49). Resample test dialogues (shared by all seeds) and seeds."""
    n, ns = cms.shape[:2]
    rng = np.random.default_rng(seed)
    flat = cms.reshape(n, -1).astype(float)
    keys = ('mf1', 'acc', 'wf1', 'f1', 'prec', 'rec')
    out, done = {k: [] for k in keys}, 0
    while done < B:
        b = min(chunk, B - done)
        sc = P.scores((P._boot_weights(rng, n, b) @ flat).reshape(b, ns, 2, C, C))
        idx = rng.integers(0, ns, size=(b, ns))
        rows = np.arange(b)[:, None]
        for k in keys:
            d = sc[k][:, :, 1] - sc[k][:, :, 0]
            out[k].append(100 * d[rows, idx].mean(1))
        done += b
    return {k: np.concatenate(v) for k, v in out.items()}


def decide(est, lo, hi, d_f1, d_acc, d_wf1, frac_positive):
    c = dict(delta_ge_1=est >= 1.0, ci_lower_gt_0=lo > 0, fear_improves=d_f1[FEA] > 0,
             disgust_improves=d_f1[DIS] > 0, consistent_across_seeds=frac_positive >= 0.7,
             accuracy_guardrail=d_acc >= -1.0, weighted_f1_guardrail=d_wf1 >= -0.5,
             class_guardrail=bool(np.all(np.asarray(d_f1) >= -2.0)))
    efficacy = c['delta_ge_1'] and c['ci_lower_gt_0']
    minority = c['fear_improves'] and c['disgust_improves'] and c['consistent_across_seeds']
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
    F_ = _load_frozen(S)
    if S.get('stage2') is None:
        S['results'] = None
        _write_summary(S, F_)
        return
    P.log_event(S, 'TEST evaluation starts (test dialogues are run and test labels read only from here on)')
    seeds, arms = F_['stage2_seeds'], [('control', F_['control']), ('winner', F_['winner'])]
    test_ds = {name: P.make_dataset(dict(S, D=S['D_variants'][name]), 'test')
               for name in {v['features'] for _, v in arms}}
    n_te = len(S['keys']['test'])
    cms = np.zeros((n_te, len(seeds), 2, C * C), dtype=np.int64)
    for a, (arm, v) in enumerate(arms):
        for i, r in enumerate(S['stage2'][arm]):
            model = P.build_model(S)
            model.load_state_dict(torch.load(r['checkpoint'], map_location=S['device'], weights_only=True), strict=True)
            model.to(S['device']).eval()
            d = P.run_inference(S, model, test_ds[v['features']])
            cms[:, i, a, :] = P.per_dialogue_cms(d['y'], d['logp'].argmax(1)[None, :], d['dlg'], n_te)[:, 0, :]
            np.savez_compressed(os.path.join(S['out'], f'test_{arm}_seed{seeds[i]}.npz'), logp=d['logp'].astype(np.float32),
                                y=d['y'], dlg=d['dlg'])
    sc = P.scores(cms.sum(0).reshape(len(seeds), 2, C, C))
    per_seed = 100 * (sc['mf1'][:, 1] - sc['mf1'][:, 0])
    dl = {k: 100 * (sc[k][:, 1] - sc[k][:, 0]).mean(0) for k in ('mf1', 'acc', 'wf1', 'f1', 'prec', 'rec')}
    bt = hierarchical_bootstrap(cms, F_['bootstrap']['B'], F_['bootstrap']['seed'])
    ci = lambda x: np.percentile(x, [2.5, 97.5], axis=0)  # noqa: E731
    lo, hi = ci(bt['mf1'])
    try:
        from scipy.stats import t as student_t
        tq = float(student_t.ppf(0.975, len(seeds) - 1))
    except Exception:
        tq = 2.262
    sd_ = float(per_seed.std(ddof=1)) if len(seeds) > 1 else float('nan')
    t_ci = [float(per_seed.mean() - tq * sd_ / math.sqrt(len(seeds))), float(per_seed.mean() + tq * sd_ / math.sqrt(len(seeds)))]
    verdict, conditions = decide(float(dl['mf1']), lo, hi, dl['f1'], float(dl['acc']), float(dl['wf1']),
                                 float((per_seed > 0).mean()))
    arm_means = {arm: dict(acc=100 * sc['acc'][:, a].mean(), macro_f1=100 * sc['mf1'][:, a].mean(),
                           weighted_f1=100 * sc['wf1'][:, a].mean(), f1=100 * sc['f1'][:, a].mean(0),
                           precision=100 * sc['prec'][:, a].mean(0), recall=100 * sc['rec'][:, a].mean(0),
                           macro_f1_sd=100 * sc['mf1'][:, a].std(ddof=1) if len(seeds) > 1 else None)
                 for a, (arm, _) in enumerate(arms)}
    S['results'] = dict(
        verdict=verdict, conditions=conditions, winner=F_['winner'], seeds=seeds,
        delta_macro_f1=float(dl['mf1']), ci95_hierarchical=[lo, hi], ci95_seed_t=t_ci, per_seed_delta=per_seed,
        seeds_positive=int((per_seed > 0).sum()),
        per_class=dict(delta_f1=dl['f1'], delta_f1_ci95=ci(bt['f1']).T, delta_precision=dl['prec'],
                       delta_recall=dl['rec']),
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
    P.write_json(os.path.join(S['out'], 'test_results_pathb.json'), S['results'])
    _write_summary(S, F_)
    P.log_event(S, 'TEST done', verdict=verdict, delta=round(float(dl['mf1']), 2), ci=[round(lo, 2), round(hi, 2)])


def _write_summary(S, F_):
    st, env = S['stage1'], S['env']
    L = ['# ECERC Limitation 2, Path B: training-time representation changes', '',
         f'Run {time.strftime("%Y-%m-%d %H:%M")} | device {env["device"]} ({env["gpu"]}) | torch {env["torch"]} | '
         f'data sha256 {S["data_sha256"][:16]}...', '',
         '## Stage 1: dev screen (mean best-epoch dev macro-F1 over seeds; selection only, not a result)', '',
         '| candidate | dev macro-F1 | vs control |', '|---|---|---|']
    base = st['means'][BASE['name']]
    for name, m in st['means'].items():
        L.append(f'| {name} | {m:.2f} | {m - base:+.2f} |')
    L += ['', f'Winner: **{st["winner"]["name"]}** ({st["gain"]:+.2f} on dev); gate {S["cfg"]["gate_min_gain"]:+.2f} -> '
              + ('stage 2 ran' if st['proceed'] else 'stage 2 skipped'), '']
    res = S.get('results')
    if res:
        L += [f'## Verdict: {res["verdict"]}', '', '| condition | met |', '|---|---|']
        L += [f'| {k} | {"yes" if v else "no"} |' for k, v in res['conditions'].items()]
        a, w = res['arms']['control'], res['arms']['winner']
        L += ['', f'## Test, {len(res["seeds"])} seeds per arm (mean)', '',
              '| arm | acc | weighted F1 | macro-F1 (SD) | ' + ' | '.join(f'F1 {c}' for c in CLASSES) + ' |',
              '|---' * (4 + C) + '|']
        for name, m in (('control (C0)', a), (f'winner ({res["winner"]["name"]})', w)):
            L.append(f'| {name} | {m["acc"]:.2f} | {m["weighted_f1"]:.2f} | {m["macro_f1"]:.2f} ({m["macro_f1_sd"]:.2f}) | '
                     + ' | '.join(f'{x:.2f}' for x in m['f1']) + ' |')
        g = res['guardrails']
        L += ['', f'- Delta macro-F1: **{P._fmt(res["delta_macro_f1"], *res["ci95_hierarchical"])}** (hierarchical '
                  f'bootstrap); seed-level t-interval [{res["ci95_seed_t"][0]:+.2f}, {res["ci95_seed_t"][1]:+.2f}]',
              f'- positive in {res["seeds_positive"]}/{len(res["seeds"])} seeds: '
              + ', '.join(f'{x:+.2f}' for x in res['per_seed_delta']),
              f'- Delta accuracy {P._fmt(g["delta_acc"], *g["delta_acc_ci95"])}; '
              f'Delta weighted F1 {P._fmt(g["delta_weighted_f1"], *g["delta_weighted_f1_ci95"])}',
              '', '| class | Delta F1 [95% CI] | Delta precision | Delta recall | support |', '|---|---|---|---|---|']
        pc = res['per_class']
        for c in range(C):
            L.append(f'| {CLASSES[c]} | {P._fmt(pc["delta_f1"][c], *pc["delta_f1_ci95"][c])} | '
                     f'{pc["delta_precision"][c]:+.2f} | {pc["delta_recall"][c]:+.2f} | {int(res["support"][c])} |')
        L += ['', f'- control weighted F1 {res["fidelity"]["control_weighted_f1"]:.2f} vs paper {P.PAPER_TEST_WF1}']
    warn = [c for c in S['checks'] if not c['passed']]
    L += ['', f'Checks: {sum(c["passed"] for c in S["checks"])} passed, {len(warn)} warnings'
          + ('' if not warn else ' -- ' + '; '.join(f'{c["name"]} ({c["detail"]})' for c in warn))]
    with open(os.path.join(S['out'], 'SUMMARY_pathb.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')


def stage_package(S):
    zpath = os.path.join(S['cfg']['work_dir'], 'ecerc_l2_pathb_results.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(S['out']):
            for name in files:
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, os.path.dirname(S['out'])))
    with open(os.path.join(S['out'], 'SUMMARY_pathb.md')) as f:
        print(f.read())
    print(f'Results archive: {zpath} ({os.path.getsize(zpath) / 2 ** 20:.1f} MB)')
    return zpath
