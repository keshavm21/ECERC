"""
ECERC Limitation 2, Experiment 4: a context-aware text expert fused with ECERC.

New code for research/limitation2. Runs 1-3 showed that ECERC's rare-emotion ceiling is set by its precomputed
input features, not by its decision rule, objective or training. This experiment adds evidence ECERC never sees:
a RoBERTa-large classifier fine-tuned end to end on the utterance text with the preceding dialogue as context
(TelME-style prompt "... Now Speaker k feels <mask>"; past context only, the same task setting as ECERC).
Its log-probabilities are combined with ECERC's by one weight w chosen on dev:

    score = (1 - w) * log p_ECERC + w * log p_expert,   w in {0.00, 0.05, ..., 1.00}  (w = 0 is ECERC alone)

Pairs: each released ECERC checkpoint (C1-C3) x each expert seed = 9 pairs. Primary estimand: the mean over
pairs of test macro-F1(fused) - macro-F1(ECERC alone). The test runs only if the cross-fitted dev gain is at
least CONFIG['gate_min_gain'], and only after every decision is frozen in frozen_config_expert.json.
"""
import json
import math
import os
import platform
import random
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import l2_posthoc as P  # downloads, data gate, ECERC builders, inference and metrics (unchanged)

C, CLASSES = P.C, P.CLASSES
NEU, SUR, FEA, SAD, JOY, DIS, ANG = range(C)
W_GRID = [round(0.05 * i, 2) for i in range(21)]
TAU_GRID = [round(0.1 * i, 1) for i in range(11)]  # secondary analysis only
CRITERIA = dict(
    primary='mean over (ECERC checkpoint x expert seed) pairs of test macro-F1(fused) - macro-F1(ECERC alone)',
    success=['delta >= +1.0', 'procedure-level 95% CI lower bound > 0', 'mean F1 change > 0 for Fear and for Disgust',
             'delta > 0 in at least 70% of pairs', 'accuracy change >= -1.0', 'weighted-F1 change >= -0.5',
             'every class: mean F1 change >= -2.0'],
    verdict_order=['SUCCESS', 'TRADE-OFF (efficacy met, a guardrail fails)',
                   'PARTIAL (macro-F1 target met, Fear/Disgust/consistency not)',
                   'FAILURE (CI upper bound < +1.0)', 'INCONCLUSIVE'],
    guardrails_judged_on='point estimates; 95% CIs reported')
_FIX = {'\x92': "'", '\x91': "'", '\x93': '"', '\x94': '"', '\x96': '-', '\x97': '-', '\x85': '...', '\xa0': ' '}


def _cm(y, p):
    return np.bincount(y * C + p, minlength=C * C).reshape(C, C)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _clean(s):
    s = str(s)
    for a, b in _FIX.items():
        s = s.replace(a, b)
    return ' '.join(s.split())


# ----------------------------------------------------------------------------- the expert
class Expert(nn.Module):
    """RoBERTa encoder; the <mask> token's final hidden state feeds a linear emotion head."""

    def __init__(self, name, n_classes=C, dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(name, add_pooling_layer=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.enc.config.hidden_size, n_classes)

    def forward(self, ids, att, pos):
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        return self.head(self.drop(h[torch.arange(ids.shape[0], device=ids.device), pos]))


def build_texts(S, split):
    """One input per utterance: the dialogue so far ("Speaker k: text" turns) plus the prompt for the target."""
    D, tok = S['D'], S['tok']
    texts = []
    for vid in S['keys'][split]:
        sents = D['sentences'][vid]
        spk = np.asarray(D['speakers'][vid]).argmax(-1)
        turns = [f'Speaker {int(spk[i]) + 1}: {_clean(sents[i])}' for i in range(len(sents))]
        for t in range(len(sents)):
            texts.append(f' {tok.sep_token} '.join(turns[:t + 1])
                         + f' {tok.sep_token} Now Speaker {int(spk[t]) + 1} feels {tok.mask_token}')
    return texts


def encode(S, split, with_labels):
    tok, cfg = S['tok'], S['cfg']
    texts = build_texts(S, split)
    enc = tok(texts, truncation=True, max_length=cfg['expert_max_len'])  # truncation_side = left: keeps the prompt
    ids = enc['input_ids']
    pos = [row.index(tok.mask_token_id) for row in ids]
    out = dict(ids=ids, pos=np.asarray(pos), n=len(ids), example=texts[min(5, len(texts) - 1)])
    if with_labels:
        out['y'] = np.concatenate([np.asarray(S['D']['emotion_labels'][v], dtype=np.int64) for v in S['keys'][split]])
    return out


def _batch(data, idx, dev, pad_id):
    L = max(len(data['ids'][i]) for i in idx)
    ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx):
        seq = data['ids'][i]
        ids[r, :len(seq)] = torch.tensor(seq)
        att[r, :len(seq)] = 1
    pos = torch.as_tensor(data['pos'][idx], dtype=torch.long)
    return ids.to(dev), att.to(dev), pos.to(dev)


def predict_expert(S, model, data, bs=64):
    dev, cuda = S['device'], S['device'] == 'cuda'
    model.eval()
    order = np.argsort([len(x) for x in data['ids']])
    out = np.zeros((data['n'], C))
    with torch.no_grad():
        for b in range(0, data['n'], bs):
            idx = order[b:b + bs]
            ids, att, pos = _batch(data, idx, dev, S['tok'].pad_token_id)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cuda):
                logits = model(ids, att, pos)
            out[idx] = F.log_softmax(logits.float(), dim=-1).cpu().numpy()
    return out


def train_expert(S, seed):
    """Fine-tune one expert; keep the epoch with the best dev macro-F1 (earliest on ties). Train/dev only."""
    from transformers import get_linear_schedule_with_warmup
    cfg, dev = S['cfg'], S['device']
    cuda = dev == 'cuda'
    seed_everything(seed)
    model = Expert(cfg['expert_model']).to(dev)
    tr, dv = S['enc']['train'], S['enc']['valid']
    bs = cfg['expert_batch']
    steps = math.ceil(tr['n'] / bs)
    total = steps * cfg['expert_epochs']
    no_decay = ('bias', 'LayerNorm.weight', 'LayerNorm.bias')
    groups = [dict(params=[p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], weight_decay=0.01),
              dict(params=[p for n, p in model.named_parameters() if any(k in n for k in no_decay)], weight_decay=0.0)]
    opt = torch.optim.AdamW(groups, lr=cfg['expert_lr'])
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler('cuda', enabled=cuda)
    gen = torch.Generator().manual_seed(seed)
    best, curve, t0 = None, [], time.time()
    for epoch in range(cfg['expert_epochs']):
        model.train()
        perm = torch.randperm(tr['n'], generator=gen).numpy()
        for b in range(steps):
            idx = perm[b * bs:(b + 1) * bs]
            ids, att, pos = _batch(tr, idx, dev, S['tok'].pad_token_id)
            y = torch.as_tensor(tr['y'][idx], device=dev)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cuda):
                logits = model(ids, att, pos)
            loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        lp = predict_expert(S, model, dv)
        m = P.scores(_cm(dv['y'], lp.argmax(1)))
        curve.append(dict(epoch=epoch, macro_f1=100 * float(m['mf1']), weighted_f1=100 * float(m['wf1']),
                          acc=100 * float(m['acc']), fear_f1=100 * float(m['f1'][FEA]), disgust_f1=100 * float(m['f1'][DIS])))
        print(f'  [expert seed {seed}] epoch {epoch}: dev macro-F1 {curve[-1]["macro_f1"]:.2f}, weighted F1 '
              f'{curve[-1]["weighted_f1"]:.2f}, Fear {curve[-1]["fear_f1"]:.1f}, Disgust {curve[-1]["disgust_f1"]:.1f} '
              f'({time.time() - t0:.0f}s)', flush=True)
        if best is None or curve[-1]['macro_f1'] > best['dev_macro_f1'] + 1e-9:
            best = dict(epoch=epoch, dev_macro_f1=curve[-1]['macro_f1'], dev_logp=lp,
                        state={k: v.detach().to('cpu', torch.float16).clone() for k, v in model.state_dict().items()})
    path = os.path.join(cfg['work_dir'], 'checkpoints_expert', f'expert_seed{seed}.pt')
    torch.save(best.pop('state'), path)
    del model, opt
    if cuda:
        torch.cuda.empty_cache()
    best.update(seed=seed, path=path, curve=curve, seconds=round(time.time() - t0, 1))
    return best


def load_expert(S, path):
    model = Expert(S['cfg']['expert_model'])
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
    return model.to(S['device'])


# ----------------------------------------------------------------------------- fusion helpers
def fused_preds(lpE, lpT, taus=None, log_prior=None):
    """Predictions for every weight in W_GRID (and, if given, every tau): (R, N)."""
    rules = []
    for w in W_GRID:
        s = (1 - w) * lpE + w * lpT
        if taus is None:
            rules.append(s.argmax(1))
        else:
            for t in taus:
                rules.append((s - t * log_prior[None, :]).argmax(1))
    return np.stack(rules)


def pair_cms(E, T, n_dlg, taus=None, log_prior=None):
    """(n_dlg, n_pairs, R, 49) confusion counts for every (ECERC checkpoint, expert seed) pair and rule."""
    out = []
    for e in E:
        for t in T:
            assert np.array_equal(e['y'], t['y']), 'ECERC and expert items are not aligned'
            out.append(P.per_dialogue_cms(e['y'], fused_preds(e['logp'], t['logp'], taus, log_prior), e['dlg'], n_dlg))
    return np.stack(out, axis=1)


def _select(tot):
    """Rule with the highest mean macro-F1 over pairs; ties go to the smaller index (closer to ECERC alone)."""
    mean = P.scores(tot.reshape(tot.shape[0], tot.shape[1], C, C))['mf1'].mean(0)
    return int(P.first_argmax(mean)), mean


def crossfit_forecast(dev_cms, n_folds, seed, B, n_test_dlg):
    n_dev, K = dev_cms.shape[:2]
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n_dev), n_folds)
    cf, chosen = np.zeros((n_dev, K, C * C)), []
    for f in folds:
        keep = np.ones(n_dev, bool)
        keep[f] = False
        r, _ = _select(dev_cms[keep].sum(0))
        chosen.append(W_GRID[r])
        cf[f] = dev_cms[f][:, :, r, :]
    base = dev_cms[:, :, 0, :].astype(float)

    def delta(w):
        a = P.scores((w @ cf.reshape(n_dev, -1)).reshape(-1, K, C, C))['mf1']
        b = P.scores((w @ base.reshape(n_dev, -1)).reshape(-1, K, C, C))['mf1']
        return 100 * (a - b).mean(1)

    point = float(delta(np.ones((1, n_dev)))[0])
    boots = delta(P._boot_weights(rng, n_dev, B))
    se = float(boots.std(ddof=1)) * math.sqrt(n_dev / n_test_dlg)
    return dict(fold_w=chosen, crossfit_delta_macro_f1=point, dev_bootstrap_ci95=np.percentile(boots, [2.5, 97.5]),
                se_test_projected=se, mde80_projected=2.8 * se)


def procedure_bootstrap(dev_cms, test_cms, B, seed, chunk=250):
    """Resample dev dialogues and re-select w with the frozen rule, resample test dialogues, recompute every
    difference against w = 0 (ECERC alone). Returns per-replicate arrays."""
    n_dev, K, R = dev_cms.shape[:3]
    n_te = test_cms.shape[0]
    rng = np.random.default_rng(seed)
    dflat, tflat = dev_cms.reshape(n_dev, -1).astype(float), test_cms.reshape(n_te, -1).astype(float)
    keys = ('mf1', 'acc', 'wf1', 'f1', 'prec', 'rec')
    out, done = {k: [] for k in keys + ('per_pair', 'w_sel')}, 0
    while done < B:
        b = min(chunk, B - done)
        dmf1 = P.scores((P._boot_weights(rng, n_dev, b) @ dflat).reshape(b, K, R, C, C))['mf1'].mean(1)
        sel = P.first_argmax(dmf1, axis=1)
        tsc = P.scores((P._boot_weights(rng, n_te, b) @ tflat).reshape(b, K, R, C, C))
        rows, cols = np.arange(b)[:, None], np.arange(K)[None, :]
        for k in keys:
            d = tsc[k][rows, cols, sel[:, None]] - tsc[k][:, :, 0]
            out[k].append(100 * d.mean(1))
            if k == 'mf1':
                out['per_pair'].append(100 * d)
        out['w_sel'].append(np.asarray(W_GRID)[sel])
        done += b
    return {k: np.concatenate(v) for k, v in out.items()}


def decide(est, lo, hi, d_f1, d_acc, d_wf1, frac_positive):
    c = dict(delta_ge_1=est >= 1.0, ci_lower_gt_0=lo > 0, fear_improves=d_f1[FEA] > 0,
             disgust_improves=d_f1[DIS] > 0, consistent_across_pairs=frac_positive >= 0.7,
             accuracy_guardrail=d_acc >= -1.0, weighted_f1_guardrail=d_wf1 >= -0.5,
             class_guardrail=bool(np.all(np.asarray(d_f1) >= -2.0)))
    efficacy = c['delta_ge_1'] and c['ci_lower_gt_0']
    minority = c['fear_improves'] and c['disgust_improves'] and c['consistent_across_pairs']
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


# ----------------------------------------------------------------------------- stages
def stage_setup(CONFIG, code_dir):
    out = os.path.join(CONFIG['work_dir'], 'outputs_expert')
    for sub in ('', 'tables', 'figures', 'dumps'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    os.makedirs(os.path.join(CONFIG['work_dir'], 'checkpoints_expert'), exist_ok=True)
    import pandas
    import sklearn
    import transformers
    S = dict(cfg=CONFIG, out=out, code_dir=code_dir, checks=[], torch=torch)
    cuda = torch.cuda.is_available()
    if not cuda:
        P.install_cpu_cuda_shim(torch)
    S['device'] = 'cuda' if cuda else 'cpu'
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    for name in ('model', 'dataloader', 'loss', 'inference'):
        S['m_' + name] = __import__(name)
    S['env'] = dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                    transformers=transformers.__version__, numpy=np.__version__, pandas=pandas.__version__,
                    sklearn=sklearn.__version__, device=S['device'], gpu=torch.cuda.get_device_name(0) if cuda else None,
                    code_sha256={f: P.sha256_file(os.path.join(code_dir, f))
                                 for f in sorted(os.listdir(code_dir)) if f.endswith('.py')})
    P.write_json(os.path.join(out, 'environment.json'), S['env'])
    P.write_json(os.path.join(out, 'design.json'),  # written before any data is loaded
                 dict(expert_model=CONFIG['expert_model'], expert_seeds=CONFIG['expert_seeds'],
                      expert_epochs=CONFIG['expert_epochs'], expert_lr=CONFIG['expert_lr'],
                      expert_batch=CONFIG['expert_batch'], expert_max_len=CONFIG['expert_max_len'],
                      expert_retry_rule=f'one retry (seed + 1000) if best dev macro-F1 < {CONFIG.get("expert_min_dev_mf1", 0.0)}',
                      fusion='score = (1 - w) log p_ECERC + w log p_expert', w_grid=W_GRID,
                      selection='highest mean dev macro-F1 over pairs; ties to the smaller w',
                      gate=f'cross-fitted dev gain >= {CONFIG["gate_min_gain"]}', criteria=CRITERIA))
    if not cuda:
        print('WARNING: no GPU. Fine-tuning RoBERTa-large on CPU is far too slow; switch the accelerator to GPU.')
    P.log_event(S, 'setup', device=S['device'], gpu=S['env']['gpu'], torch=torch.__version__,
                transformers=transformers.__version__)
    return S


def stage_prepare(S):
    """Downloads (checkpoints from HuggingFace, MELD features from Google Drive), the data gate, text inputs."""
    from transformers import AutoTokenizer
    P.stage_download(S)
    P.stage_load_data(S)  # reads train/dev labels only
    D = S['D']
    ok = all(len(D['sentences'][v]) == len(D['videoAudio'][v]) and all(isinstance(x, str) for x in D['sentences'][v])
             for s in ('train', 'valid', 'test') for v in S['keys'][s])
    P.record_check(S, 'utterance text is present for every utterance (label-free check)', ok, hard=True)
    tok = AutoTokenizer.from_pretrained(S['cfg']['expert_model'])
    tok.truncation_side = 'left'
    S['tok'] = tok
    S['enc'] = {s: encode(S, s, with_labels=True) for s in ('train', 'valid')}
    lens = [len(x) for x in S['enc']['train']['ids']]
    print('  example input:', S['enc']['train']['example'])
    P.record_check(S, 'every input keeps its <mask> prompt after truncation',
                   all(tok.mask_token_id in row for s in ('train', 'valid') for row in S['enc'][s]['ids']), hard=True,
                   detail=f'max {max(lens)} tokens, mean {np.mean(lens):.0f}; truncated at {S["cfg"]["expert_max_len"]}')
    P._save_checks(S)
    P.log_event(S, 'inputs ready', train=S['enc']['train']['n'], dev=S['enc']['valid']['n'])


def stage_ecerc_dev(S):
    S['ecerc_models'], S['ecerc_dev'] = [], []
    dev_ds = P.make_dataset(S, 'valid')
    for ck in S['ckpts']:
        model = P.build_model(S)
        model.load_state_dict(torch.load(ck['path'], map_location=S['device'], weights_only=True), strict=True)
        model.to(S['device']).eval()
        d = P.run_inference(S, model, dev_ds)
        acc = float((d['logp'].argmax(1) == d['y']).mean())
        P.record_check(S, f'{ck["name"]}: released ECERC dev accuracy >= 55%', acc >= S['cfg']['min_dev_acc'], hard=True,
                       detail=f'{100 * acc:.2f}%')
        S['ecerc_models'].append(model)
        S['ecerc_dev'].append(d)
    P._save_checks(S)


def stage_train_experts(S):
    t0 = time.time()
    S['experts'] = []
    floor = S['cfg'].get('expert_min_dev_mf1', 0.0)
    for sd in S['cfg']['expert_seeds']:
        r, retried = train_expert(S, sd), None
        if r['dev_macro_f1'] < floor:  # pre-registered: one retry for a collapsed run (decided on dev only)
            print(f'  seed {sd} collapsed (best dev macro-F1 {r["dev_macro_f1"]:.2f} < {floor}); one retry with seed {sd + 1000}')
            retried = dict(seed=sd, dev_macro_f1=r['dev_macro_f1'])
            r = train_expert(S, sd + 1000)
        dv = S['ecerc_dev'][0]
        S['experts'].append(dict(seed=r['seed'], path=r['path'], epoch=r['epoch'], dev_macro_f1=r['dev_macro_f1'],
                                 curve=r['curve'], seconds=r['seconds'], retried_after=retried,
                                 dev=dict(logp=r['dev_logp'], y=S['enc']['valid']['y'], dlg=dv['dlg'])))
        P.log_event(S, 'expert trained', seed=r['seed'], best_epoch=r['epoch'], dev_macro_f1=round(r['dev_macro_f1'], 2),
                    minutes=round(r['seconds'] / 60, 1), retry=retried is not None)
    P.write_json(os.path.join(S['out'], 'expert_training.json'),
                 [{k: v for k, v in e.items() if k not in ('dev', 'path')} for e in S['experts']])
    P.log_event(S, 'experts done', minutes=round((time.time() - t0) / 60, 1))


def stage_dev_analysis(S):
    cfg = S['cfg']
    E, T = S['ecerc_dev'], [e['dev'] for e in S['experts']]
    n_dev = len(S['keys']['valid'])
    S['dev_cms'] = pair_cms(E, T, n_dev)
    r_star, mean = _select(S['dev_cms'].sum(0))
    fc = crossfit_forecast(S['dev_cms'], cfg['n_folds'], cfg['fold_seed'], cfg['power_B'], len(S['keys']['test']))
    S['dev_cms_tau'] = pair_cms(E, T, n_dev, TAU_GRID, S['log_prior'])
    r2, _ = _select(S['dev_cms_tau'].sum(0))
    w2, t2 = W_GRID[r2 // len(TAU_GRID)], TAU_GRID[r2 % len(TAU_GRID)]
    proceed = fc['crossfit_delta_macro_f1'] >= cfg['gate_min_gain']
    alone = []
    for e in S['experts']:
        m = P.scores(_cm(e['dev']['y'], e['dev']['logp'].argmax(1)))
        alone.append(dict(seed=e['seed'], macro_f1=100 * m['mf1'], weighted_f1=100 * m['wf1'], acc=100 * m['acc'],
                          f1=100 * m['f1']))
    ecerc = [dict(name=c['name'], **{k: 100 * v for k, v in P.scores(_cm(d['y'], d['logp'].argmax(1))).items()
                                     if k in ('mf1', 'wf1', 'acc', 'f1')}) for c, d in zip(S['ckpts'], E)]
    div = {}
    for c in (FEA, DIS, SAD):
        rows = []
        for e in E:
            for t in T:
                m = e['y'] == c
                ew, tw = e['logp'].argmax(1)[m] != c, t['logp'].argmax(1)[m] != c
                rows.append([int((ew & ~tw).sum()), int((~ew & tw).sum()), int((ew & tw).sum()), int(m.sum())])
        r = np.mean(rows, 0)
        div[CLASSES[c]] = dict(ecerc_wrong_expert_right=r[0], ecerc_right_expert_wrong=r[1], both_wrong=r[2], n=r[3])
    S['selection'] = dict(w_star=W_GRID[r_star], dev_mean_macro_f1_by_w={str(w): 100 * m for w, m in zip(W_GRID, mean)},
                          forecast=fc, proceed=proceed, secondary_w_tau=dict(w=w2, tau=t2),
                          expert_alone_dev=alone, ecerc_alone_dev=ecerc, diversity_dev=div)
    P.write_json(os.path.join(S['out'], 'dev_selection.json'), S['selection'])
    print('\nDEV-ONLY RESULTS (frozen before the test set is touched)')
    print('  ECERC alone, dev macro-F1:', ', '.join(f'{e["name"]} {e["mf1"]:.2f}' for e in ecerc))
    print('  expert alone, dev macro-F1:', ', '.join(f'seed {a["seed"]} {a["macro_f1"]:.2f}' for a in alone))
    print(f'  w* = {W_GRID[r_star]}: mean dev macro-F1 {100 * mean[r_star]:.2f} vs {100 * mean[0]:.2f} for ECERC alone')
    print(f'  cross-fitted dev gain {fc["crossfit_delta_macro_f1"]:+.2f} (fold w {fc["fold_w"]}); projected test SE '
          f'{fc["se_test_projected"]:.2f}; gate {cfg["gate_min_gain"]:+.2f} -> ' + ('TEST WILL RUN' if proceed else 'test skipped'))
    print('  Fear/Disgust/Sadness on dev (mean per pair): ' + '; '.join(
        f'{k}: ECERC wrong but expert right {v["ecerc_wrong_expert_right"]:.1f}, the reverse {v["ecerc_right_expert_wrong"]:.1f} '
        f'of {v["n"]:.0f}' for k, v in div.items()))
    P.log_event(S, 'dev analysis done', w_star=W_GRID[r_star], forecast=round(fc['crossfit_delta_macro_f1'], 2),
                proceed=proceed)


def stage_freeze(S):
    sel = S['selection']
    frozen = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),
                  note='Written before any test-set access. The test evaluation reads its rules from this file.',
                  code_sha256=S['env']['code_sha256'], data_sha256=S['data_sha256'],
                  ecerc_checkpoints=[dict(name=c['name'], sha256=c['sha256'], archive=c['archive']) for c in S['ckpts']],
                  experts=[dict(seed=e['seed'], best_epoch=e['epoch'], dev_macro_f1=e['dev_macro_f1']) for e in S['experts']],
                  expert_model=S['cfg']['expert_model'], w_grid=W_GRID, w_star=sel['w_star'],
                  secondary_w_tau=sel['secondary_w_tau'], log_prior_train=S['log_prior'],
                  gate=dict(statistic='cross-fitted dev gain', value=sel['forecast']['crossfit_delta_macro_f1'],
                            threshold=S['cfg']['gate_min_gain'], proceed=sel['proceed']),
                  forecast=sel['forecast'],
                  bootstrap=dict(B=S['cfg']['bootstrap_B'], seed=S['cfg']['boot_seed'], unit='dialogue',
                                 design='dev and test resampled independently; w re-selected in every replicate'),
                  criteria=CRITERIA, device=S['device'], env=S['env'])
    path = os.path.join(S['out'], 'frozen_config_expert.json')
    P.write_json(path, frozen)
    S['frozen_path'], S['frozen_sha256'] = path, P.sha256_file(path)
    P.log_event(S, 'FREEZE', sha256=S['frozen_sha256'][:16], w_star=sel['w_star'], proceed=sel['proceed'])


def stage_test(S):
    if not S.get('frozen_path') or P.sha256_file(S['frozen_path']) != S['frozen_sha256']:
        raise P.CheckFailed('frozen_config_expert.json is missing or was modified: run the freeze cell first')
    with open(S['frozen_path']) as f:
        F_ = json.load(f)
    if any(P.sha256_file(os.path.join(S['code_dir'], n)) != h for n, h in F_['code_sha256'].items()):
        raise P.CheckFailed('code changed after the freeze')
    if not F_['gate']['proceed']:
        print('Test skipped by the pre-registered gate: the cross-fitted dev gain is below the threshold.')
        S['results'] = None
        _write_summary(S, F_)
        return
    P.log_event(S, 'TEST evaluation starts (test dialogues are run and test labels read only from here on)')
    test_ds = P.make_dataset(S, 'test')
    E = [P.run_inference(S, m, test_ds) for m in S['ecerc_models']]
    te = encode(S, 'test', with_labels=True)
    T = []
    for e in S['experts']:
        model = load_expert(S, e['path'])
        T.append(dict(logp=predict_expert(S, model, te), y=te['y'], dlg=E[0]['dlg']))
        del model
        if S['device'] == 'cuda':
            torch.cuda.empty_cache()
    for c, d in zip(S['ckpts'], E):
        np.savez_compressed(os.path.join(S['out'], 'dumps', f'test_ecerc_{c["name"]}.npz'), logp=d['logp'].astype(np.float32),
                            y=d['y'], dlg=d['dlg'])
    for e, d in zip(S['experts'], T):
        np.savez_compressed(os.path.join(S['out'], 'dumps', f'test_expert_seed{e["seed"]}.npz'),
                            logp=d['logp'].astype(np.float32), y=d['y'])
    n_te = len(S['keys']['test'])
    test_cms = pair_cms(E, T, n_te)
    K = test_cms.shape[1]
    r_star = W_GRID.index(F_['w_star'])
    sc = P.scores(test_cms.sum(0).reshape(K, len(W_GRID), C, C))

    def arm(r):
        return {k: 100 * sc[k][:, r].mean(0) for k in ('acc', 'mf1', 'wf1', 'f1', 'prec', 'rec')}

    base, fused, alone = arm(0), arm(r_star), arm(len(W_GRID) - 1)
    per_pair = 100 * (sc['mf1'][:, r_star] - sc['mf1'][:, 0])
    bt = procedure_bootstrap(S['dev_cms'], test_cms, F_['bootstrap']['B'], F_['bootstrap']['seed'])
    ci = lambda x: np.percentile(x, [2.5, 97.5], axis=0)  # noqa: E731
    lo, hi = ci(bt['mf1'])
    d = {k: fused[k] - base[k] for k in fused}
    verdict, cond = decide(float(d['mf1']), lo, hi, d['f1'], float(d['acc']), float(d['wf1']), float((per_pair > 0).mean()))
    # secondary: joint (w, tau) chosen on dev, test-only bootstrap at that fixed rule
    test_cms_tau = pair_cms(E, T, n_te, TAU_GRID, np.asarray(F_['log_prior_train']))
    r2 = W_GRID.index(F_['secondary_w_tau']['w']) * len(TAU_GRID) + TAU_GRID.index(F_['secondary_w_tau']['tau'])
    sc2 = P.scores(test_cms_tau.sum(0).reshape(K, -1, C, C))
    sec = 100 * (sc2['mf1'][:, r2] - sc2['mf1'][:, 0]).mean()
    rng = np.random.default_rng(F_['bootstrap']['seed'] + 1)
    flat2 = test_cms_tau[:, :, [0, r2], :].reshape(n_te, -1).astype(float)
    b2, done = [], 0
    while done < F_['bootstrap']['B']:
        nb = min(1000, F_['bootstrap']['B'] - done)
        s2 = P.scores((P._boot_weights(rng, n_te, nb) @ flat2).reshape(nb, K, 2, C, C))['mf1']
        b2.append(100 * (s2[:, :, 1] - s2[:, :, 0]).mean(1))
        done += nb
    # descriptive: full ensembles (mean of the 3 ECERC and of the expert log-probs)
    eE, eT = np.mean([x['logp'] for x in E], 0), np.mean([x['logp'] for x in T], 0)
    y = E[0]['y']
    ens = {name: 100 * P.scores(_cm(y, s.argmax(1)))['mf1'] for name, s in
           (('ECERC x3 ensemble', eE), ('expert ensemble', eT),
            ('fused ensembles at w*', (1 - F_['w_star']) * eE + F_['w_star'] * eT))}
    S['results'] = dict(
        verdict=verdict, conditions=cond, w_star=F_['w_star'],
        delta_macro_f1=float(d['mf1']), ci95=[lo, hi], per_pair=per_pair, pairs_positive=int((per_pair > 0).sum()),
        ci_kind='procedure-level two-sample dialogue bootstrap (w re-selected per replicate)',
        per_class=dict(delta_f1=d['f1'], delta_f1_ci95=ci(bt['f1']).T, delta_precision=d['prec'], delta_recall=d['rec']),
        guardrails=dict(delta_acc=float(d['acc']), delta_acc_ci95=ci(bt['acc']), delta_weighted_f1=float(d['wf1']),
                        delta_weighted_f1_ci95=ci(bt['wf1'])),
        arms={'ECERC alone (w=0)': base, f'fused (w={F_["w_star"]})': fused, 'expert alone (w=1)': alone},
        secondary=dict(w_tau=F_['secondary_w_tau'], delta_macro_f1=float(sec), test_only_ci95=ci(np.concatenate(b2))),
        ensembles_descriptive=ens, support=sc['support'][0, 0],
        bootstrap_w_selected=dict(zip(*np.unique(bt['w_sel'], return_counts=True))))
    P.write_json(os.path.join(S['out'], 'test_results_expert.json'), S['results'])
    _write_summary(S, F_)
    P.log_event(S, 'TEST done', verdict=verdict, delta=round(float(d['mf1']), 2), ci=[round(lo, 2), round(hi, 2)])


def _write_summary(S, F_):
    sel, env = S['selection'], S['env']
    fc = sel['forecast']
    L = ['# ECERC Limitation 2, Experiment 4: context-aware text expert fused with ECERC', '',
         f'Run {time.strftime("%Y-%m-%d %H:%M")} | {env["device"]} ({env["gpu"]}) | torch {env["torch"]} | '
         f'transformers {env["transformers"]} | data sha256 {S["data_sha256"][:16]}...', '',
         '## Dev (selection only)', '',
         '- ECERC alone, dev macro-F1: ' + ', '.join(f'{e["name"]} {e["mf1"]:.2f}' for e in sel['ecerc_alone_dev']),
         '- expert alone, dev macro-F1: ' + ', '.join(f'seed {a["seed"]} {a["macro_f1"]:.2f} (weighted F1 {a["weighted_f1"]:.2f})'
                                                    for a in sel['expert_alone_dev']),
         f'- w* = {sel["w_star"]}; dev macro-F1 {sel["dev_mean_macro_f1_by_w"][str(sel["w_star"])]:.2f} vs '
         f'{sel["dev_mean_macro_f1_by_w"]["0.0"]:.2f} for ECERC alone',
         f'- cross-fitted dev gain **{fc["crossfit_delta_macro_f1"]:+.2f}** (gate {S["cfg"]["gate_min_gain"]:+.2f}: '
         + ('passed' if sel['proceed'] else 'not passed, so the test was skipped') + f'); projected test SE {fc["se_test_projected"]:.2f}',
         '- Fear/Disgust/Sadness on dev, mean per pair: ' + '; '.join(
             f'{k} ECERC wrong but expert right {v["ecerc_wrong_expert_right"]:.1f} / reverse {v["ecerc_right_expert_wrong"]:.1f} '
             f'(n={v["n"]:.0f})' for k, v in sel['diversity_dev'].items()), '']
    res = S.get('results')
    if res:
        L += [f'## Verdict: {res["verdict"]}', '', '| condition | met |', '|---|---|']
        L += [f'| {k} | {"yes" if v else "no"} |' for k, v in res['conditions'].items()]
        L += ['', '## Test (mean over the 9 checkpoint x seed pairs)', '',
              '| arm | acc | weighted F1 | macro-F1 | ' + ' | '.join(f'F1 {c}' for c in CLASSES) + ' |', '|---' * (4 + C) + '|']
        for name, m in res['arms'].items():
            L.append(f'| {name} | {m["acc"]:.2f} | {m["wf1"]:.2f} | {m["mf1"]:.2f} | ' + ' | '.join(f'{x:.2f}' for x in m['f1']) + ' |')
        g, pc = res['guardrails'], res['per_class']
        L += ['', f'- Delta macro-F1 (fused - ECERC alone): **{P._fmt(res["delta_macro_f1"], *res["ci95"])}** ({res["ci_kind"]})',
              f'- positive in {res["pairs_positive"]}/{len(res["per_pair"])} pairs: ' + ', '.join(f'{x:+.2f}' for x in res['per_pair']),
              f'- Delta accuracy {P._fmt(g["delta_acc"], *g["delta_acc_ci95"])}; Delta weighted F1 '
              f'{P._fmt(g["delta_weighted_f1"], *g["delta_weighted_f1_ci95"])}', '',
              '| class | Delta F1 [95% CI] | Delta precision | Delta recall | support |', '|---|---|---|---|---|']
        for c in range(C):
            L.append(f'| {CLASSES[c]} | {P._fmt(pc["delta_f1"][c], *pc["delta_f1_ci95"][c])} | {pc["delta_precision"][c]:+.2f} | '
                     f'{pc["delta_recall"][c]:+.2f} | {int(res["support"][c])} |')
        s = res['secondary']
        L += ['', f'- secondary, w and tau chosen together on dev ({s["w_tau"]}): {P._fmt(s["delta_macro_f1"], *s["test_only_ci95"])}',
              '- descriptive ensembles, test macro-F1: ' + ', '.join(f'{k} {v:.2f}' for k, v in res['ensembles_descriptive'].items())]
    warn = [c for c in S['checks'] if not c['passed']]
    L += ['', f'Checks: {sum(c["passed"] for c in S["checks"])} passed, {len(warn)} warnings'
          + ('' if not warn else ' -- ' + '; '.join(f'{c["name"]} ({c["detail"]})' for c in warn))]
    with open(os.path.join(S['out'], 'SUMMARY_expert.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')


def stage_package(S):
    zpath = os.path.join(S['cfg']['work_dir'], 'ecerc_l2_expert_results.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(S['out']):
            for name in files:
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, os.path.dirname(S['out'])))
    with open(os.path.join(S['out'], 'SUMMARY_expert.md')) as f:
        print(f.read())
    print(f'Results archive: {zpath} ({os.path.getsize(zpath) / 2 ** 20:.1f} MB)')
    return zpath
