"""
ECERC Limitation 2, Experiment 7: a zero-shot LLM fused with ECERC.

New code for research/limitation2. Experiments 4 and 5 fused ECERC with experts fine-tuned on MELD; both inherit
MELD's label skew, so their fusion traded one class against another. A zero-shot LLM is never trained on MELD's
labels, and published zero-shot results on the MELD test set show the error profile ECERC lacks: higher F1 on Fear,
Disgust and Sadness, lower on Neutral and Joy. This experiment runs a frozen instruction LLM on every dev and test
utterance with its past context only (ECERC's task setting) and reads its next-token probabilities for the seven
label words. It removes the LLM's own label bias without labels (prior matching: the LLM's mean prediction on the
dev inputs is replaced by the training prior; frozen before the test) and fuses it with each released ECERC
checkpoint through one weight chosen on dev:

    score = (1 - w) * log p_ECERC + w * log p~_LLM,   w in {0.00, 0.05, ..., 1.00}  (w = 0 is ECERC alone)

The pairs are the released checkpoints C1-C3 with the one LLM. Primary estimand: the mean over the three pairs of test
macro-F1(fused) - macro-F1(ECERC alone). The test runs only if the cross-fitted dev gain is at least
CONFIG['gate_min_gain'], and only after every decision is frozen in frozen_config_zeroshot.json. The selection,
gate, bootstrap and verdict are Experiment 4's own functions. A dev-only pilot with Qwen2.5-3B fixed the prompt
and motivated the calibration (PROTOCOL.md, addendum of 2026-10-19).
"""
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

import l2_posthoc as P  # downloads, data gate, ECERC builders, inference and metrics (unchanged)
import l2_expert as LE  # Experiment 4: fusion grid, selection, cross-fitted gate, procedure bootstrap, verdict

C, CLASSES = P.C, P.CLASSES
NEU, SUR, FEA, SAD, JOY, DIS, ANG = range(C)
W_GRID = LE.W_GRID
LABELS = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']  # MELD code order
SYSTEM = 'You are an expert annotator of emotions in TV-show dialogue.'
QUESTION = ('Which emotion does {who} express in the last line? Answer with one word: neutral, surprise, fear, '
            'sadness, joy, disgust, or anger.')
PROMPT_SHA256 = hashlib.sha256((SYSTEM + '\n' + QUESTION).encode()).hexdigest()
CRITERIA = dict(LE.CRITERIA, primary='mean over the three released checkpoints of test macro-F1(fused) - '
                                     'macro-F1(ECERC alone), in points')


def _as_ids(x):
    return list(x['input_ids']) if hasattr(x, 'keys') else list(x)


def label_token_ids(tok):
    """Per class, the first tokens of the label word written four ways; they must not collide across classes."""
    ids = []
    for w in LABELS:
        ids.append(sorted({tok.encode(v, add_special_tokens=False)[0]
                           for v in (w, w.capitalize(), ' ' + w, ' ' + w.capitalize())}))
    flat = [i for g in ids for i in g]
    return ids, len(flat) == len(set(flat))


def utterance_messages(D, vid):
    """One chat per utterance: the dialogue so far ("Speaker k: text", generic labels) and the question."""
    sents = D['sentences'][vid]
    spk = np.asarray(D['speakers'][vid]).argmax(-1)
    turns = [f'Speaker {int(spk[i]) + 1}: {LE._clean(sents[i])}' for i in range(len(sents))]
    for t in range(len(sents)):
        yield [dict(role='system', content=SYSTEM),
               dict(role='user', content='Here is a dialogue, in order:\n' + '\n'.join(turns[:t + 1]) + '\n\n'
                    + QUESTION.format(who=f'Speaker {int(spk[t]) + 1}'))]


def load_llm(S):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg, cuda = S['cfg'], S['device'] == 'cuda'
    kw = dict(revision=cfg.get('llm_revision'), cache_dir=os.path.join(cfg['cache_dir'], 'hf'))
    tok = AutoTokenizer.from_pretrained(cfg['llm_model'], **kw)
    mkw = dict(kw)
    if cuda:  # the pre-quantized 4-bit weights are spread over every visible GPU
        mkw.update(device_map='auto', max_memory={i: cfg['llm_max_memory'] for i in range(torch.cuda.device_count())})
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(cfg['llm_model'], dtype=torch.float16 if cuda else torch.float32, **mkw)
    model.eval()
    P.log_event(S, 'LLM loaded', model=cfg['llm_model'], seconds=round(time.time() - t0, 1),
                devices=sorted({str(p.device) for p in model.parameters()}))
    return model, tok


def llm_logprobs(S, model, tok, split):
    """(N, 7) log-probabilities over the label words for every utterance of `split`, in data-file order. Label-free:
    only the text and speakers are read. The cached prefix (system prompt and dialogue so far) is reused across turns;
    a local check showed the cached logits equal a full recompute (max difference about 1e-7)."""
    from transformers import DynamicCache
    D, ids, keys = S['D'], S['label_ids'], S['keys'][split]
    dev0 = model.get_input_embeddings().weight.device
    cache, cached, rows, n_new, t0, report = DynamicCache(), [], [], 0, time.time(), 0
    with torch.no_grad():
        for d, vid in enumerate(keys):
            for msgs in utterance_messages(D, vid):
                toks = _as_ids(tok.apply_chat_template(msgs, add_generation_prompt=True))
                k = 0
                for a, b in zip(cached, toks):
                    if a != b:
                        break
                    k += 1
                k = min(k, len(toks) - 1)
                if len(cached) > k:
                    cache.crop(k - len(cached))  # negative: drop that many tokens from the end
                x = torch.tensor([toks[k:]], device=dev0)
                try:
                    out = model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
                except TypeError:
                    out = model(input_ids=x, past_key_values=cache, use_cache=True)
                lp = torch.log_softmax(out.logits[0, -1].float(), -1).cpu()
                rows.append(np.array([float(torch.logsumexp(lp[g], 0)) for g in ids]))
                n_new += len(toks) - k
                cached = toks
            if d + 1 >= report or d + 1 == len(keys):
                print(f'  [LLM {split}] {d + 1}/{len(keys)} dialogues, {len(rows)} utterances, {n_new} new tokens, '
                      f'{time.time() - t0:.0f}s', flush=True)
                report += max(1, len(keys) // 10)
    raw = np.stack(rows)
    mass = np.logaddexp.reduce(raw, axis=1)
    return raw - mass[:, None], dict(split=split, utterances=len(rows), new_tokens=n_new,
                                     seconds=round(time.time() - t0, 1), label_mass_mean=float(np.exp(mass).mean()),
                                     label_mass_min=float(np.exp(mass).min()))


def prior_match(lp, log_q, log_prior):
    """Label-free calibration: divide out the LLM's mean prediction q, multiply in the training prior, renormalise."""
    s = lp - log_q[None, :] + log_prior[None, :]
    return s - np.logaddexp.reduce(s, axis=1, keepdims=True)


def _metrics(y, logp):
    m = P.scores(LE._cm(y, logp.argmax(1)))
    return dict(acc=100 * float(m['acc']), weighted_f1=100 * float(m['wf1']), macro_f1=100 * float(m['mf1']),
                f1=100 * m['f1'], predicted=np.bincount(logp.argmax(1), minlength=C))


# ----------------------------------------------------------------------------- stages
def stage_setup(CONFIG, code_dir):
    out = os.path.join(CONFIG['work_dir'], 'outputs_zeroshot')
    for sub in ('', 'tables', 'dumps'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    os.makedirs(os.path.join(CONFIG['cache_dir'], 'hf'), exist_ok=True)
    cuda = torch.cuda.is_available()
    if cuda:  # 4-bit loading needs bitsandbytes and accelerate (as in Experiment 5)
        need = [p for p in ('bitsandbytes', 'accelerate') if importlib.util.find_spec(p) is None]
        if need:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + need, check=False)
    import pandas
    import sklearn
    import transformers
    S = dict(cfg=CONFIG, out=out, code_dir=code_dir, checks=[], torch=torch)
    if not cuda:
        P.install_cpu_cuda_shim(torch)
    S['device'] = 'cuda' if cuda else 'cpu'
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    for name in ('model', 'dataloader', 'loss', 'inference'):
        S['m_' + name] = __import__(name)
    versions = {}
    for p in ('bitsandbytes', 'accelerate'):
        versions[p] = __import__(p).__version__ if importlib.util.find_spec(p) else None
    gpus = [(torch.cuda.get_device_name(i), round(torch.cuda.get_device_properties(i).total_memory / 2 ** 30, 1))
            for i in range(torch.cuda.device_count())] if cuda else []
    S['env'] = dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                    transformers=transformers.__version__, numpy=np.__version__, pandas=pandas.__version__,
                    sklearn=sklearn.__version__, device=S['device'], gpus=gpus, **versions,
                    code_sha256={f: P.sha256_file(os.path.join(code_dir, f))
                                 for f in sorted(os.listdir(code_dir)) if f.endswith('.py')})
    P.write_json(os.path.join(out, 'environment.json'), S['env'])
    P.write_json(os.path.join(out, 'design.json'),  # written before any data is loaded
                 dict(llm=dict(model=CONFIG['llm_model'], revision=CONFIG.get('llm_revision')),
                      prompt=dict(system=SYSTEM, question=QUESTION, sha256=PROMPT_SHA256,
                                  context='past turns and the target, "Speaker k: text", generic speaker labels'),
                      scoring='next-token log-probabilities of the first tokens of each label word (4 spellings), '
                              'log-sum-exp per class, renormalised over the seven classes',
                      calibration='prior matching: p~ ∝ p / q * prior_train, q = the LLM\'s mean prediction on the dev '
                                  'inputs (label-free), frozen before the test',
                      fusion='score = (1 - w) log p_ECERC + w log p~_LLM', w_grid=W_GRID,
                      selection='highest mean dev macro-F1 over the three checkpoints; ties to the smaller w',
                      gate=f'cross-fitted dev gain >= {CONFIG["gate_min_gain"]}', test_looks=CONFIG['test_looks'],
                      secondary=['fusion without calibration (its own dev-chosen w; test-only CI)', 'LLM alone'],
                      pilot='dev-only pilot with Qwen2.5-3B-Instruct (2026-10-19), see PROTOCOL.md',
                      criteria=CRITERIA))
    if cuda:
        total = sum(g[1] for g in gpus)
        P.record_check(S, f'GPU memory is enough for the 4-bit LLM (>= {CONFIG["llm_min_gpu_gb"]} GB in total)',
                       total >= CONFIG['llm_min_gpu_gb'], hard=True,
                       detail=f'{gpus}; on Kaggle choose Accelerator -> GPU T4 x2')
    else:
        print('WARNING: no GPU. A 32B model cannot run on CPU; switch the accelerator to GPU T4 x2.')
    P._save_checks(S)
    P.log_event(S, 'setup', device=S['device'], gpus=gpus, torch=torch.__version__, transformers=transformers.__version__)
    return S


def stage_prepare(S):
    """Downloads (checkpoints from HuggingFace, MELD features from Google Drive) and the data gate."""
    P.stage_download(S)
    P.stage_load_data(S)  # reads train/dev labels only
    D = S['D']
    ok = all(len(D['sentences'][v]) == len(D['videoAudio'][v]) and all(isinstance(x, str) for x in D['sentences'][v])
             for s in ('train', 'valid', 'test') for v in S['keys'][s])
    P.record_check(S, 'utterance text is present for every utterance (label-free check)', ok, hard=True)
    P._save_checks(S)


def stage_ecerc_dev(S):
    LE.stage_ecerc_dev(S)  # the released checkpoints on dev, with Experiment 4's accuracy floor


def stage_llm_dev(S):
    model, tok = load_llm(S)
    S['llm'] = (model, tok)
    S['label_ids'], unique = label_token_ids(tok)
    spelled = {w: [tok.decode([i]) for i in g] for w, g in zip(LABELS, S['label_ids'])}
    P.record_check(S, 'label words start with distinct tokens', unique, hard=True, detail=spelled)
    lp, info = llm_logprobs(S, model, tok, 'valid')
    e = S['ecerc_dev'][0]
    S['llm_dev'] = dict(logp=lp, y=e['y'], dlg=e['dlg'])
    P.record_check(S, 'the seven label words carry most of the next-token probability on dev',
                   info['label_mass_mean'] >= 0.5, hard=False, detail=f'mean {info["label_mass_mean"]:.3f}')
    np.savez_compressed(os.path.join(S['out'], 'dumps', 'dev_llm.npz'), logp=lp.astype(np.float32), y=e['y'])
    S['llm_info'] = dict(label_tokens=spelled, dev=info)
    P.write_json(os.path.join(S['out'], 'llm_inference.json'), S['llm_info'])
    P._save_checks(S)
    P.log_event(S, 'LLM dev predictions done', minutes=round(info['seconds'] / 60, 1),
                label_mass=round(info['label_mass_mean'], 3))


def stage_dev_analysis(S):
    cfg, E = S['cfg'], S['ecerc_dev']
    n_dev, y = len(S['keys']['valid']), S['llm_dev']['y']
    lp = S['llm_dev']['logp']
    log_q = np.log(np.exp(lp).mean(0))  # label-free: the LLM's own mean prediction on the dev inputs
    cal = prior_match(lp, log_q, S['log_prior'])
    S['log_q'] = log_q
    S['dev_cms'] = LE.pair_cms(E, [dict(logp=cal, y=y, dlg=S['llm_dev']['dlg'])], n_dev)
    r_star, mean = LE._select(S['dev_cms'].sum(0))
    fc = LE.crossfit_forecast(S['dev_cms'], cfg['n_folds'], cfg['fold_seed'], cfg['power_B'], len(S['keys']['test']))
    S['dev_cms_uncal'] = LE.pair_cms(E, [dict(logp=lp, y=y, dlg=S['llm_dev']['dlg'])], n_dev)
    r_u, mean_u = LE._select(S['dev_cms_uncal'].sum(0))
    fc_u = LE.crossfit_forecast(S['dev_cms_uncal'], cfg['n_folds'], cfg['fold_seed'], cfg['power_B'],
                                len(S['keys']['test']))
    proceed = fc['crossfit_delta_macro_f1'] >= cfg['gate_min_gain']
    sc = P.scores(S['dev_cms'].sum(0).reshape(len(E), len(W_GRID), C, C))
    div = {}
    for c in (FEA, DIS, SAD):
        m = y == c
        rows = [[int(((e['logp'].argmax(1)[m] != c) & (cal.argmax(1)[m] == c)).sum()),
                 int(((e['logp'].argmax(1)[m] == c) & (cal.argmax(1)[m] != c)).sum())] for e in E]
        r = np.mean(rows, 0)
        div[CLASSES[c]] = dict(ecerc_wrong_llm_right=r[0], ecerc_right_llm_wrong=r[1], n=int(m.sum()))
    S['selection'] = dict(
        w_star=W_GRID[r_star], dev_mean_macro_f1_by_w={str(w): 100 * m for w, m in zip(W_GRID, mean)},
        forecast=fc, proceed=proceed, log_q_dev=log_q,
        dev_change_at_w_star=dict(macro_f1=100 * float(mean[r_star] - mean[0]),
                                  weighted_f1=100 * float((sc['wf1'][:, r_star] - sc['wf1'][:, 0]).mean()),
                                  acc=100 * float((sc['acc'][:, r_star] - sc['acc'][:, 0]).mean()),
                                  f1=100 * (sc['f1'][:, r_star] - sc['f1'][:, 0]).mean(0)),
        secondary_uncalibrated=dict(w_star=W_GRID[r_u], in_sample_gain=100 * float(mean_u[r_u] - mean_u[0]),
                                    crossfit_delta_macro_f1=fc_u['crossfit_delta_macro_f1']),
        llm_alone_dev=dict(raw=_metrics(y, lp), calibrated=_metrics(y, cal)),
        ecerc_alone_dev=[dict(name=c['name'], **_metrics(d['y'], d['logp'])) for c, d in zip(S['ckpts'], E)],
        diversity_dev=div)
    P.write_json(os.path.join(S['out'], 'dev_selection.json'), S['selection'])
    a = S['selection']['llm_alone_dev']
    print('\nDEV-ONLY RESULTS (frozen before the test set is touched)')
    print('  ECERC alone, dev macro-F1:', ', '.join(f'{e["name"]} {e["macro_f1"]:.2f}' for e in S['selection']['ecerc_alone_dev']))
    for k in ('raw', 'calibrated'):
        print(f'  LLM alone ({k}): acc {a[k]["acc"]:.2f}, weighted F1 {a[k]["weighted_f1"]:.2f}, macro-F1 '
              f'{a[k]["macro_f1"]:.2f} | ' + ' '.join(f'{c} {f:.1f}' for c, f in zip(CLASSES, a[k]['f1'])))
    print(f'  w* = {W_GRID[r_star]}: mean dev macro-F1 {100 * mean[r_star]:.2f} vs {100 * mean[0]:.2f} for ECERC alone')
    print(f'  cross-fitted dev gain {fc["crossfit_delta_macro_f1"]:+.2f} (fold w {fc["fold_w"]}); projected test SE '
          f'{fc["se_test_projected"]:.2f}; gate {cfg["gate_min_gain"]:+.2f} -> ' + ('TEST WILL RUN' if proceed else 'test skipped'))
    print(f'  secondary, without calibration: w* {W_GRID[r_u]}, cross-fitted dev gain {fc_u["crossfit_delta_macro_f1"]:+.2f}')
    print('  Fear/Disgust/Sadness on dev (mean per checkpoint): ' + '; '.join(
        f'{k}: ECERC wrong but LLM right {v["ecerc_wrong_llm_right"]:.1f}, the reverse {v["ecerc_right_llm_wrong"]:.1f} '
        f'of {v["n"]}' for k, v in div.items()))
    P.log_event(S, 'dev analysis done', w_star=W_GRID[r_star], forecast=round(fc['crossfit_delta_macro_f1'], 2),
                proceed=proceed)


def stage_freeze(S):
    sel, cfg = S['selection'], S['cfg']
    frozen = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),
                  note='Written before any test-set access. The test evaluation reads its rules from this file.',
                  code_sha256=S['env']['code_sha256'], data_sha256=S['data_sha256'],
                  ecerc_checkpoints=[dict(name=c['name'], sha256=c['sha256'], archive=c['archive']) for c in S['ckpts']],
                  llm=dict(model=cfg['llm_model'], revision=cfg.get('llm_revision'), label_token_ids=S['label_ids'],
                           prompt_sha256=PROMPT_SHA256),
                  log_q_dev=S['log_q'], log_prior_train=S['log_prior'], w_grid=W_GRID, w_star=sel['w_star'],
                  secondary_uncalibrated_w=sel['secondary_uncalibrated']['w_star'],
                  gate=dict(statistic='cross-fitted dev gain', value=sel['forecast']['crossfit_delta_macro_f1'],
                            threshold=cfg['gate_min_gain'], proceed=sel['proceed']),
                  forecast=sel['forecast'],
                  bootstrap=dict(B=cfg['bootstrap_B'], seed=cfg['boot_seed'], unit='dialogue',
                                 design='dev and test resampled independently; w re-selected in every replicate'),
                  multiplicity=dict(test_looks=cfg['test_looks'],
                                    note='the verdict uses the 95% rule; a Bonferroni-adjusted interval is also reported'),
                  criteria=CRITERIA, device=S['device'], env=S['env'])
    path = os.path.join(S['out'], 'frozen_config_zeroshot.json')
    P.write_json(path, frozen)
    S['frozen_path'], S['frozen_sha256'] = path, P.sha256_file(path)
    P.log_event(S, 'FREEZE', sha256=S['frozen_sha256'][:16], w_star=sel['w_star'], proceed=sel['proceed'])


def stage_test(S):
    if not S.get('frozen_path') or P.sha256_file(S['frozen_path']) != S['frozen_sha256']:
        raise P.CheckFailed('frozen_config_zeroshot.json is missing or was modified: run the freeze cell first')
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
    model, tok = S['llm']
    lp, info = llm_logprobs(S, model, tok, 'test')
    S['llm_info']['test'] = info
    P.write_json(os.path.join(S['out'], 'llm_inference.json'), S['llm_info'])
    log_q, log_prior = np.asarray(F_['log_q_dev']), np.asarray(F_['log_prior_train'])
    cal = prior_match(lp, log_q, log_prior)
    y, dlg = E[0]['y'], E[0]['dlg']
    for c, d in zip(S['ckpts'], E):
        np.savez_compressed(os.path.join(S['out'], 'dumps', f'test_ecerc_{c["name"]}.npz'), logp=d['logp'].astype(np.float32),
                            y=d['y'], dlg=d['dlg'])
    np.savez_compressed(os.path.join(S['out'], 'dumps', 'test_llm.npz'), logp=lp.astype(np.float32),
                        calibrated=cal.astype(np.float32), y=y)
    n_te = len(S['keys']['test'])
    test_cms = LE.pair_cms(E, [dict(logp=cal, y=y, dlg=dlg)], n_te)
    K = test_cms.shape[1]
    r_star = W_GRID.index(F_['w_star'])
    sc = P.scores(test_cms.sum(0).reshape(K, len(W_GRID), C, C))

    def arm(r):
        return {k: 100 * sc[k][:, r].mean(0) for k in ('acc', 'mf1', 'wf1', 'f1', 'prec', 'rec')}

    base, fused, alone = arm(0), arm(r_star), arm(len(W_GRID) - 1)
    per_pair = 100 * (sc['mf1'][:, r_star] - sc['mf1'][:, 0])
    bt = LE.procedure_bootstrap(S['dev_cms'], test_cms, F_['bootstrap']['B'], F_['bootstrap']['seed'])
    ci = lambda x, q=(2.5, 97.5): np.percentile(x, q, axis=0)  # noqa: E731
    lo, hi = ci(bt['mf1'])
    alpha = 0.05 / F_['multiplicity']['test_looks']
    b_lo, b_hi = ci(bt['mf1'], (100 * alpha / 2, 100 * (1 - alpha / 2)))
    d = {k: fused[k] - base[k] for k in fused}
    verdict, cond = LE.decide(float(d['mf1']), lo, hi, d['f1'], float(d['acc']), float(d['wf1']),
                              float((per_pair > 0).mean()))
    # secondary: fusion without calibration at its own dev-chosen w, test-only bootstrap at that fixed rule
    test_u = LE.pair_cms(E, [dict(logp=lp, y=y, dlg=dlg)], n_te)
    r_u = W_GRID.index(F_['secondary_uncalibrated_w'])
    flat = test_u[:, :, [0, r_u], :].reshape(n_te, -1).astype(float)
    rng, b2, done = np.random.default_rng(F_['bootstrap']['seed'] + 1), [], 0
    while done < F_['bootstrap']['B']:
        nb = min(1000, F_['bootstrap']['B'] - done)
        s2 = P.scores((P._boot_weights(rng, n_te, nb) @ flat).reshape(nb, K, 2, C, C))['mf1']
        b2.append(100 * (s2[:, :, 1] - s2[:, :, 0]).mean(1))
        done += nb
    su = P.scores(test_u.sum(0).reshape(K, len(W_GRID), C, C))
    S['results'] = dict(
        verdict=verdict, conditions=cond, w_star=F_['w_star'],
        delta_macro_f1=float(d['mf1']), ci95=[lo, hi], per_pair=per_pair, pairs_positive=int((per_pair > 0).sum()),
        ci_kind='procedure-level two-sample dialogue bootstrap (w re-selected per replicate)',
        multiplicity=dict(test_looks=F_['multiplicity']['test_looks'], level=100 * (1 - alpha), ci=[b_lo, b_hi],
                          lower_bound_above_0=bool(b_lo > 0)),
        per_class=dict(delta_f1=d['f1'], delta_f1_ci95=ci(bt['f1']).T, delta_precision=d['prec'], delta_recall=d['rec']),
        guardrails=dict(delta_acc=float(d['acc']), delta_acc_ci95=ci(bt['acc']), delta_weighted_f1=float(d['wf1']),
                        delta_weighted_f1_ci95=ci(bt['wf1'])),
        arms={'ECERC alone (w=0)': base, f'fused (w={F_["w_star"]})': fused, 'LLM alone, calibrated (w=1)': alone},
        llm_alone_raw=_metrics(y, lp),
        secondary_uncalibrated=dict(w=F_['secondary_uncalibrated_w'],
                                    delta_macro_f1=100 * float((su['mf1'][:, r_u] - su['mf1'][:, 0]).mean()),
                                    test_only_ci95=ci(np.concatenate(b2))),
        support=sc['support'][0, 0], bootstrap_w_selected=dict(zip(*np.unique(bt['w_sel'], return_counts=True))))
    P.write_json(os.path.join(S['out'], 'test_results_zeroshot.json'), S['results'])
    _write_summary(S, F_)
    P.log_event(S, 'TEST done', verdict=verdict, delta=round(float(d['mf1']), 2),
                ci=[round(float(lo), 2), round(float(hi), 2)])


def _write_summary(S, F_):
    sel, env = S['selection'], S['env']
    fc, a = sel['forecast'], sel['llm_alone_dev']
    L = ['# ECERC Limitation 2, Experiment 7: a zero-shot LLM fused with ECERC', '',
         f'Run {time.strftime("%Y-%m-%d %H:%M")} | {env["device"]} {env["gpus"]} | torch {env["torch"]} | '
         f'transformers {env["transformers"]} | LLM {S["cfg"]["llm_model"]} | data sha256 {S["data_sha256"][:16]}...', '',
         '## Dev (selection only)', '',
         '- ECERC alone, dev macro-F1: ' + ', '.join(f'{e["name"]} {e["macro_f1"]:.2f}' for e in sel['ecerc_alone_dev']),
         '- LLM alone, dev macro-F1: raw ' + f'{a["raw"]["macro_f1"]:.2f}, calibrated {a["calibrated"]["macro_f1"]:.2f} '
         '(per class, calibrated: ' + ', '.join(f'{c} {f:.1f}' for c, f in zip(CLASSES, a['calibrated']['f1'])) + ')',
         f'- w* = {sel["w_star"]}; dev macro-F1 {sel["dev_mean_macro_f1_by_w"][str(sel["w_star"])]:.2f} vs '
         f'{sel["dev_mean_macro_f1_by_w"]["0.0"]:.2f} for ECERC alone',
         f'- cross-fitted dev gain **{fc["crossfit_delta_macro_f1"]:+.2f}** (gate {S["cfg"]["gate_min_gain"]:+.2f}: '
         + ('passed' if sel['proceed'] else 'not passed, so the test was skipped') + f'); projected test SE {fc["se_test_projected"]:.2f}',
         f'- secondary, without calibration: cross-fitted dev gain {sel["secondary_uncalibrated"]["crossfit_delta_macro_f1"]:+.2f}',
         '- Fear/Disgust/Sadness on dev, mean per checkpoint: ' + '; '.join(
             f'{k} ECERC wrong but LLM right {v["ecerc_wrong_llm_right"]:.1f} / reverse {v["ecerc_right_llm_wrong"]:.1f} '
             f'(n={v["n"]})' for k, v in sel['diversity_dev'].items()), '']
    res = S.get('results')
    if res:
        L += [f'## Verdict: {res["verdict"]}', '', '| condition | met |', '|---|---|']
        L += [f'| {k} | {"yes" if v else "no"} |' for k, v in res['conditions'].items()]
        L += ['', '## Test (mean over the three released checkpoints)', '',
              '| arm | acc | weighted F1 | macro-F1 | ' + ' | '.join(f'F1 {c}' for c in CLASSES) + ' |', '|---' * (4 + C) + '|']
        for name, m in res['arms'].items():
            L.append(f'| {name} | {m["acc"]:.2f} | {m["wf1"]:.2f} | {m["mf1"]:.2f} | ' + ' | '.join(f'{x:.2f}' for x in m['f1']) + ' |')
        g, pc, mu = res['guardrails'], res['per_class'], res['multiplicity']
        L += ['', f'- Delta macro-F1 (fused - ECERC alone): **{P._fmt(res["delta_macro_f1"], *res["ci95"])}** ({res["ci_kind"]})',
              f'- third-look adjustment: {mu["level"]:.2f}% interval [{mu["ci"][0]:+.2f}, {mu["ci"][1]:+.2f}], lower bound '
              f'above 0: {"yes" if mu["lower_bound_above_0"] else "no"}',
              f'- positive for {res["pairs_positive"]}/{len(res["per_pair"])} checkpoints: '
              + ', '.join(f'{x:+.2f}' for x in res['per_pair']),
              f'- Delta accuracy {P._fmt(g["delta_acc"], *g["delta_acc_ci95"])}; Delta weighted F1 '
              f'{P._fmt(g["delta_weighted_f1"], *g["delta_weighted_f1_ci95"])}', '',
              '| class | Delta F1 [95% CI] | Delta precision | Delta recall | support |', '|---|---|---|---|---|']
        for c in range(C):
            L.append(f'| {CLASSES[c]} | {P._fmt(pc["delta_f1"][c], *pc["delta_f1_ci95"][c])} | {pc["delta_precision"][c]:+.2f} | '
                     f'{pc["delta_recall"][c]:+.2f} | {int(res["support"][c])} |')
        s = res['secondary_uncalibrated']
        L += ['', f'- secondary, fusion without calibration (w = {s["w"]}): {P._fmt(s["delta_macro_f1"], *s["test_only_ci95"])} '
                  '(test-only CI)',
              f'- LLM alone, raw: macro-F1 {res["llm_alone_raw"]["macro_f1"]:.2f}, weighted F1 {res["llm_alone_raw"]["weighted_f1"]:.2f}']
    warn = [c for c in S['checks'] if not c['passed']]
    L += ['', f'Checks: {sum(c["passed"] for c in S["checks"])} passed, {len(warn)} warnings'
          + ('' if not warn else ' -- ' + '; '.join(f'{c["name"]} ({c["detail"]})' for c in warn))]
    with open(os.path.join(S['out'], 'SUMMARY_zeroshot.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')


def stage_package(S):
    zpath = os.path.join(S['cfg']['work_dir'], 'ecerc_l2_zeroshot_results.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(S['out']):
            for name in files:
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, os.path.dirname(S['out'])))
    with open(os.path.join(S['out'], 'SUMMARY_zeroshot.md')) as f:
        print(f.read())
    print(f'Results archive: {zpath} ({os.path.getsize(zpath) / 2 ** 20:.1f} MB)')
    return zpath
