"""
ECERC Limitation 2, Experiment 5: a fine-tuned LLM expert (Qwen2.5-7B-Instruct, QLoRA) fused with ECERC.

New code for research/limitation2. Same design as Experiment 4, with a much stronger expert:
  - the expert is an instruction-tuned 7B LLM with a sequence-classification head, fine-tuned with LoRA on a
    4-bit quantised base (QLoRA). Input: an instruction, the preceding turns ("Speaker k: text") and the
    target utterance; past context only, ECERC's task setting;
  - seeds train in parallel worker processes, one per GPU (l2_llm_worker.py);
  - fusion score = (1 - w) log p_ECERC + w log p_LLM with w chosen on dev; pairs = released ECERC
    checkpoints x LLM seeds; the test runs once, only if the cross-fitted dev gain >= CONFIG['gate_min_gain'],
    after every decision is frozen in frozen_config_llm.json.
This is the third evaluation on the MELD test set (after experiments 1 and 4). Besides the pre-registered 95%
interval, a Bonferroni-adjusted interval for three looks (98.33%) is reported.
"""
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

import l2_expert as LE  # fusion, selection, cross-fitting, bootstrap and verdict helpers (unchanged)
import l2_posthoc as P  # downloads, data gate, ECERC builders, inference and metrics (unchanged)

C, CLASSES = P.C, P.CLASSES
FEA, DIS, SAD = 2, 5, 3
LOOKS = 3  # test-set evaluations so far, counting this one
INSTRUCTION = ('Task: identify the emotion that the target speaker expresses in the last utterance of this conversation '
               'from a TV show. Answer with one of: neutral, surprise, fear, sadness, joy, disgust, anger.\n\nConversation:\n')


def _cm(y, p):
    return np.bincount(y * C + p, minlength=C * C).reshape(C, C)


# ----------------------------------------------------------------------------- setup and inputs
def stage_setup(CONFIG, code_dir):
    out = os.path.join(CONFIG['work_dir'], 'outputs_llm')
    for sub in ('', 'tables', 'dumps', 'logs'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    os.makedirs(os.path.join(CONFIG['work_dir'], 'llm_runs'), exist_ok=True)
    if not CONFIG.get('offline'):
        need = [p for p in ('accelerate',) if importlib.util.find_spec(p) is None]
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-U', 'bitsandbytes', 'peft'] + need, check=False)
    import peft
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
    try:
        import bitsandbytes
        bnb = bitsandbytes.__version__
    except Exception as e:  # not needed for the CPU smoke test
        bnb = f'unavailable ({type(e).__name__})'
    S['env'] = dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                    transformers=transformers.__version__, peft=peft.__version__, bitsandbytes=bnb,
                    device=S['device'], gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    code_sha256={f: P.sha256_file(os.path.join(code_dir, f))
                                 for f in sorted(os.listdir(code_dir)) if f.endswith('.py')})
    P.write_json(os.path.join(out, 'environment.json'), S['env'])
    P.write_json(os.path.join(out, 'design.json'),  # written before any data is loaded
                 dict({k: v for k, v in CONFIG.items() if k.startswith('llm_') or k in ('gate_min_gain', 'n_folds')},
                      instruction=INSTRUCTION, fusion='score = (1 - w) log p_ECERC + w log p_LLM', w_grid=LE.W_GRID,
                      selection='highest mean dev macro-F1 over pairs; ties to the smaller w',
                      gate=f'cross-fitted dev gain >= {CONFIG["gate_min_gain"]}', criteria=LE.CRITERIA,
                      multiplicity=f'test look {LOOKS} of {LOOKS}; Bonferroni-adjusted {100 * (1 - 0.05 / LOOKS):.2f}% CI also reported'))
    if not cuda:
        print('WARNING: no GPU. Fine-tuning a 7B model needs a GPU; switch the accelerator to GPU T4 x2.')
    P.log_event(S, 'setup', device=S['device'], gpus=len(S['env']['gpus']), torch=torch.__version__,
                transformers=transformers.__version__, bitsandbytes=bnb)
    return S


def build_inputs(S, split):
    """Token ids per utterance: instruction + as many preceding turns as fit (most recent kept) + the target."""
    tok, cfg, D = S['tok'], S['cfg'], S['D']
    head = tok(INSTRUCTION, add_special_tokens=False)['input_ids']
    seqs = []
    for vid in S['keys'][split]:
        sents = D['sentences'][vid]
        spk = np.asarray(D['speakers'][vid]).argmax(-1)
        turns = [tok(f'Speaker {int(spk[i]) + 1}: {LE._clean(sents[i])}\n', add_special_tokens=False)['input_ids']
                 for i in range(len(sents))]
        for t in range(len(sents)):
            tail = tok(f'\nTarget utterance (Speaker {int(spk[t]) + 1}): {LE._clean(sents[t])}\nEmotion:',
                       add_special_tokens=False)['input_ids']
            budget, ctx = cfg['llm_max_len'] - len(head) - len(tail), []
            for j in range(t - 1, -1, -1):
                if len(turns[j]) > budget:
                    break
                ctx = turns[j] + ctx
                budget -= len(turns[j])
            seqs.append((head + ctx + tail)[-cfg['llm_max_len']:])
    return seqs


def stage_prepare(S):
    from transformers import AutoTokenizer
    cfg = S['cfg']
    P.stage_download(S)
    P.stage_load_data(S)  # reads train/dev labels only
    D = S['D']
    ok = all(len(D['sentences'][v]) == len(D['videoAudio'][v]) and all(isinstance(x, str) for x in D['sentences'][v])
             for s in ('train', 'valid', 'test') for v in S['keys'][s])
    P.record_check(S, 'utterance text is present for every utterance (label-free check)', ok, hard=True)
    model_path = cfg['llm_model']
    if not os.path.isdir(model_path):
        from huggingface_hub import snapshot_download
        t0 = time.time()
        model_path = snapshot_download(model_path, allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model',
                                                                   'tokenizer*', 'merges.txt', 'vocab.json'])
        P.log_event(S, 'LLM downloaded', minutes=round((time.time() - t0) / 60, 1))
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    S['tok'], S['llm_path'] = tok, model_path
    enc = {s: build_inputs(S, s) for s in ('train', 'valid')}
    ys = {s: np.concatenate([np.asarray(D['emotion_labels'][v], dtype=np.int64) for v in S['keys'][s]]) for s in enc}
    lens = [len(x) for x in enc['train']]
    print('  example input:\n' + tok.decode(enc['train'][min(5, len(lens) - 1)]))
    P.record_check(S, 'inputs fit the length budget', max(lens) <= cfg['llm_max_len'], hard=True,
                   detail=f'max {max(lens)} tokens, mean {np.mean(lens):.0f}, budget {cfg["llm_max_len"]}')
    inp_dir = os.path.join(cfg['work_dir'], 'llm_inputs')
    os.makedirs(inp_dir, exist_ok=True)
    S['trainval_path'] = os.path.join(inp_dir, 'trainval.json')
    with open(S['trainval_path'], 'w') as f:
        json.dump({s: dict(ids=enc[s], y=ys[s].tolist()) for s in enc}, f)
    S['dev_y'] = ys['valid']
    run_cfg = dict(model=model_path, pad_id=int(tok.pad_token_id), quantize=bool(cfg['llm_quantize']),
                   lora_r=cfg['llm_lora_r'], lora_alpha=cfg['llm_lora_alpha'], lora_dropout=cfg['llm_lora_dropout'],
                   lora_targets=cfg['llm_lora_targets'], lr=cfg['llm_lr'], epochs=cfg['llm_epochs'],
                   micro_batch=cfg['llm_micro_batch'], grad_accum=cfg['llm_grad_accum'], warmup=cfg['llm_warmup'],
                   eval_every=cfg['llm_eval_every'], patience=cfg['llm_patience'], eval_batch=cfg['llm_eval_batch'],
                   max_minutes=cfg.get('llm_max_minutes', 1e9),
                   trainval_path=S['trainval_path'])
    S['run_cfg_path'] = os.path.join(inp_dir, 'run_cfg.json')
    P.write_json(S['run_cfg_path'], run_cfg)
    P._save_checks(S)
    P.log_event(S, 'inputs ready', train=len(enc['train']), dev=len(enc['valid']), mean_tokens=round(float(np.mean(lens))))


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
    assert np.array_equal(S['ecerc_dev'][0]['y'], S['dev_y']), 'ECERC and LLM dev items are not aligned'
    P._save_checks(S)


# ----------------------------------------------------------------------------- workers
def run_workers(S, jobs):
    """Run worker jobs, one per GPU at a time (CPU: one at a time), streaming their progress lines."""
    worker = os.path.join(S['code_dir'], 'l2_llm_worker.py')
    n_gpu = torch.cuda.device_count()
    slots = list(range(n_gpu)) if n_gpu else [None]
    queue, running, done = list(jobs), {}, []
    while queue or running:
        for slot in slots:
            if slot not in running and queue:
                job = queue.pop(0)
                env = dict(os.environ, TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
                if slot is not None:
                    env['CUDA_VISIBLE_DEVICES'] = str(slot)
                fh = open(job['log'], 'wb')
                proc = subprocess.Popen([sys.executable, '-u', worker] + job['args'], stdout=fh,
                                        stderr=subprocess.STDOUT, env=env)
                running[slot] = dict(proc=proc, job=job, fh=fh, pos=0)
                print(f'  started {job["name"]} on ' + (f'GPU {slot}' if slot is not None else 'CPU'), flush=True)
                if queue and len(running) < len(slots):
                    time.sleep(S['cfg'].get('stagger_seconds', 0))  # don't load two large models at the same moment
        time.sleep(S['cfg'].get('poll_seconds', 30))
        for slot, r in list(running.items()):
            with open(r['job']['log'], 'rb') as g:
                g.seek(r['pos'])
                new = g.read()
            r['pos'] += len(new)
            for line in new.decode('utf-8', 'replace').splitlines():
                if line.startswith('[seed') or line.startswith('predicted') or 'Error' in line or 'Traceback' in line:
                    print(f'    {r["job"]["name"]} | {line}', flush=True)
            if r['proc'].poll() is not None:
                r['fh'].close()
                done.append((r['job'], r['proc'].returncode))
                del running[slot]
    failed = [(j, rc) for j, rc in done if rc != 0]
    if failed:
        tails = []
        for j, rc in failed:
            with open(j['log'], 'rb') as g:
                tails.append(f'{j["name"]} (exit {rc}):\n' + g.read().decode('utf-8', 'replace')[-3000:])
        raise P.CheckFailed('worker failed:\n' + '\n'.join(tails))


def _train_jobs(S, seeds):
    jobs = []
    for sd in seeds:
        d = os.path.join(S['cfg']['work_dir'], 'llm_runs', f'seed{sd}')
        os.makedirs(d, exist_ok=True)
        jobs.append(dict(name=f'LLM seed {sd}', log=os.path.join(S['out'], 'logs', f'train_seed{sd}.log'), seed=sd, dir=d,
                         args=['train', '--cfg', S['run_cfg_path'], '--seed', str(sd), '--out', d]))
    return jobs


def _load_run(d):
    with open(os.path.join(d, 'result.json')) as f:
        res = json.load(f)
    dev = np.load(os.path.join(d, 'dev_logp.npz'))
    return res, dev['logp'].astype(np.float64)


def stage_train_llm(S):
    cfg, t0 = S['cfg'], time.time()
    jobs = _train_jobs(S, cfg['llm_seeds'])
    run_workers(S, jobs)
    results, retry = {}, []
    for j in jobs:
        res, lp = _load_run(j['dir'])
        results[j['seed']] = (j, res, lp)
        if res['best_dev_macro_f1'] < cfg['llm_min_dev_mf1']:  # pre-registered: one retry for a collapsed run
            retry.append(j['seed'])
    if retry:
        print(f'  collapsed seeds {retry} (best dev macro-F1 < {cfg["llm_min_dev_mf1"]}); one retry each with seed + 1000')
        rjobs = _train_jobs(S, [sd + 1000 for sd in retry])
        run_workers(S, rjobs)
        for sd, j in zip(retry, rjobs):
            res, lp = _load_run(j['dir'])
            results[sd] = (j, dict(res, retried_after=results[sd][1]['best_dev_macro_f1']), lp)
    S['llms'] = []
    for sd in cfg['llm_seeds']:
        j, res, lp = results[sd]
        S['llms'].append(dict(seed=res['seed'], dir=j['dir'], best_dev_macro_f1=res['best_dev_macro_f1'], curve=res['curve'],
                              minutes=res['minutes'], retried_after=res.get('retried_after'),
                              dev=dict(logp=lp, y=S['dev_y'], dlg=S['ecerc_dev'][0]['dlg'])))
    P.write_json(os.path.join(S['out'], 'llm_training.json'), [{k: v for k, v in m.items() if k != 'dev'} for m in S['llms']])
    P.log_event(S, 'LLM experts trained', minutes=round((time.time() - t0) / 60, 1),
                best_dev_macro_f1=[round(m['best_dev_macro_f1'], 2) for m in S['llms']])


# ----------------------------------------------------------------------------- analysis, freeze, test
def stage_dev_analysis(S):
    cfg = S['cfg']
    E, T = S['ecerc_dev'], [m['dev'] for m in S['llms']]
    n_dev = len(S['keys']['valid'])
    S['dev_cms'] = LE.pair_cms(E, T, n_dev)
    r_star, mean = LE._select(S['dev_cms'].sum(0))
    fc = LE.crossfit_forecast(S['dev_cms'], cfg['n_folds'], cfg['fold_seed'], cfg['power_B'], len(S['keys']['test']))
    r2, _ = LE._select(LE.pair_cms(E, T, n_dev, LE.TAU_GRID, S['log_prior']).sum(0))
    w2, t2 = LE.W_GRID[r2 // len(LE.TAU_GRID)], LE.TAU_GRID[r2 % len(LE.TAU_GRID)]
    proceed = fc['crossfit_delta_macro_f1'] >= cfg['gate_min_gain']
    alone = [dict(seed=m['seed'], **{k: 100 * v for k, v in P.scores(_cm(S['dev_y'], m['dev']['logp'].argmax(1))).items()
                                     if k in ('mf1', 'wf1', 'acc', 'f1')}) for m in S['llms']]
    ecerc = [dict(name=c['name'], **{k: 100 * v for k, v in P.scores(_cm(d['y'], d['logp'].argmax(1))).items()
                                     if k in ('mf1', 'wf1', 'acc', 'f1')}) for c, d in zip(S['ckpts'], E)]
    div = {}
    for c in (FEA, DIS, SAD):
        rows = []
        for e in E:
            for t in T:
                m = e['y'] == c
                ew, tw = e['logp'].argmax(1)[m] != c, t['logp'].argmax(1)[m] != c
                rows.append([int((ew & ~tw).sum()), int((~ew & tw).sum()), int(m.sum())])
        r = np.mean(rows, 0)
        div[CLASSES[c]] = dict(ecerc_wrong_llm_right=r[0], ecerc_right_llm_wrong=r[1], n=r[2])
    S['selection'] = dict(w_star=LE.W_GRID[r_star], dev_mean_macro_f1_by_w={str(w): 100 * m for w, m in zip(LE.W_GRID, mean)},
                          forecast=fc, proceed=proceed, secondary_w_tau=dict(w=w2, tau=t2),
                          llm_alone_dev=alone, ecerc_alone_dev=ecerc, diversity_dev=div)
    P.write_json(os.path.join(S['out'], 'dev_selection.json'), S['selection'])
    print('\nDEV-ONLY RESULTS (frozen before the test set is touched)')
    print('  ECERC alone, dev macro-F1:', ', '.join(f'{e["name"]} {e["mf1"]:.2f}' for e in ecerc))
    print('  LLM alone, dev macro-F1:', ', '.join(f'seed {a["seed"]} {a["mf1"]:.2f} (weighted F1 {a["wf1"]:.2f})' for a in alone))
    print(f'  w* = {LE.W_GRID[r_star]}: mean dev macro-F1 {100 * mean[r_star]:.2f} vs {100 * mean[0]:.2f} for ECERC alone')
    print(f'  cross-fitted dev gain {fc["crossfit_delta_macro_f1"]:+.2f} (fold w {fc["fold_w"]}); projected test SE '
          f'{fc["se_test_projected"]:.2f}; gate {cfg["gate_min_gain"]:+.2f} -> ' + ('TEST WILL RUN' if proceed else 'test skipped'))
    P.log_event(S, 'dev analysis done', w_star=LE.W_GRID[r_star], forecast=round(fc['crossfit_delta_macro_f1'], 2),
                proceed=proceed)


def stage_freeze(S):
    sel = S['selection']
    frozen = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'),
                  note='Written before any test-set access. The test evaluation reads its rules from this file.',
                  code_sha256=S['env']['code_sha256'], data_sha256=S['data_sha256'], llm_model=S['cfg']['llm_model'],
                  ecerc_checkpoints=[dict(name=c['name'], sha256=c['sha256'], archive=c['archive']) for c in S['ckpts']],
                  llms=[dict(seed=m['seed'], best_dev_macro_f1=m['best_dev_macro_f1'], retried_after=m['retried_after'])
                        for m in S['llms']],
                  w_grid=LE.W_GRID, w_star=sel['w_star'], secondary_w_tau=sel['secondary_w_tau'],
                  log_prior_train=S['log_prior'],
                  gate=dict(statistic='cross-fitted dev gain', value=sel['forecast']['crossfit_delta_macro_f1'],
                            threshold=S['cfg']['gate_min_gain'], proceed=sel['proceed']),
                  bootstrap=dict(B=S['cfg']['bootstrap_B'], seed=S['cfg']['boot_seed'], unit='dialogue',
                                 design='dev and test resampled independently; w re-selected in every replicate'),
                  multiplicity=dict(test_looks=LOOKS, adjusted_level=1 - 0.05 / LOOKS),
                  criteria=LE.CRITERIA, device=S['device'], env=S['env'])
    path = os.path.join(S['out'], 'frozen_config_llm.json')
    P.write_json(path, frozen)
    S['frozen_path'], S['frozen_sha256'] = path, P.sha256_file(path)
    P.log_event(S, 'FREEZE', sha256=S['frozen_sha256'][:16], w_star=sel['w_star'], proceed=sel['proceed'])


def stage_test(S):
    if not S.get('frozen_path') or P.sha256_file(S['frozen_path']) != S['frozen_sha256']:
        raise P.CheckFailed('frozen_config_llm.json is missing or was modified: run the freeze cell first')
    with open(S['frozen_path']) as f:
        F_ = json.load(f)
    if any(P.sha256_file(os.path.join(S['code_dir'], n)) != h for n, h in F_['code_sha256'].items()):
        raise P.CheckFailed('code changed after the freeze')
    if not F_['gate']['proceed']:
        print('Test skipped by the pre-registered gate: the cross-fitted dev gain is below the threshold.')
        S['results'] = None
        _write_summary(S)
        return
    P.log_event(S, 'TEST evaluation starts (test dialogues are run and test labels read only from here on)')
    test_ds = P.make_dataset(S, 'test')
    E = [P.run_inference(S, m, test_ds) for m in S['ecerc_models']]
    test_path = os.path.join(S['cfg']['work_dir'], 'llm_inputs', 'test_inputs.json')
    with open(test_path, 'w') as f:
        json.dump(dict(ids=build_inputs(S, 'test')), f)  # inputs only, no labels
    jobs = [dict(name=f'LLM seed {m["seed"]} test', log=os.path.join(S['out'], 'logs', f'test_seed{m["seed"]}.log'),
                 args=['predict', '--cfg', S['run_cfg_path'], '--adapter', os.path.join(m['dir'], 'adapter'),
                       '--inputs', test_path, '--out', os.path.join(m['dir'], 'test_logp.npz')]) for m in S['llms']]
    run_workers(S, jobs)
    T = [dict(logp=np.load(os.path.join(m['dir'], 'test_logp.npz'))['logp'].astype(np.float64), y=E[0]['y'], dlg=E[0]['dlg'])
         for m in S['llms']]
    for c, d in zip(S['ckpts'], E):
        np.savez_compressed(os.path.join(S['out'], 'dumps', f'test_ecerc_{c["name"]}.npz'), logp=d['logp'].astype(np.float32),
                            y=d['y'], dlg=d['dlg'])
    for m, d in zip(S['llms'], T):
        np.savez_compressed(os.path.join(S['out'], 'dumps', f'test_llm_seed{m["seed"]}.npz'), logp=d['logp'].astype(np.float32))
    n_te = len(S['keys']['test'])
    test_cms = LE.pair_cms(E, T, n_te)
    K = test_cms.shape[1]
    r_star = LE.W_GRID.index(F_['w_star'])
    sc = P.scores(test_cms.sum(0).reshape(K, len(LE.W_GRID), C, C))

    def arm(r):
        return {k: 100 * sc[k][:, r].mean(0) for k in ('acc', 'mf1', 'wf1', 'f1', 'prec', 'rec')}

    base, fused, alone = arm(0), arm(r_star), arm(len(LE.W_GRID) - 1)
    per_pair = 100 * (sc['mf1'][:, r_star] - sc['mf1'][:, 0])
    bt = LE.procedure_bootstrap(S['dev_cms'], test_cms, F_['bootstrap']['B'], F_['bootstrap']['seed'])
    adj = 100 * (0.05 / LOOKS) / 2
    ci = lambda x, a=2.5: np.percentile(x, [a, 100 - a], axis=0)  # noqa: E731
    lo, hi = ci(bt['mf1'])
    alo, ahi = ci(bt['mf1'], adj)
    d = {k: fused[k] - base[k] for k in fused}
    verdict, cond = LE.decide(float(d['mf1']), lo, hi, d['f1'], float(d['acc']), float(d['wf1']),
                              float((per_pair > 0).mean()))
    cond['multiplicity_adjusted_ci_lower_gt_0'] = bool(alo > 0)
    test_cms_tau = LE.pair_cms(E, T, n_te, LE.TAU_GRID, np.asarray(F_['log_prior_train']))
    r2 = LE.W_GRID.index(F_['secondary_w_tau']['w']) * len(LE.TAU_GRID) + LE.TAU_GRID.index(F_['secondary_w_tau']['tau'])
    sc2 = P.scores(test_cms_tau.sum(0).reshape(K, -1, C, C))
    sec = float(100 * (sc2['mf1'][:, r2] - sc2['mf1'][:, 0]).mean())
    eE, eT, y = np.mean([x['logp'] for x in E], 0), np.mean([x['logp'] for x in T], 0), E[0]['y']
    ens = {name: 100 * float(P.scores(_cm(y, s.argmax(1)))['mf1']) for name, s in
           (('ECERC x3 ensemble', eE), ('LLM ensemble', eT), ('fused ensembles at w*', (1 - F_['w_star']) * eE + F_['w_star'] * eT))}
    S['results'] = dict(
        verdict=verdict, conditions=cond, w_star=F_['w_star'], delta_macro_f1=float(d['mf1']), ci95=[lo, hi],
        ci_multiplicity_adjusted=dict(level=1 - 0.05 / LOOKS, interval=[alo, ahi]),
        per_pair=per_pair, pairs_positive=int((per_pair > 0).sum()),
        ci_kind='procedure-level two-sample dialogue bootstrap (w re-selected per replicate)',
        per_class=dict(delta_f1=d['f1'], delta_f1_ci95=ci(bt['f1']).T, delta_precision=d['prec'], delta_recall=d['rec']),
        guardrails=dict(delta_acc=float(d['acc']), delta_acc_ci95=ci(bt['acc']), delta_weighted_f1=float(d['wf1']),
                        delta_weighted_f1_ci95=ci(bt['wf1'])),
        arms={'ECERC alone (w=0)': base, f'fused (w={F_["w_star"]})': fused, 'LLM alone (w=1)': alone},
        secondary=dict(w_tau=F_['secondary_w_tau'], delta_macro_f1=sec), ensembles_descriptive=ens,
        support=sc['support'][0, 0], bootstrap_w_selected=dict(zip(*np.unique(bt['w_sel'], return_counts=True))))
    P.write_json(os.path.join(S['out'], 'test_results_llm.json'), S['results'])
    _write_summary(S)
    P.log_event(S, 'TEST done', verdict=verdict, delta=round(float(d['mf1']), 2), ci=[round(lo, 2), round(hi, 2)],
                adjusted_ci=[round(alo, 2), round(ahi, 2)])


def _write_summary(S):
    sel, env = S['selection'], S['env']
    fc = sel['forecast']
    L = ['# ECERC Limitation 2, Experiment 5: fine-tuned LLM expert fused with ECERC', '',
         f'Run {time.strftime("%Y-%m-%d %H:%M")} | {env["device"]} {env["gpus"]} | torch {env["torch"]} | transformers '
         f'{env["transformers"]} | peft {env["peft"]} | bitsandbytes {env["bitsandbytes"]}', '',
         '## Dev (selection only)', '',
         '- ECERC alone, dev macro-F1: ' + ', '.join(f'{e["name"]} {e["mf1"]:.2f}' for e in sel['ecerc_alone_dev']),
         '- LLM alone, dev macro-F1: ' + ', '.join(f'seed {a["seed"]} {a["mf1"]:.2f} (weighted F1 {a["wf1"]:.2f})'
                                                 for a in sel['llm_alone_dev']),
         f'- w* = {sel["w_star"]}; dev macro-F1 {sel["dev_mean_macro_f1_by_w"][str(sel["w_star"])]:.2f} vs '
         f'{sel["dev_mean_macro_f1_by_w"]["0.0"]:.2f} for ECERC alone',
         f'- cross-fitted dev gain **{fc["crossfit_delta_macro_f1"]:+.2f}** (gate {S["cfg"]["gate_min_gain"]:+.2f}: '
         + ('passed' if sel['proceed'] else 'not passed, so the test was skipped') + f'); projected test SE {fc["se_test_projected"]:.2f}',
         '- dev, mean per pair: ' + '; '.join(f'{k} ECERC wrong but LLM right {v["ecerc_wrong_llm_right"]:.1f} / reverse '
                                             f'{v["ecerc_right_llm_wrong"]:.1f} (n={v["n"]:.0f})' for k, v in sel['diversity_dev'].items()), '']
    res = S.get('results')
    if res:
        a = res['ci_multiplicity_adjusted']
        L += [f'## Verdict: {res["verdict"]}', '', '| condition | met |', '|---|---|']
        L += [f'| {k} | {"yes" if v else "no"} |' for k, v in res['conditions'].items()]
        L += ['', '## Test (mean over the checkpoint x seed pairs)', '',
              '| arm | acc | weighted F1 | macro-F1 | ' + ' | '.join(f'F1 {c}' for c in CLASSES) + ' |', '|---' * (4 + C) + '|']
        for name, m in res['arms'].items():
            L.append(f'| {name} | {m["acc"]:.2f} | {m["wf1"]:.2f} | {m["mf1"]:.2f} | ' + ' | '.join(f'{x:.2f}' for x in m['f1']) + ' |')
        g, pc = res['guardrails'], res['per_class']
        L += ['', f'- Delta macro-F1 (fused - ECERC alone): **{P._fmt(res["delta_macro_f1"], *res["ci95"])}** ({res["ci_kind"]}); '
                  f'multiplicity-adjusted {100 * a["level"]:.2f}% CI [{a["interval"][0]:+.2f}, {a["interval"][1]:+.2f}]',
              f'- positive in {res["pairs_positive"]}/{len(res["per_pair"])} pairs: ' + ', '.join(f'{x:+.2f}' for x in res['per_pair']),
              f'- Delta accuracy {P._fmt(g["delta_acc"], *g["delta_acc_ci95"])}; Delta weighted F1 '
              f'{P._fmt(g["delta_weighted_f1"], *g["delta_weighted_f1_ci95"])}', '',
              '| class | Delta F1 [95% CI] | Delta precision | Delta recall | support |', '|---|---|---|---|---|']
        for c in range(C):
            L.append(f'| {CLASSES[c]} | {P._fmt(pc["delta_f1"][c], *pc["delta_f1_ci95"][c])} | {pc["delta_precision"][c]:+.2f} | '
                     f'{pc["delta_recall"][c]:+.2f} | {int(res["support"][c])} |')
        L += ['', f'- secondary, w and tau chosen together on dev ({res["secondary"]["w_tau"]}): {res["secondary"]["delta_macro_f1"]:+.2f}',
              '- descriptive ensembles, test macro-F1: ' + ', '.join(f'{k} {v:.2f}' for k, v in res['ensembles_descriptive'].items())]
    warn = [c for c in S['checks'] if not c['passed']]
    L += ['', f'Checks: {sum(c["passed"] for c in S["checks"])} passed, {len(warn)} warnings'
          + ('' if not warn else ' -- ' + '; '.join(f'{c["name"]} ({c["detail"]})' for c in warn))]
    with open(os.path.join(S['out'], 'SUMMARY_llm.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')


def stage_package(S):
    zpath = os.path.join(S['cfg']['work_dir'], 'ecerc_l2_llm_results.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(S['out']):
            for name in files:
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, os.path.dirname(S['out'])))
    with open(os.path.join(S['out'], 'SUMMARY_llm.md')) as f:
        print(f.read())
    print(f'Results archive: {zpath} ({os.path.getsize(zpath) / 2 ** 20:.1f} MB)')
    return zpath
