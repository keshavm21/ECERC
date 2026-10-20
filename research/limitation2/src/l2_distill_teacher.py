"""
ECERC Limitation 2, Experiment 8a: zero-shot LLM soft labels for the training split.

New code for research/limitation2. Experiment 7 fused ECERC with a frozen zero-shot LLM at inference. Experiment 8
asks whether a single ECERC model trained with that LLM's soft labels keeps part of the gain with no LLM at inference.
This first part runs the LLM of Experiment 7, unchanged (model revision, prompt, label tokens, cached scoring), on the
utterances of the TRAINING dialogues and saves its log-probabilities for the distillation notebook (8b).

Label-free: the LLM reads only the text and speakers. The test split is not scored and no test label is read. Before
the training split is scored, the first dev dialogues are scored again and compared with Experiment 7's saved dev
log-probabilities, so the teacher is the same model as in Experiment 7.
"""
import importlib.util
import os
import platform
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

import l2_posthoc as P   # data gate, logging and checks (unchanged)
import l2_pathb as LB    # data download (unchanged)
import l2_zeroshot as LZ  # Experiment 7: prompt, label tokens, model loading and cached scoring (unchanged)

C, CLASSES = P.C, P.CLASSES


def stage_setup(CONFIG, code_dir):
    out = os.path.join(CONFIG['work_dir'], 'outputs_distill_teacher')
    for sub in ('', 'dumps'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    os.makedirs(os.path.join(CONFIG['cache_dir'], 'hf'), exist_ok=True)
    cuda = torch.cuda.is_available()
    if cuda:  # 4-bit loading needs bitsandbytes and accelerate (as in Experiment 7)
        need = [p for p in ('bitsandbytes', 'accelerate') if importlib.util.find_spec(p) is None]
        if need:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + need, check=False)
    import transformers
    if CONFIG.get('reference_dev_dump') == 'embedded':  # Experiment 7's dev_llm.npz, written by the code cell
        CONFIG['reference_dev_dump'] = os.path.join(code_dir, 'dev_llm.npz')
    S = dict(cfg=CONFIG, out=out, code_dir=code_dir, checks=[], torch=torch)
    S['device'] = 'cuda' if cuda else 'cpu'
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    versions = {p: __import__(p).__version__ if importlib.util.find_spec(p) else None
                for p in ('bitsandbytes', 'accelerate')}
    gpus = [(torch.cuda.get_device_name(i), round(torch.cuda.get_device_properties(i).total_memory / 2 ** 30, 1))
            for i in range(torch.cuda.device_count())] if cuda else []
    S['env'] = dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                    transformers=transformers.__version__, numpy=np.__version__, device=S['device'], gpus=gpus,
                    **versions, code_sha256={f: P.sha256_file(os.path.join(code_dir, f))
                                             for f in sorted(os.listdir(code_dir)) if f.endswith('.py')})
    P.write_json(os.path.join(out, 'environment.json'), S['env'])
    P.write_json(os.path.join(out, 'design.json'),  # written before any data is loaded
                 dict(llm=dict(model=CONFIG['llm_model'], revision=CONFIG.get('llm_revision')),
                      prompt=dict(system=LZ.SYSTEM, question=LZ.QUESTION, sha256=LZ.PROMPT_SHA256),
                      scoring="Experiment 7's, unchanged (l2_zeroshot.llm_logprobs)", split_scored='train',
                      reproducibility=dict(dev_dialogues=CONFIG['repro_dialogues'],
                                           min_argmax_agreement=CONFIG['repro_min_agreement'],
                                           reference=os.path.basename(CONFIG['reference_dev_dump'] or 'none')),
                      test='not scored; no test label is read'))
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
    """MELD features (authors' Google Drive) and the data gate; reads train/dev labels only."""
    LB.stage_download(S)
    P.stage_load_data(S)
    D = S['D']
    ok = all(len(D['sentences'][v]) == len(D['videoAudio'][v]) and all(isinstance(x, str) for x in D['sentences'][v])
             for s in ('train', 'valid') for v in S['keys'][s])
    P.record_check(S, 'utterance text is present for every train and dev utterance (label-free check)', ok, hard=True)
    P._save_checks(S)


def stage_llm(S):
    """Loads the LLM, checks it against Experiment 7 and scores the training split."""
    cfg = S['cfg']
    model, tok = LZ.load_llm(S)
    S['label_ids'], unique = LZ.label_token_ids(tok)
    spelled = {w: [tok.decode([i]) for i in g] for w, g in zip(LZ.LABELS, S['label_ids'])}
    P.record_check(S, 'label words start with distinct tokens', unique, hard=True, detail=spelled)
    if cfg.get('expected_label_token_ids') is not None:
        P.record_check(S, "label token ids equal Experiment 7's frozen ids",
                       [list(g) for g in S['label_ids']] == cfg['expected_label_token_ids'], hard=True)
    P.record_check(S, "prompt equals Experiment 7's (SHA-256)", LZ.PROMPT_SHA256 == cfg['expected_prompt_sha256'],
                   hard=True, detail=LZ.PROMPT_SHA256[:16])

    # reproducibility: the first dev dialogues again, against Experiment 7's saved dev log-probabilities
    keys = S['keys']['valid'][:cfg['repro_dialogues']]
    S['keys']['repro_dev'] = keys
    lp_dev, info_dev = LZ.llm_logprobs(S, model, tok, 'repro_dev')
    np.savez_compressed(os.path.join(S['out'], 'dumps', 'repro_dev_llm.npz'), logp=lp_dev.astype(np.float32),
                        keys=np.asarray(keys))
    ref_path = cfg.get('reference_dev_dump')
    repro = dict(dialogues=len(keys), utterances=len(lp_dev))
    if ref_path:
        ref = np.load(ref_path)['logp'][:len(lp_dev)].astype(np.float64)
        agree = float((ref.argmax(1) == lp_dev.argmax(1)).mean())
        repro.update(reference_sha256=P.sha256_file(ref_path), argmax_agreement=agree,
                     max_abs_diff=float(np.abs(ref - lp_dev).max()), mean_abs_diff=float(np.abs(ref - lp_dev).mean()))
        P.record_check(S, f"dev re-scoring reproduces Experiment 7 (argmax agreement >= {cfg['repro_min_agreement']})",
                       agree >= cfg['repro_min_agreement'], hard=True,
                       detail=f'{100 * agree:.1f}% of {len(lp_dev)} utterances; max |diff| {repro["max_abs_diff"]:.4f}')
    else:
        P.record_check(S, 'dev re-scoring compared with Experiment 7', False, hard=False, detail='no reference given')

    lp, info = LZ.llm_logprobs(S, model, tok, 'train')
    lengths = np.array([len(S['D']['videoAudio'][v]) for v in S['keys']['train']])
    P.record_check(S, 'one row per training utterance', len(lp) == int(lengths.sum()), hard=True,
                   detail=f'{len(lp)} rows')
    P.record_check(S, 'the seven label words carry most of the next-token probability on train',
                   info['label_mass_mean'] >= 0.5, hard=False, detail=f'mean {info["label_mass_mean"]:.3f}')
    path = os.path.join(S['out'], 'dumps', 'train_llm.npz')
    np.savez_compressed(path, logp=lp.astype(np.float32), keys=np.asarray(S['keys']['train']), lengths=lengths,
                        data_sha256=np.asarray(S['data_sha256']))
    S['llm_info'] = dict(label_tokens=spelled, reproducibility=repro, repro_dev=info_dev, train=info,
                         train_llm_npz_sha256=P.sha256_file(path))
    P.write_json(os.path.join(S['out'], 'llm_inference.json'), S['llm_info'])
    P._save_checks(S)
    P.log_event(S, 'LLM train predictions done', minutes=round(info['seconds'] / 60, 1),
                label_mass=round(info['label_mass_mean'], 3), sha256=S['llm_info']['train_llm_npz_sha256'][:16])


def _write_summary(S):
    env, li = S['env'], S['llm_info']
    r, t = li['reproducibility'], li['train']
    L = ['# ECERC Limitation 2, Experiment 8a: zero-shot LLM soft labels for the training split', '',
         f'Run {time.strftime("%Y-%m-%d %H:%M")} | {env["device"]} {env["gpus"]} | torch {env["torch"]} | '
         f'transformers {env["transformers"]} | LLM {S["cfg"]["llm_model"]} | data sha256 {S["data_sha256"][:16]}...', '',
         f'- dev re-scoring ({r["dialogues"]} dialogues, {r["utterances"]} utterances): '
         + (f'argmax agreement with Experiment 7 {100 * r["argmax_agreement"]:.1f}%, max |diff| {r["max_abs_diff"]:.4f}'
            if 'argmax_agreement' in r else 'no reference'),
         f'- training split: {t["utterances"]} utterances, {t["new_tokens"]} new tokens, {t["seconds"] / 60:.0f} minutes, '
         f'label-word probability mass {t["label_mass_mean"]:.3f} on average',
         f'- `dumps/train_llm.npz` SHA-256: `{li["train_llm_npz_sha256"]}`',
         '- the test split was not scored and no test label was read']
    warn = [c for c in S['checks'] if not c['passed']]
    L += ['', f'Checks: {sum(c["passed"] for c in S["checks"])} passed, {len(warn)} warnings'
          + ('' if not warn else ' -- ' + '; '.join(f'{c["name"]} ({c["detail"]})' for c in warn))]
    with open(os.path.join(S['out'], 'SUMMARY_distill_teacher.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')


def stage_package(S):
    _write_summary(S)
    zpath = os.path.join(S['cfg']['work_dir'], 'ecerc_l2_distill_teacher_results.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(S['out']):
            for name in files:
                full = os.path.join(root, name)
                z.write(full, os.path.relpath(full, os.path.dirname(S['out'])))
    with open(os.path.join(S['out'], 'SUMMARY_distill_teacher.md')) as f:
        print(f.read())
    print(f'Results archive: {zpath} ({os.path.getsize(zpath) / 2 ** 20:.1f} MB)')
    return zpath
