#!/usr/bin/env python3
"""
Worker for Experiment 5 (LLM expert). Runs in its own process, one per GPU, so seeds can train in parallel.

    python l2_llm_worker.py train   --cfg run_cfg.json --seed 1 --out DIR
    python l2_llm_worker.py predict --cfg run_cfg.json --adapter DIR/adapter --inputs inputs.json --out FILE.npz

train: fine-tunes a LoRA adapter plus the classification head on the training inputs, evaluates on dev every
cfg['eval_every'] epochs and keeps the adapter with the best dev macro-F1 (DIR/adapter, DIR/dev_logp.npz,
DIR/result.json). It never reads test inputs. predict: writes log-probabilities for the given inputs (no labels).
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

C = 7


def log(*a):
    print(*a, flush=True)


def macro_f1(y, p):
    cm = np.bincount(y * C + p, minlength=C * C).reshape(C, C)
    tp, s, q = np.diag(cm).astype(float), cm.sum(1), cm.sum(0)
    f1 = np.divide(2 * tp, s + q, out=np.zeros(C), where=(s + q) > 0)
    return 100 * f1.mean(), 100 * f1


def load_model(cfg, adapter=None):
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
    cuda = torch.cuda.is_available()
    quant = bool(cfg['quantize']) and cuda
    kw = dict(num_labels=C)
    if quant:
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                        bnb_4bit_use_double_quant=True,
                                                        bnb_4bit_compute_dtype=torch.float16)
        kw['device_map'] = {'': 0}
    dtype = torch.float16 if cuda else torch.float32
    try:
        model = AutoModelForSequenceClassification.from_pretrained(cfg['model'], dtype=dtype, **kw)
    except TypeError:  # older transformers
        model = AutoModelForSequenceClassification.from_pretrained(cfg['model'], torch_dtype=dtype, **kw)
    model.config.pad_token_id = cfg['pad_id']
    model.config.use_cache = False
    if adapter is None:
        if quant:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                    gradient_checkpointing_kwargs={'use_reentrant': False})
        elif cuda:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'],
                                                 lora_dropout=cfg['lora_dropout'], task_type='SEQ_CLS',
                                                 target_modules=cfg['lora_targets']))
    else:
        model = PeftModel.from_pretrained(model, adapter)
    if not quant:
        model.to('cuda' if cuda else 'cpu')
    for p in model.parameters():  # trainable weights in fp32 (needed by the fp16 gradient scaler)
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return model


def collate(ids, idx, pad_id, device):
    L = max(len(ids[i]) for i in idx)
    x = torch.full((len(idx), L), pad_id, dtype=torch.long)
    a = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx):  # right padding: the classifier reads the last real token
        x[r, :len(ids[i])] = torch.tensor(ids[i])
        a[r, :len(ids[i])] = 1
    return x.to(device), a.to(device)


def predict(model, ids, pad_id, bs, device):
    model.eval()
    order = np.argsort([len(s) for s in ids])
    out = np.zeros((len(ids), C))
    with torch.no_grad():
        for b in range(0, len(ids), bs):
            idx = order[b:b + bs]
            x, a = collate(ids, idx, pad_id, device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device == 'cuda'):
                logits = model(input_ids=x, attention_mask=a).logits
            out[idx] = F.log_softmax(logits.float(), dim=-1).cpu().numpy()
    if not np.isfinite(out).all():
        raise RuntimeError('non-finite predictions (fp16 overflow?)')
    return out


def train(cfg, seed, out):
    from transformers import get_cosine_schedule_with_warmup
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with open(cfg['trainval_path']) as f:
        data = json.load(f)
    tr_ids, tr_y = data['train']['ids'], np.asarray(data['train']['y'])
    dv_ids, dv_y = data['valid']['ids'], np.asarray(data['valid']['y'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    model = load_model(cfg)
    log(f'[seed {seed}] model loaded on {device} in {time.time() - t0:.0f}s; trainable parameters '
        f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M')
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=0.0)
    micro, accum = cfg['micro_batch'], cfg['grad_accum']
    n = len(tr_ids)
    micro_per_epoch = math.ceil(n / micro)
    opt_steps = math.ceil(micro_per_epoch / accum) * cfg['epochs']
    sched = get_cosine_schedule_with_warmup(opt, int(cfg['warmup'] * opt_steps), opt_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    rng = np.random.default_rng(seed)
    eval_every = max(1, int(round(micro_per_epoch * cfg['eval_every'])))
    lengths = np.asarray([len(s) for s in tr_ids])
    best, bad, curve, mstep, stop, t0, bad_loss = -1.0, 0, [], 0, False, time.time(), 0
    for epoch in range(cfg['epochs']):
        model.train()
        perm, mega = rng.permutation(n), micro * 64
        batches = []
        for i in range(0, n, mega):  # group similar lengths to save padding, then shuffle the batches
            ch = perm[i:i + mega]
            ch = ch[np.argsort(lengths[ch], kind='stable')]
            batches += [ch[j:j + micro] for j in range(0, len(ch), micro)]
        order = rng.permutation(len(batches))
        for k, bi in enumerate(order):
            idx = batches[bi]
            x, a = collate(tr_ids, idx, cfg['pad_id'], device)
            y = torch.as_tensor(tr_y[idx], device=device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device == 'cuda'):
                logits = model(input_ids=x, attention_mask=a).logits
            loss = F.cross_entropy(logits.float(), y) / accum
            if not torch.isfinite(loss):  # fp16 trouble: skip the batch, give up early if it keeps happening
                bad_loss += 1
                opt.zero_grad(set_to_none=True)
                if bad_loss > 20:
                    raise RuntimeError('training diverged: more than 20 non-finite losses')
                continue
            scaler.scale(loss).backward()
            mstep += 1
            last = k == len(order) - 1
            if mstep % accum == 0 or last:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            time_up = (time.time() - t0) / 60 > cfg.get('max_minutes', float('inf'))
            if mstep % eval_every == 0 or last or time_up:
                lp = predict(model, dv_ids, cfg['pad_id'], cfg['eval_batch'], device)
                mf, f1 = macro_f1(dv_y, lp.argmax(1))
                acc = 100 * float((lp.argmax(1) == dv_y).mean())
                curve.append(dict(epoch=epoch, micro_step=mstep, dev_macro_f1=mf, dev_acc=acc, f1=f1.tolist(),
                                  minutes=round((time.time() - t0) / 60, 1)))
                log(f'[seed {seed}] epoch {epoch} step {mstep}/{micro_per_epoch * cfg["epochs"]}: dev macro-F1 {mf:.2f}, '
                    f'acc {acc:.2f}, Fear {f1[2]:.1f}, Disgust {f1[5]:.1f}, loss {loss.item() * accum:.3f} '
                    f'({curve[-1]["minutes"]} min)')
                if mf > best + 1e-9:
                    best, bad = mf, 0
                    model.save_pretrained(os.path.join(out, 'adapter'))
                    np.savez_compressed(os.path.join(out, 'dev_logp.npz'), logp=lp.astype(np.float32), y=dv_y)
                else:
                    bad += 1
                if bad >= cfg['patience'] or time_up:  # patience exhausted, or the training time cap reached
                    stop = True
                    if time_up:
                        log(f'[seed {seed}] training time cap of {cfg["max_minutes"]} min reached; stopping')
                    break
                model.train()
        if stop:
            break
    with open(os.path.join(out, 'result.json'), 'w') as f:
        json.dump(dict(seed=seed, best_dev_macro_f1=best, curve=curve, minutes=round((time.time() - t0) / 60, 1),
                       stopped_early=stop), f, indent=2)
    log(f'[seed {seed}] done: best dev macro-F1 {best:.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['train', 'predict'])
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--adapter')
    ap.add_argument('--inputs')
    a = ap.parse_args()
    with open(a.cfg) as f:
        cfg = json.load(f)
    if a.mode == 'train':
        os.makedirs(a.out, exist_ok=True)
        train(cfg, a.seed, a.out)
    else:
        with open(a.inputs) as f:
            ids = json.load(f)['ids']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model(cfg, adapter=a.adapter)
        lp = predict(model, ids, cfg['pad_id'], cfg['eval_batch'], device)
        np.savez_compressed(a.out, logp=lp.astype(np.float32))
        log(f'predicted {len(ids)} inputs -> {a.out}')


if __name__ == '__main__':
    main()
