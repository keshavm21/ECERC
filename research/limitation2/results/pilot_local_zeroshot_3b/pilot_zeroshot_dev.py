"""Dev-only pilot: is a zero-shot LLM complementary to ECERC on MELD's rare emotions?

Runs a zero-shot instruction model (MLX, Apple GPU) on every dev utterance with its past context only (ECERC's
setting, generic speaker labels), reads the next-token probabilities of the seven label words, and measures the
fusion with the three released ECERC checkpoints' saved dev predictions using Experiment 4's own functions
(including its cross-fitted gate statistic). The test split is never read. The prompt is fixed before any dev
result is seen. Progress is checkpointed every 10 dialogues, so an interrupted run resumes where it stopped.

usage: python -I pilot_zeroshot_dev.py SRC_DIR PKL DUMPS_DIR OUT_DIR MODEL REVISION
"""
import os
import sys
import time

src, pkl, dumps, out, model_id, revision = sys.argv[1:7]
sys.path.insert(0, src)
import numpy as np  # noqa: E402

import l2_posthoc as P  # noqa: E402
import l2_expert as LE  # noqa: E402

LABELS = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']  # MELD code order
SYSTEM = 'You are an expert annotator of emotions in TV-show dialogue.'
QUESTION = ('Which emotion does {who} express in the last line? Answer with one word: neutral, surprise, fear, '
            'sadness, joy, disgust, or anger.')


def main():
    os.makedirs(out, exist_ok=True)
    D = dict(zip(P.FIELDS, P.load_pickle(pkl)))
    keys = list(D['valid_ids'])  # dev only
    tag = model_id.split('/')[-1]
    part = os.path.join(out, f'partial_{tag}.npz')
    final = os.path.join(out, f'dev_llm_logp_{tag}.npz')
    if os.path.exists(final):
        lp7 = np.load(final)['logp']
        print('using saved predictions', final, flush=True)
        return analyse(D, keys, lp7)
    rows, done = [], 0
    if os.path.exists(part):
        z = np.load(part)
        rows, done = list(z['raw']), int(z['done'])
        print(f'resuming after {done} dialogues ({len(rows)} utterances)', flush=True)
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
    t0 = time.time()
    model, tok = load(model_id, revision=revision)
    print(f'loaded {model_id}@{revision[:8]} in {time.time() - t0:.0f}s', flush=True)
    ids = []
    for w in LABELS:
        first = {tok.encode(v, add_special_tokens=False)[0] for v in (w, w.capitalize(), ' ' + w, ' ' + w.capitalize())}
        ids.append(sorted(first))
    flat = [i for g in ids for i in g]
    assert len(flat) == len(set(flat)), f'label first tokens collide: {ids}'
    print('label first tokens:', {w: [tok.decode([i]) for i in g] for w, g in zip(LABELS, ids)}, flush=True)
    cache, cached, n_tok = make_prompt_cache(model), [], 0
    t0 = time.time()
    for d in range(done, len(keys)):
        vid = keys[d]
        sents = D['sentences'][vid]
        spk = np.asarray(D['speakers'][vid]).argmax(-1)
        turns = [f'Speaker {int(spk[i]) + 1}: {LE._clean(sents[i])}' for i in range(len(sents))]
        for t in range(len(sents)):
            msgs = [dict(role='system', content=SYSTEM),
                    dict(role='user', content='Here is a dialogue, in order:\n' + '\n'.join(turns[:t + 1]) + '\n\n'
                         + QUESTION.format(who=f'Speaker {int(spk[t]) + 1}'))]
            toks = tok.apply_chat_template(msgs, add_generation_prompt=True)
            n = 0
            for a, b in zip(cached, toks):
                if a != b:
                    break
                n += 1
            n = min(n, len(toks) - 1)
            if len(cached) > n:
                trim_prompt_cache(cache, len(cached) - n)
            logits = model(mx.array(toks[n:])[None], cache=cache)[0, -1].astype(mx.float32)
            lp = logits - mx.logsumexp(logits)
            row = mx.stack([mx.logsumexp(lp[mx.array(g)]) for g in ids])
            mx.eval(row)
            rows.append(np.array(row))
            n_tok += len(toks) - n
            cached = toks
        if (d + 1) % 5 == 0 or d + 1 == len(keys):
            el = time.time() - t0
            print(f'  dialogue {d + 1}/{len(keys)}, {len(rows)} utterances, {n_tok} new tokens, {el:.0f}s '
                  f'({el / (d + 1 - done):.1f}s per dialogue)', flush=True)
        if (d + 1) % 10 == 0:
            np.savez_compressed(part, raw=np.stack(rows), done=d + 1)
    raw = np.stack(rows)
    lp7 = raw - np.logaddexp.reduce(raw, axis=1, keepdims=True)  # renormalised over the seven labels
    np.savez_compressed(final, logp=lp7, raw=raw)
    print(f'inference done: {len(rows)} utterances; label probability mass mean '
          f'{np.exp(np.logaddexp.reduce(raw, axis=1)).mean():.3f}', flush=True)
    analyse(D, keys, lp7)


def analyse(D, keys, lp7):
    C, CL = P.C, P.CLASSES
    E = []
    for name in ('C1', 'C2', 'C3'):
        z = np.load(os.path.join(dumps, f'dev_{name}.npz'))
        assert [str(v) for v in keys] == list(z['vids']), 'dump order differs from the dev keys'
        E.append(dict(name=name, logp=z['logp'].astype(np.float64), y=z['y'], dlg=z['dlg']))
    y = E[0]['y']
    assert np.array_equal(y, np.concatenate([np.asarray(D['emotion_labels'][v]) for v in keys]))
    cm = lambda p: np.bincount(y * C + p, minlength=C * C).reshape(C, C)  # noqa: E731

    def report(name, pred):
        s = P.scores(cm(pred))
        print(f'  {name:<34s} acc {100 * s["acc"]:5.2f}  wF1 {100 * s["wf1"]:5.2f}  macro {100 * s["mf1"]:5.2f} | '
              + ' '.join(f'{c} {100 * f:5.1f}' for c, f in zip(CL, s['f1'])))
        return s

    print('\nDEV (1109 utterances; 40 Fear, 22 Disgust)')
    report('zero-shot LLM alone', lp7.argmax(1))
    pred = lp7.argmax(1)
    print('  LLM predicted-label counts:', dict(zip(CL, np.bincount(pred, minlength=C).tolist())),
          '| true:', dict(zip(CL, np.bincount(y, minlength=C).tolist())))
    for e in E:
        report(f'ECERC {e["name"]}', e['logp'].argmax(1))
    T = [dict(logp=lp7, y=y, dlg=E[0]['dlg'])]
    n_dev = len(keys)
    dev_cms = LE.pair_cms(E, T, n_dev)
    r, mean = LE._select(dev_cms.sum(0))
    fc = LE.crossfit_forecast(dev_cms, 5, 12345, 2000, 280)  # Experiment 4/5's gate statistic and settings
    print('\nFUSION score = (1-w) log p_ECERC + w log p_LLM, mean over C1-C3:')
    print('  dev macro-F1 by w: ' + ', '.join(f'{w}:{100 * m:.2f}' for w, m in zip(LE.W_GRID, mean)))
    print(f'  in-sample best w = {LE.W_GRID[r]}: {100 * mean[r]:.2f} vs {100 * mean[0]:.2f} for ECERC alone '
          f'({100 * (mean[r] - mean[0]):+.2f})')
    print(f'  CROSS-FITTED dev gain {fc["crossfit_delta_macro_f1"]:+.2f} (fold w {fc["fold_w"]}); '
          f'dev bootstrap 95% CI {np.round(fc["dev_bootstrap_ci95"], 2).tolist()}; projected test SE '
          f'{fc["se_test_projected"]:.2f}  [Experiment 4: +0.97, Experiment 5: -0.98, gate +0.50]')
    w = LE.W_GRID[r]
    sc = P.scores(dev_cms.sum(0).reshape(len(E), len(LE.W_GRID), C, C))
    print(f'  per-class dev F1 change at w={w} (mean over C1-C3): '
          + ', '.join(f'{c} {100 * (sc["f1"][:, r, i] - sc["f1"][:, 0, i]).mean():+.1f}' for i, c in enumerate(CL)))
    print(f'  weighted F1 change at w={w}: {100 * (sc["wf1"][:, r] - sc["wf1"][:, 0]).mean():+.2f}; accuracy change '
          f'{100 * (sc["acc"][:, r] - sc["acc"][:, 0]).mean():+.2f}')
    for c in (2, 5, 3):
        m = y == c
        a = np.mean([((e['logp'].argmax(1)[m] != c) & (pred[m] == c)).sum() for e in E])
        b = np.mean([((e['logp'].argmax(1)[m] == c) & (pred[m] != c)).sum() for e in E])
        print(f'  {CL[c]}: ECERC wrong but LLM right {a:.1f}, ECERC right but LLM wrong {b:.1f} (n={int(m.sum())})')
    # secondary, label-free: remove the LLM's own average prediction on the dev inputs, add the training prior
    prior = np.bincount(np.concatenate([np.asarray(D['emotion_labels'][v]) for v in D['train_ids']]), minlength=C)
    log_prior = np.log(prior / prior.sum())
    q = np.log(np.exp(lp7).mean(0))
    cal = lp7 - q[None, :] + log_prior[None, :]
    cal = cal - np.logaddexp.reduce(cal, axis=1, keepdims=True)
    report('LLM, prior-matched (label-free)', cal.argmax(1))
    dev_cms2 = LE.pair_cms(E, [dict(logp=cal, y=y, dlg=E[0]['dlg'])], n_dev)
    r2, mean2 = LE._select(dev_cms2.sum(0))
    fc2 = LE.crossfit_forecast(dev_cms2, 5, 12345, 2000, 280)
    print(f'  secondary, prior-matched LLM: in-sample best w {LE.W_GRID[r2]} {100 * (mean2[r2] - mean2[0]):+.2f}; '
          f'cross-fitted dev gain {fc2["crossfit_delta_macro_f1"]:+.2f}')


if __name__ == '__main__':
    main()
