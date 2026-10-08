"""Person B class-imbalance experiments on MELD: R0, B0, CB, FL, OS and FL+OS.

The ECERC model, MELD dataset, loss and data loaders are imported unchanged from this folder.
With no Person B option set, training follows MELD/train.py step for step (B0).

    python train_imbalance.py --eval_only --load_model_state_dir ECERC_MODEL.pkl   # R0: released checkpoint
    python train_imbalance.py --seed 0                                             # B0: retrained baseline
    python train_imbalance.py --seed 0 --cb_beta 0.999                             # CB: class-balanced loss
    python train_imbalance.py --seed 0 --gamma 2                                   # FL: focal variant
    python train_imbalance.py --seed 0 --oversample --os_rho 2                     # OS: Fear/Disgust oversampling
    python train_imbalance.py --seed 0 --gamma 2 --oversample --os_rho 2           # FL+OS
    python train_imbalance.py --dry_run [options]                                  # data and settings checks only

Each run writes config.json, train.log, epochs.csv, best_model.pkl, preds_valid.csv, preds_test.csv
and metrics.json to results/<variant>_s<seed>/ (or --run_dir). metrics.json is written last.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings

import numpy as np
import sklearn
import torch
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, SequentialSampler, WeightedRandomSampler

from loss import Loss
from model import ECERC
from train import get_MELD_bert_loaders, seed_everything

warnings.filterwarnings("ignore")

# Fixed settings, identical to the MELD/train.py defaults (emotion labels, multi-modal features).
CLASS_NAMES = ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang']  # label order of target_names in train.py
N_CLASSES = len(CLASS_NAMES)
FEAR, DISGUST = CLASS_NAMES.index('fea'), CLASS_NAMES.index('dis')
N_SPEAKERS, HIDDEN_SIZE, BASE_LAYER = 9, 128, 1
D_TEXT, D_AUDIO, D_VISUAL = 1024, 300, 342  # train.py: D_text, feat2dim['MELD_audio'], feat2dim['denseface']
BATCH_SIZE, LR, L2, EPOCHS, PATIENCE = 32, 1e-5, 2e-4, 40, 20
BASELINE_FILES = ['model.py', 'dataloader.py', 'loss.py', 'train.py', 'inference.py']
EPOCH_COLUMNS = ['epoch', 'train_loss', 'train_acc', 'train_fscore', 'valid_loss', 'valid_acc', 'valid_fscore',
                 'test_loss', 'test_acc', 'test_fscore', 'valid_macro_f1', 'valid_fear_f1', 'valid_disgust_f1',
                 'time_sec']


def parse_args():
    parser = argparse.ArgumentParser(description='Person B class-imbalance experiments on MELD.')
    parser.add_argument('--seed', type=int, default=0, help='random seed (MELD/train.py always uses 0)')
    parser.add_argument('--gamma', type=float, default=1, help='focal exponent of Loss: 1 = baseline, 2 = FL')
    parser.add_argument('--cb_beta', type=float, default=None,
                        help='CB: class-balanced weights with this effective-number beta, e.g. 0.999')
    parser.add_argument('--oversample', action='store_true',
                        help='OS: oversample training dialogues that contain Fear or Disgust')
    parser.add_argument('--os_rho', type=float, default=None,
                        help='OS: sampling weight of those dialogues (all others have weight 1)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='number of epochs (40 as in MELD/train.py; fewer only for smoke tests)')
    parser.add_argument('--run_dir', type=str, default=None, help='output folder (default: results/<variant>_s<seed>)')
    parser.add_argument('--overwrite', action='store_true', help='replace a finished run in the output folder')
    parser.add_argument('--eval_only', action='store_true',
                        help='R0: only evaluate the checkpoint given by --load_model_state_dir')
    parser.add_argument('--load_model_state_dir', type=str, default='ECERC_MODEL.pkl', help='checkpoint for --eval_only')
    parser.add_argument('--dry_run', action='store_true',
                        help='load the data, print the checks and the loss/sampler settings, then exit')
    args = parser.parse_args()

    if float(args.gamma).is_integer():
        args.gamma = int(args.gamma)  # same type as the baseline's Loss(gamma=1)
    if args.cb_beta is not None and not 0 < args.cb_beta < 1:
        parser.error('--cb_beta must be between 0 and 1')
    if args.oversample != (args.os_rho is not None):
        parser.error('--oversample and --os_rho must be given together')
    if args.os_rho is not None and args.os_rho <= 0:
        parser.error('--os_rho must be positive')
    if args.epochs < 1:
        parser.error('--epochs must be at least 1')
    if args.eval_only and (args.cb_beta is not None or args.oversample or args.gamma != 1):
        parser.error('--eval_only evaluates a checkpoint; the training options do not apply')
    return args


def variant_name(args):
    if args.eval_only:
        return 'R0'
    parts = []
    if args.cb_beta is not None:
        parts.append('CB{:g}'.format(args.cb_beta))
    if args.gamma != 1:
        parts.append('FL{:g}'.format(args.gamma))
    if args.oversample:
        parts.append('OS{:g}'.format(args.os_rho))
    return '-'.join(parts) or 'B0'


def default_run_dir(args, variant):
    if args.eval_only:
        return os.path.join('results', 'R0_' + os.path.splitext(os.path.basename(args.load_model_state_dir))[0])
    name = '{}_s{}'.format(variant, args.seed)
    if args.epochs != EPOCHS:
        name += '_e{}'.format(args.epochs)
    return os.path.join('results', name)


class Tee:
    """Copies everything printed into the run's log file."""

    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, 'w')

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


def open_run_dir(run_dir, overwrite, log_name):
    if os.path.exists(os.path.join(run_dir, 'metrics.json')) and not overwrite:
        sys.exit('{} already holds a finished run; pass --overwrite to replace it.'.format(run_dir))
    os.makedirs(run_dir, exist_ok=True)
    sys.stdout = Tee(sys.stdout, os.path.join(run_dir, log_name))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=here, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ''
    try:
        gpu = torch.cuda.get_device_name(0)
    except Exception:
        gpu = None
    return {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'gpu': gpu, 'numpy': np.__version__,
            'sklearn': sklearn.__version__, 'git_commit': commit or None,
            'baseline_sha256': {f: sha256(os.path.join(here, f)) for f in BASELINE_FILES},
            'command': ' '.join(sys.argv)}


def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


# ---------------------------------------------------------------- data, class weights and sampler

def split_label_counts(dataset):
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for vid in dataset.keys:
        for y in dataset.emotion_labels[vid]:
            counts[int(y)] += 1
    return counts


def train_class_counts(trainset):
    """Per-class utterance counts of the training split; refuses any other split."""
    keys = set(trainset.keys)
    if keys != set(trainset.train_ids) or keys & (set(trainset.valid_ids) | set(trainset.test_ids)):
        raise ValueError('class weights must be computed from the training split only')
    return split_label_counts(trainset)


def class_balanced_weights(counts, beta):
    """Effective-number weights (Cui et al., 2019), rescaled so the mean weight per training utterance is 1."""
    counts = np.asarray(counts, dtype=np.float64)
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    return weights * counts.sum() / (counts * weights).sum()


def minority_dialogue_mask(trainset):
    """True for each training dialogue that contains at least one Fear or Disgust utterance."""
    return [any(int(y) in (FEAR, DISGUST) for y in trainset.emotion_labels[vid]) for vid in trainset.keys]


def class_mix(trainset, dialogue_weights):
    """Share of each class among training utterances when dialogues are drawn in proportion to these weights."""
    mix = np.zeros(N_CLASSES)
    for vid, w in zip(trainset.keys, dialogue_weights):
        for y in trainset.emotion_labels[vid]:
            mix[int(y)] += w
    return mix / mix.sum()


def check_splits(train_loader, valid_loader, test_loader):
    """Checks that validation and test are the untouched official splits; returns per-split statistics."""
    train, valid, test = train_loader.dataset, valid_loader.dataset, test_loader.dataset
    train_ids, valid_ids, test_ids = set(train.keys), set(valid.keys), set(test.keys)
    assert train_ids == set(train.train_ids) and valid_ids == set(valid.valid_ids) and test_ids == set(test.test_ids)
    assert not (train_ids & valid_ids or train_ids & test_ids or valid_ids & test_ids), 'splits overlap'
    for loader in (valid_loader, test_loader):
        assert isinstance(loader.sampler, SequentialSampler) and loader.batch_size == BATCH_SIZE
    stats = {}
    for name, dataset in (('train', train), ('valid', valid), ('test', test)):
        counts = split_label_counts(dataset)
        stats[name] = {'dialogues': len(dataset), 'utterances': int(counts.sum()), 'label_counts': counts.tolist()}
    return stats


def build_loaders(args):
    """The original MELD loaders; with --oversample only the training loader's sampler changes."""
    train_loader, valid_loader, test_loader = get_MELD_bert_loaders(None, batch_size=BATCH_SIZE, num_workers=0)
    stats = check_splits(train_loader, valid_loader, test_loader)
    oversampling = None
    if args.oversample:
        trainset = train_loader.dataset
        mask = minority_dialogue_mask(trainset)
        weights = [args.os_rho if m else 1.0 for m in mask]
        sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True,
                                        generator=torch.Generator().manual_seed(args.seed))
        train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=trainset.collate_fn,
                                  num_workers=0, pin_memory=False)
        n_minority = sum(mask)
        oversampling = {'rho': args.os_rho, 'classes': [CLASS_NAMES[FEAR], CLASS_NAMES[DISGUST]],
                        'train_dialogues': len(trainset), 'minority_dialogues': n_minority,
                        'expected_minority_dialogues_per_epoch': len(trainset) * args.os_rho * n_minority / sum(weights),
                        'natural_class_mix': class_mix(trainset, [1.0] * len(trainset)).tolist(),
                        'expected_class_mix': class_mix(trainset, weights).tolist()}
    return train_loader, valid_loader, test_loader, stats, oversampling


def build_losses(args, train_loader):
    """The variant's training loss, and the baseline loss used for validation/test loss (early stopping)."""
    alpha, cb_info = None, None
    if args.cb_beta is not None:
        counts = train_class_counts(train_loader.dataset)
        weights = class_balanced_weights(counts, args.cb_beta)
        alpha = torch.tensor(weights, dtype=torch.float32)
        raw = (1.0 - args.cb_beta) / (1.0 - np.power(args.cb_beta, counts.astype(np.float64)))
        cb_info = {'beta': args.cb_beta, 'train_counts': counts.tolist(), 'weights': weights.tolist(),
                   'cui_weights_sum_to_n_classes': (raw * N_CLASSES / raw.sum()).tolist()}
    return Loss(gamma=args.gamma, alpha=alpha), Loss(gamma=1, alpha=None), cb_info


def print_split_table(stats):
    print('{:<6}{:>10}{:>11}'.format('split', 'dialogues', 'utterances') + ''.join('{:>6}'.format(c) for c in CLASS_NAMES))
    for name, s in stats.items():
        print('{:<6}{:>10}{:>11}'.format(name, s['dialogues'], s['utterances'])
              + ''.join('{:>6}'.format(n) for n in s['label_counts']))


def print_class_weights(cb_info):
    print('Class-balanced weights (beta={:g}, training counts only, mean weight per training utterance = 1):'
          .format(cb_info['beta']))
    print('        ' + ''.join('{:>7}'.format(c) for c in CLASS_NAMES))
    print('count   ' + ''.join('{:>7}'.format(n) for n in cb_info['train_counts']))
    print('weight  ' + ''.join('{:>7.3f}'.format(w) for w in cb_info['weights']))


def print_oversampling(info):
    print('Oversampling: rho={:g} for the {} of {} training dialogues that contain Fear or Disgust; '
          '{} dialogues drawn per epoch with replacement (expected {:.1f} with Fear or Disgust).'
          .format(info['rho'], info['minority_dialogues'], info['train_dialogues'], info['train_dialogues'],
                  info['expected_minority_dialogues_per_epoch']))
    print('training utterance mix (%)  ' + ''.join('{:>7}'.format(c) for c in CLASS_NAMES))
    print('  natural                   ' + ''.join('{:>7.2f}'.format(100 * v) for v in info['natural_class_mix']))
    print('  expected with sampler     ' + ''.join('{:>7.2f}'.format(100 * v) for v in info['expected_class_mix']))


# ---------------------------------------------------------------- model passes and metrics

def build_model(args):
    return ECERC(args, d_t=D_TEXT, d_a=D_AUDIO, d_v=D_VISUAL, base_layer=BASE_LAYER,
                 input_size=D_TEXT + D_AUDIO + D_VISUAL, hidden_size=HIDDEN_SIZE,
                 n_speakers=N_SPEAKERS, n_classes=N_CLASSES, cuda_flag=True)


def run_epoch(model, loss_f, dataloader, optimizer=None):
    """One pass over `dataloader`, training when `optimizer` is given.

    The steps are those of train_or_eval_model in MELD/train.py (multi-modal features, GPU), and the
    loss/accuracy/weighted F1 are rounded the same way. Evaluation passes also keep the log-probs and
    a (dialogue id, utterance index) key per utterance.
    """
    train_flag = optimizer is not None
    losses, preds, labels, log_probs, keys = [], [], [], [], []
    if train_flag:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if train_flag:
            optimizer.zero_grad()
        emo_roberta, sem_roberta, audio, vision, qmask, umask, label = [d.cuda() for d in data[:-1]]
        seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        emo_roberta = torch.cat([emo_roberta, audio, vision], dim=-1)
        sem_roberta = torch.cat([sem_roberta], dim=-1)

        log_prob = model(emo_roberta, sem_roberta, qmask, umask, seq_lengths)

        label = torch.cat([label[j][:seq_lengths[j]] for j in range(len(label))])
        loss = loss_f(log_prob, label)

        preds.append(torch.argmax(log_prob, 1).data.cpu().numpy())
        labels.append(label.data.cpu().numpy())
        losses.append(loss.item())

        if train_flag:
            loss.backward()
            optimizer.step()
        else:
            log_probs.append(log_prob.data.cpu().numpy())
            keys.extend((vid, k) for vid, n in zip(data[-1], seq_lengths) for k in range(n))

    preds, labels = np.concatenate(preds), np.concatenate(labels)
    return {'loss': round(np.sum(losses) / len(losses), 4),
            'acc': round(accuracy_score(labels, preds) * 100, 2),
            'fscore': round(f1_score(labels, preds, average='weighted') * 100, 2),
            'preds': preds, 'labels': labels, 'log_probs': np.concatenate(log_probs) if log_probs else None,
            'keys': keys}


def compute_metrics(labels, preds):
    """Every reported metric for one split, in percent and unrounded."""
    classes = list(range(N_CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, labels=classes, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=classes)
    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        tp = int(cm[c, c])
        per_class[name] = {'precision': 100 * float(precision[c]), 'recall': 100 * float(recall[c]),
                           'f1': 100 * float(f1[c]), 'support': int(support[c]),
                           'tp': tp, 'fp': int(cm[:, c].sum()) - tp, 'fn': int(cm[c, :].sum()) - tp}
    return {'n': int(len(labels)),
            'accuracy': 100 * float(accuracy_score(labels, preds)),
            'weighted_f1': 100 * float(f1_score(labels, preds, average='weighted')),
            'macro_f1': 100 * float(f1_score(labels, preds, labels=classes, average='macro', zero_division=0)),
            'per_class': per_class,
            'confusion_matrix': cm.tolist(),  # rows: true label, columns: predicted label
            'true_counts': cm.sum(axis=1).tolist(),
            'pred_counts': cm.sum(axis=0).tolist()}


def print_result(split, m):
    fear, disgust = m['per_class']['fea'], m['per_class']['dis']
    print('{:<6} acc {:6.2f}  wF1 {:6.2f}  macro-F1 {:6.2f} | Fear P/R/F1 {:6.2f} {:6.2f} {:6.2f} '
          '| Disgust P/R/F1 {:6.2f} {:6.2f} {:6.2f}'
          .format(split, m['accuracy'], m['weighted_f1'], m['macro_f1'], fear['precision'], fear['recall'],
                  fear['f1'], disgust['precision'], disgust['recall'], disgust['f1']))


def write_predictions(path, result):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dialogue_id', 'utterance_index', 'label', 'pred'] + ['logp_' + c for c in CLASS_NAMES])
        for (vid, k), y, p, lp in zip(result['keys'], result['labels'], result['preds'], result['log_probs']):
            writer.writerow([vid, k, int(y), int(p)] + ['{:.6f}'.format(v) for v in lp])


# ---------------------------------------------------------------- the three modes

def dry_run(args):
    variant = variant_name(args)
    print('Dry run for {}: data and settings checks only, no model and no training.'.format(variant))
    train_loader, valid_loader, test_loader, stats, oversampling = build_loaders(args)
    print_split_table(stats)
    shapes = [tuple(t.shape) for t in train_loader.dataset[0][:5]]
    print('First training dialogue: text {}, event {}, videoVisual {} (passed in the "audio" position), '
          'videoAudio {} (passed in the "visual" position), speakers {}.'.format(*shapes))
    print('The model splits the concatenated input as text {}, audio {}, visual {}, unchanged from MELD/train.py.'
          .format(D_TEXT, D_AUDIO, D_VISUAL))
    _, _, cb_info = build_losses(args, train_loader)
    print('Training loss: Loss(gamma={}, alpha={}); validation/test loss: Loss(gamma=1, alpha=None).'
          .format(args.gamma, 'class-balanced weights' if cb_info else None))
    if cb_info:
        print_class_weights(cb_info)
        counts = np.array(cb_info['train_counts'])
        print('check: mean weight per training utterance = {:.6f}'
              .format((counts * np.array(cb_info['weights'])).sum() / counts.sum()))
    if oversampling:
        print_oversampling(oversampling)
        mask = minority_dialogue_mask(train_loader.dataset)
        drawn = list(iter(train_loader.sampler))
        print('check: one simulated epoch drew {} dialogues, {} of them with Fear or Disgust.'
              .format(len(drawn), sum(mask[i] for i in drawn)))
    print('Training loader: {}. Validation and test loaders: sequential, batch {}, no sampler (checked).'
          .format('weighted sampler' if args.oversample else 'shuffled, as in MELD/train.py', BATCH_SIZE))


def train(args):
    variant = variant_name(args)
    run_dir = args.run_dir or default_run_dir(args, variant)
    open_run_dir(run_dir, args.overwrite, 'train.log')
    print('{} (seed {}) -> {}'.format(variant, args.seed, run_dir))

    # Same order as MELD/train.py: seed, model, GPU, data loaders, optimizer, loss.
    seed_everything(args.seed)
    model = build_model(args)
    model.cuda()
    print('Running on GPU')
    print("The model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print('Running on the multi features........')

    train_loader, valid_loader, test_loader, stats, oversampling = build_loaders(args)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    train_loss_f, eval_loss_f, cb_info = build_losses(args, train_loader)
    print_split_table(stats)
    if cb_info:
        print_class_weights(cb_info)
    if oversampling:
        print_oversampling(oversampling)
    write_json(os.path.join(run_dir, 'config.json'), {
        'variant': variant, 'seed': args.seed, 'mode': 'train',
        'hyperparameters': {'batch_size': BATCH_SIZE, 'lr': LR, 'l2': L2, 'epochs': args.epochs,
                            'patience': PATIENCE, 'base_layer': BASE_LAYER, 'hidden_size': HIDDEN_SIZE,
                            'features': 'multi'},
        'loss': {'gamma': args.gamma, 'cb_beta': args.cb_beta, 'class_balanced': cb_info},
        'oversampling': oversampling, 'data': stats, 'environment': environment()})

    epochs_file = open(os.path.join(run_dir, 'epochs.csv'), 'w', newline='')
    epochs_csv = csv.writer(epochs_file)
    epochs_csv.writerow(EPOCH_COLUMNS)

    all_test_fscore, all_test_acc = [], []
    best_epoch, best_epoch2, patience, best_eval_fscore, best_eval_loss = -1, -1, 0, 0, None
    patience2 = 0
    best, stopped_early = None, False
    for e in range(args.epochs):
        start_time = time.time()
        train_res = run_epoch(model, train_loss_f, train_loader, optimizer)
        valid_res = run_epoch(model, eval_loss_f, valid_loader)
        test_res = run_epoch(model, eval_loss_f, test_loader)
        all_test_fscore.append(test_res['fscore'])
        all_test_acc.append(test_res['acc'])

        # Model selection and early stopping exactly as in MELD/train.py, on the validation split.
        eval_loss, eval_fscore = valid_res['loss'], valid_res['fscore']
        if e == 0 or best_eval_fscore < eval_fscore:
            patience = 0
            best_epoch, best_eval_fscore = e, eval_fscore
            best = {'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    'valid': valid_res, 'test': test_res}
        else:
            patience += 1
        if best_eval_loss is None:
            best_eval_loss = eval_loss
            best_epoch2 = 0
        else:
            if eval_loss < best_eval_loss:
                best_epoch2, best_eval_loss = e, eval_loss
                patience2 = 0
            else:
                patience2 += 1

        valid_metrics = compute_metrics(valid_res['labels'], valid_res['preds'])
        epoch_time = round(time.time() - start_time, 2)
        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, '
              'valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'
              .format(e, train_res['loss'], train_res['acc'], train_res['fscore'], valid_res['loss'],
                      valid_res['acc'], valid_res['fscore'], test_res['loss'], test_res['acc'], test_res['fscore'],
                      epoch_time))
        print('    valid macro-F1: {:.2f}, Fear F1: {:.2f}, Disgust F1: {:.2f}'
              .format(valid_metrics['macro_f1'], valid_metrics['per_class']['fea']['f1'],
                      valid_metrics['per_class']['dis']['f1']))
        epochs_csv.writerow([e, train_res['loss'], train_res['acc'], train_res['fscore'], valid_res['loss'],
                             valid_res['acc'], valid_res['fscore'], test_res['loss'], test_res['acc'],
                             test_res['fscore'], valid_metrics['macro_f1'], valid_metrics['per_class']['fea']['f1'],
                             valid_metrics['per_class']['dis']['f1'], epoch_time])
        epochs_file.flush()

        if patience >= PATIENCE and patience2 >= PATIENCE:
            print('Early stoping...', patience, patience2)
            stopped_early = True
            break
    epochs_file.close()

    print('Final Test performance...')
    print('Early stoping...', patience, patience2)
    print('Eval-metric: F1, Epoch: {}, best_eval_fscore: {}, Accuracy: {}, F1-Score: {}'
          .format(best_epoch, best_eval_fscore, all_test_acc[best_epoch], all_test_fscore[best_epoch]))
    print('Eval-metric: Loss, Epoch: {}, Accuracy: {}, F1-Score: {}'
          .format(best_epoch2, all_test_acc[best_epoch2], all_test_fscore[best_epoch2]))

    torch.save(best['state'], os.path.join(run_dir, 'best_model.pkl'))
    write_predictions(os.path.join(run_dir, 'preds_valid.csv'), best['valid'])
    write_predictions(os.path.join(run_dir, 'preds_test.csv'), best['test'])
    result = {'variant': variant, 'seed': args.seed,
              'selection': 'epoch with the best validation weighted F1 (first maximum), as in MELD/train.py',
              'selected_epoch': best_epoch, 'epochs_run': len(all_test_acc), 'stopped_early': stopped_early,
              'valid': compute_metrics(best['valid']['labels'], best['valid']['preds']),
              'test': compute_metrics(best['test']['labels'], best['test']['preds']),
              'lowest_valid_loss_epoch': {'epoch': best_epoch2, 'test_acc': all_test_acc[best_epoch2],
                                          'test_fscore': all_test_fscore[best_epoch2]}}
    write_json(os.path.join(run_dir, 'metrics.json'), result)
    print('\nSelected epoch {} (best validation weighted F1). Metrics in %:'.format(best_epoch))
    print_result('valid', result['valid'])
    print_result('test', result['test'])
    print('Saved to {}'.format(run_dir))


def evaluate_checkpoint(args):
    run_dir = args.run_dir or default_run_dir(args, 'R0')
    open_run_dir(run_dir, args.overwrite, 'eval.log')
    print('R0: evaluating {} -> {}'.format(args.load_model_state_dir, run_dir))

    # Same steps as MELD/inference.py, which does not seed and builds the model before the loaders.
    model = build_model(args)
    model.cuda()
    train_loader, valid_loader, test_loader, stats, _ = build_loaders(args)
    eval_loss_f = Loss(alpha=None)
    print_split_table(stats)
    write_json(os.path.join(run_dir, 'config.json'), {
        'variant': 'R0', 'seed': None, 'mode': 'eval_only', 'checkpoint': args.load_model_state_dir,
        'checkpoint_sha256': sha256(args.load_model_state_dir), 'data': stats, 'environment': environment()})

    start_time = time.time()
    model.load_state_dict(torch.load(args.load_model_state_dir))
    test_res = run_epoch(model, eval_loss_f, test_loader)
    print('test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'
          .format(test_res['loss'], test_res['acc'], test_res['fscore'], round(time.time() - start_time, 2)))
    print(metrics.classification_report(test_res['labels'], test_res['preds'], target_names=CLASS_NAMES, digits=4))
    print(metrics.confusion_matrix(test_res['labels'], test_res['preds']))
    valid_res = run_epoch(model, eval_loss_f, valid_loader)

    write_predictions(os.path.join(run_dir, 'preds_valid.csv'), valid_res)
    write_predictions(os.path.join(run_dir, 'preds_test.csv'), test_res)
    result = {'variant': 'R0', 'seed': None, 'checkpoint': args.load_model_state_dir, 'selected_epoch': None,
              'valid': compute_metrics(valid_res['labels'], valid_res['preds']),
              'test': compute_metrics(test_res['labels'], test_res['preds'])}
    write_json(os.path.join(run_dir, 'metrics.json'), result)
    print('Metrics in %:')
    print_result('valid', result['valid'])
    print_result('test', result['test'])
    print('Saved to {}'.format(run_dir))


def main():
    args = parse_args()
    if args.dry_run:
        dry_run(args)
    elif not torch.cuda.is_available():
        sys.exit('No CUDA GPU found. MELD/model.py only runs on a GPU (its attention masks are built inside '
                 '"if self.cuda_flag"), so training and --eval_only need one; --dry_run works without.')
    elif args.eval_only:
        evaluate_checkpoint(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
