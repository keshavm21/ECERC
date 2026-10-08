"""Checks and summaries for the runs written by train_imbalance.py (standard library; matplotlib for PNGs).

    python analyze_imbalance.py compare results/orig_train_s0.log results/B0_s0
    python analyze_imbalance.py summary --split valid results/*_s0
    python analyze_imbalance.py summary --split test results/R0_ECERC_MODEL results/B0_s* results/CB0.999_s*

compare  checks epoch by epoch that two runs logged identical numbers. Each side is a MELD/train.py
         log (saved with `tee`) or a run folder.
summary  reads only the chosen split from each run's metrics.json and writes to results/summary_<split>/:
         per-run and per-variant tables (mean ± std over seeds), Fear/Disgust precision/recall/F1,
         per-seed differences from B0, and confusion matrices summed over runs (CSV and PNG).
         With --split valid it also applies the pilot rule that picks beta (CB) and rho (OS).
"""
import argparse
import csv
import json
import math
import os
import re
import statistics
import sys

CLASS_NAMES = ['neu', 'sur', 'fea', 'sad', 'joy', 'dis', 'ang']
FULL_NAMES = ['Neutral', 'Surprise', 'Fear', 'Sadness', 'Joy', 'Disgust', 'Anger']
EPOCH_FIELDS = ['train_loss', 'train_acc', 'train_fscore', 'valid_loss', 'valid_acc', 'valid_fscore',
                'test_loss', 'test_acc', 'test_fscore']
EPOCH_LINE = re.compile(r'^epoch: (\d+), ' + ', '.join(f + r': ([^,\s]+)' for f in EPOCH_FIELDS) + r', time: ')
PILOT_TOLERANCE = 0.5  # validation macro-F1 points; within this of the best, the milder setting is chosen

# Sequential blue ramp (light -> dark) and ink colours for the confusion-matrix figures.
SEQUENTIAL = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
              '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
SURFACE, INK, INK_SECONDARY, INK_MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'


# ---------------------------------------------------------------- compare

def load_epochs(path):
    """Per-epoch numbers from a run folder (epochs.csv) or from a MELD/train.py log."""
    rows = []
    if os.path.isdir(path):
        with open(os.path.join(path, 'epochs.csv'), newline='') as f:
            for r in csv.DictReader(f):
                rows.append(dict({'epoch': int(r['epoch'])}, **{k: float(r[k]) for k in EPOCH_FIELDS}))
    else:
        with open(path) as f:
            for line in f:
                m = EPOCH_LINE.match(line.strip())
                if m:
                    rows.append(dict({'epoch': int(m.group(1))},
                                     **{k: float(v) for k, v in zip(EPOCH_FIELDS, m.groups()[1:])}))
    if not rows:
        sys.exit('no epoch lines found in {}'.format(path))
    return rows


def same(a, b):
    return a == b or (math.isnan(a) and math.isnan(b))


def selected_epochs(rows):
    """The epochs MELD/train.py reports: best validation weighted F1 and lowest validation loss (first occurrence)."""
    by_f1 = by_loss = rows[0]
    for r in rows[1:]:
        if by_f1['valid_fscore'] < r['valid_fscore']:
            by_f1 = r
        if r['valid_loss'] < by_loss['valid_loss']:
            by_loss = r
    return by_f1, by_loss


def compare(path_a, path_b):
    a, b = load_epochs(path_a), load_epochs(path_b)
    print('A: {} ({} epochs)'.format(path_a, len(a)))
    print('B: {} ({} epochs)'.format(path_b, len(b)))
    n = min(len(a), len(b))
    identical = len(a) == len(b)
    print('{:<14}{:>18}{:>12}'.format('field', 'epochs differing', 'max |A-B|'))
    for field in ['epoch'] + EPOCH_FIELDS:
        diffs = [abs(x[field] - y[field]) for x, y in zip(a[:n], b[:n]) if not same(x[field], y[field])]
        identical = identical and not diffs
        print('{:<14}{:>18}{:>12.4g}'.format(field, len(diffs), max(diffs) if diffs else 0.0))
    for label, rows in (('A', a), ('B', b)):
        f1_row, loss_row = selected_epochs(rows)
        print('{}: best valid wF1 at epoch {} -> test acc {} / wF1 {}; lowest valid loss at epoch {} -> '
              'test acc {} / wF1 {}'.format(label, f1_row['epoch'], f1_row['test_acc'], f1_row['test_fscore'],
                                            loss_row['epoch'], loss_row['test_acc'], loss_row['test_fscore']))
    if identical:
        print('RESULT: IDENTICAL. All {} epochs match exactly.'.format(n))
    elif len(a) != len(b):
        print('RESULT: DIFFERENT. The runs have different numbers of epochs.')
    else:
        print('RESULT: DIFFERENT. Compare with two runs of MELD/train.py to see its own run-to-run variation.')


# ---------------------------------------------------------------- summary

def load_runs(paths):
    runs = []
    for path in paths:
        if not os.path.isfile(os.path.join(path, 'metrics.json')):
            print('skipping {}: not a finished run (no metrics.json)'.format(path))
            continue
        with open(os.path.join(path, 'config.json')) as f:
            config = json.load(f)
        with open(os.path.join(path, 'metrics.json')) as f:
            result = json.load(f)
        runs.append({'name': os.path.basename(os.path.normpath(path)), 'variant': config['variant'],
                     'seed': config['seed'], 'config': config, 'result': result})
    if not runs:
        sys.exit('no finished runs given')
    return sorted(runs, key=lambda r: (variant_order(r['variant']), str(r['seed']), r['name']))


def variant_order(variant):
    return (variant != 'R0', variant != 'B0', variant)


def run_values(run, split):
    m = run['result'][split]
    v = {'acc': m['accuracy'], 'wf1': m['weighted_f1'], 'macro_f1': m['macro_f1']}
    for c in CLASS_NAMES:
        v['f1_' + c] = m['per_class'][c]['f1']
    for c in ('fea', 'dis'):
        for k in ('precision', 'recall', 'tp', 'fp', 'fn'):
            v['{}_{}'.format(c, k)] = m['per_class'][c][k]
    return v


def mean_std(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else None)


def fmt(xs, signed=False):
    m, s = mean_std(xs)
    text = ('{:+.2f}' if signed else '{:.2f}').format(m)
    return text if s is None else text + ' ± {:.2f}'.format(s)


def md_table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    return '\n'.join(lines + ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows])


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def consistency_warnings(runs, split):
    def distinct(values):
        return {json.dumps(v, sort_keys=True) for v in values}

    warnings = []
    if len(distinct(r['result'][split]['true_counts'] for r in runs)) > 1:
        warnings.append('runs were evaluated on different {} data (label counts differ)'.format(split))
    trained = [r for r in runs if r['config'].get('mode') == 'train']
    if len(distinct(r['config']['hyperparameters'] for r in trained)) > 1:
        warnings.append('trained runs use different hyperparameters (e.g. a smoke-test run with fewer epochs)')
    gpus = {r['config']['environment']['gpu'] for r in runs}
    if len(gpus) > 1:
        warnings.append('runs used different GPU types: {}'.format(', '.join(sorted(str(g) for g in gpus))))
    if len(distinct(r['config']['environment']['baseline_sha256'] for r in runs)) > 1:
        warnings.append('the original ECERC files differ between runs (baseline_sha256 in config.json)')
    names = {}
    for r in runs:
        if r['seed'] is not None:
            names.setdefault((r['variant'], r['seed']), []).append(r['name'])
    for (variant, seed), same_seed in names.items():
        if len(same_seed) > 1:
            warnings.append('{} has several runs for seed {} ({}); pass one run per seed'
                            .format(variant, seed, ', '.join(same_seed)))
    return warnings


def pilot_choices(runs, split):
    """The pre-declared pilot rule: highest mean validation macro-F1; within 0.5 points, the milder setting."""
    families = [('CB', 'beta', lambda c: c['loss']['cb_beta'] if c['loss']['cb_beta'] is not None
                 and c['loss']['gamma'] == 1 and not c['oversampling'] else None),
                ('OS', 'rho', lambda c: c['oversampling']['rho'] if c['oversampling']
                 and c['loss']['gamma'] == 1 and c['loss']['cb_beta'] is None else None)]
    lines = []
    for family, param, value_of in families:
        scores = {}
        for run in runs:
            if run['config'].get('mode') == 'train' and value_of(run['config']) is not None:
                scores.setdefault(value_of(run['config']), []).append(run['result'][split]['macro_f1'])
        if len(scores) < 2:
            continue
        means = {p: statistics.mean(v) for p, v in scores.items()}
        best = max(means.values())
        choice = min(p for p, m in means.items() if best - m <= PILOT_TOLERANCE)
        lines.append('{}: '.format(family) + ', '.join('{}={:g} -> {:.2f}'.format(param, p, means[p])
                                                       for p in sorted(means))
                     + '  =>  choose {}={:g}'.format(param, choice))
    return lines


def confusion_rows(cm):
    totals = [sum(row) for row in cm]
    percent = [[100.0 * v / t if t else 0.0 for v in row] for row, t in zip(cm, totals)]
    rows = [['count: ' + FULL_NAMES[i]] + cm[i] for i in range(len(cm))]
    rows += [['% of row: ' + FULL_NAMES[i]] + ['{:.2f}'.format(v) for v in percent[i]] for i in range(len(cm))]
    return percent, rows


def text_ink(rgb):
    """Primary ink or white, whichever contrasts more with the cell colour."""
    def luminance(colour):
        lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in colour[:3]]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]

    cell = luminance(rgb)
    ink = luminance([int(INK[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return '#ffffff' if 1.05 / (cell + 0.05) > (cell + 0.05) / (ink + 0.05) else INK


def plot_confusion(cm, percent, title, subtitle, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        return False
    n = len(cm)
    cmap = LinearSegmentedColormap.from_list('sequential_blue', SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    mesh = ax.pcolormesh(percent, cmap=cmap, vmin=0, vmax=100, edgecolors=SURFACE, linewidth=2)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)  # first class at the top
    ax.set_aspect('equal')
    for i in range(n):
        for j in range(n):
            ink = text_ink(cmap(percent[i][j] / 100.0))
            ax.text(j + 0.5, i + 0.42, '{:.1f}%'.format(percent[i][j]), ha='center', va='center', fontsize=7.5,
                    color=ink)
            ax.text(j + 0.5, i + 0.72, str(cm[i][j]), ha='center', va='center', fontsize=6, color=ink)
    ax.set_xticks([k + 0.5 for k in range(n)])
    ax.set_yticks([k + 0.5 for k in range(n)])
    ax.set_xticklabels(FULL_NAMES, fontsize=8)
    ax.set_yticklabels(FULL_NAMES, fontsize=8)
    ax.tick_params(length=0, colors=INK_SECONDARY)
    ax.set_xlabel('Predicted label', fontsize=8.5, color=INK_SECONDARY)
    ax.set_ylabel('True label', fontsize=8.5, color=INK_SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03, ticks=[0, 25, 50, 75, 100])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=7, colors=INK_MUTED)
    bar.set_label('% of true class', fontsize=8, color=INK_SECONDARY)
    ax.set_title(title, loc='left', fontsize=10.5, color=INK, pad=20)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=7, color=INK_SECONDARY, va='bottom')
    fig.savefig(path, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    return True


def summary(paths, split, out_dir):
    runs = load_runs(paths)
    out_dir = out_dir or os.path.join('results', 'summary_' + split)
    os.makedirs(out_dir, exist_ok=True)
    groups = {}
    for run in runs:
        groups.setdefault(run['variant'], []).append(run)
    variants = sorted(groups, key=variant_order)
    values = {run['name']: run_values(run, split) for run in runs}
    sections = ['# Person B results: {} split'.format(split),
                'All metrics in %. Runs are scored at the epoch with the best validation weighted F1 '
                '(R0: the released checkpoint).']

    warnings = consistency_warnings(runs, split)
    if warnings:
        sections.append('**Warnings**\n\n' + '\n'.join('- ' + w for w in warnings))

    header = ['run', 'variant', 'seed', 'epoch', 'Acc', 'wF1', 'macro-F1', 'Fear P', 'Fear R', 'Fear F1',
              'Disgust P', 'Disgust R', 'Disgust F1']
    rows = []
    for run in runs:
        v = values[run['name']]
        seed = '-' if run['seed'] is None else run['seed']
        epoch = run['result'].get('selected_epoch')
        rows.append([run['name'], run['variant'], seed, '-' if epoch is None else epoch]
                    + ['{:.2f}'.format(v[k]) for k in ('acc', 'wf1', 'macro_f1', 'fea_precision', 'fea_recall',
                                                       'f1_fea', 'dis_precision', 'dis_recall', 'f1_dis')])
    write_csv(os.path.join(out_dir, 'per_run.csv'), header, rows)
    sections.append('## Per run\n\n' + md_table(header, rows))

    header = ['variant', 'runs', 'Acc', 'wF1', 'macro-F1'] + ['F1 ' + c for c in FULL_NAMES]
    rows = [[variant, len(groups[variant])]
            + [fmt([values[r['name']][k] for r in groups[variant]])
               for k in ['acc', 'wf1', 'macro_f1'] + ['f1_' + c for c in CLASS_NAMES]] for variant in variants]
    write_csv(os.path.join(out_dir, 'per_variant.csv'), header, rows)
    sections.append('## Per variant (mean ± std over runs)\n\n' + md_table(header, rows))

    header = ['variant', 'runs', 'Fear P', 'Fear R', 'Fear F1', 'Fear TP/FP/FN', 'Disgust P', 'Disgust R',
              'Disgust F1', 'Disgust TP/FP/FN']
    rows = []
    for variant in variants:
        vs = [values[r['name']] for r in groups[variant]]
        row = [variant, len(vs)]
        for c in ('fea', 'dis'):
            row += [fmt([v[c + '_precision'] for v in vs]), fmt([v[c + '_recall'] for v in vs]),
                    fmt([v['f1_' + c] for v in vs]),
                    '/'.join('{:.1f}'.format(statistics.mean(v[c + '_' + k] for v in vs)) for k in ('tp', 'fp', 'fn'))]
        rows.append(row)
    write_csv(os.path.join(out_dir, 'fear_disgust.csv'), header, rows)
    sections.append('## Fear and Disgust (mean ± std over runs; TP/FP/FN are means per run)\n\n'
                    + md_table(header, rows))

    if 'B0' in groups:
        b0 = {r['seed']: values[r['name']] for r in groups['B0']}
        keys = [('acc', 'ΔAcc'), ('wf1', 'ΔwF1'), ('macro_f1', 'Δmacro-F1'), ('f1_fea', 'ΔFear F1'),
                ('f1_dis', 'ΔDisgust F1')]
        header = ['variant', 'paired seeds'] + [label for _, label in keys]
        rows = []
        for variant in variants:
            if variant in ('R0', 'B0'):
                continue
            pairs = [(values[r['name']], b0[r['seed']]) for r in groups[variant] if r['seed'] in b0]
            if not pairs:
                continue
            row = [variant, len(pairs)]
            for k, _ in keys:
                deltas = [v[k] - base[k] for v, base in pairs]
                row.append('{} ({}/{} up)'.format(fmt(deltas, signed=True), sum(d > 0 for d in deltas), len(deltas)))
            rows.append(row)
        if rows:
            write_csv(os.path.join(out_dir, 'paired_vs_B0.csv'), header, rows)
            sections.append('## Difference from B0, paired by seed (mean ± std; seeds where the variant is higher)'
                            '\n\n' + md_table(header, rows))

    lines = []
    for variant in variants:
        cm = [[sum(r['result'][split]['confusion_matrix'][i][j] for r in groups[variant])
               for j in range(len(CLASS_NAMES))] for i in range(len(CLASS_NAMES))]
        percent, rows = confusion_rows(cm)
        stem = 'confusion_' + variant
        write_csv(os.path.join(out_dir, stem + '.csv'), ['true \\ predicted'] + FULL_NAMES, rows)
        n = len(groups[variant])
        plotted = plot_confusion(cm, percent, '{}: {} confusion matrix'.format(variant, split),
                                 'Cells: % of the true class, with counts summed over {} run{}.'
                                 .format(n, '' if n == 1 else 's'), os.path.join(out_dir, stem + '.png'))
        lines.append('- {}: {}.csv{}'.format(variant, stem, ', {}.png'.format(stem) if plotted else ''))
    sections.append('## Confusion matrices (rows: true label; summed over runs)\n\n' + '\n'.join(lines))

    if split == 'valid':
        choices = pilot_choices(runs, split)
        if choices:
            sections.append('## Pilot choice (highest validation macro-F1; within {:g} points, the milder setting)'
                            '\n\n'.format(PILOT_TOLERANCE) + '\n'.join('- ' + c for c in choices))

    report = '\n\n'.join(sections) + '\n'
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write(report)
    print(report)
    print('Written to {}'.format(out_dir))


def main():
    parser = argparse.ArgumentParser(description='Checks and summaries for train_imbalance.py runs.')
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('compare', help='check that two runs logged identical per-epoch numbers')
    p.add_argument('a', help='a MELD/train.py log or a run folder')
    p.add_argument('b', help='a MELD/train.py log or a run folder')
    p = commands.add_parser('summary', help='tables and confusion matrices for one split')
    p.add_argument('runs', nargs='+', help='run folders')
    p.add_argument('--split', choices=['valid', 'test'], required=True, help='the only split that is read')
    p.add_argument('--out', default=None, help='output folder (default: results/summary_<split>)')
    args = parser.parse_args()
    if args.command == 'compare':
        compare(args.a, args.b)
    else:
        summary(args.runs, args.split, args.out)


if __name__ == '__main__':
    main()
