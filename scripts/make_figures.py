"""Summary figures for the Stage 2 MELD class-imbalance experiment.

Writes three PNGs to docs/figures/:

    fig1_test_overall.png          test accuracy, weighted F1 and macro-F1 per configuration (3 seeds each)
    fig2_minority_pr.png           test precision vs recall for Fear and Disgust, with iso-F1 curves
    fig3_valid_vs_test_delta.png   per-seed difference from B0 on validation and on test

Two input layouts are accepted under --results (default: results):

    stage2/summary_test/, stage2/summary_valid/   the committed analyze_imbalance.py summaries. Per-seed dots
                                                  come from per_run.csv (2 decimals); every plotted mean is read
                                                  from per_variant.csv, fear_disgust.csv and paired_vs_B0.csv,
                                                  so the labels match the report tables exactly.
    stage2/runs/<run>/metrics.json                the 15 per-run files written by MELD/train_imbalance.py
                                                  (unrounded). Used when this folder exists.

    python scripts/make_figures.py                     # from the repository root
    python scripts/make_figures.py --results results --out docs/figures

Only matplotlib is needed. No number is computed here that is not also in the summary tables.
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

VARIANTS = ['B0', 'CB0.999', 'FL2', 'OS2', 'FL2-OS2']
SEEDS = [0, 1, 2]

# Reference palette (light mode): slots 1-2 validated all-pairs; chrome and ink tokens.
SERIES_1, SERIES_2 = '#2a78d6', '#eb6834'
SURFACE, INK, INK_SECONDARY, INK_MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
GRID, AXIS = '#e1e0d9', '#c3c2b7'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8.5,
    'axes.edgecolor': AXIS, 'axes.linewidth': 1, 'axes.labelcolor': INK_SECONDARY,
    'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED, 'xtick.labelcolor': INK_SECONDARY,
    'ytick.labelcolor': INK_SECONDARY, 'xtick.major.size': 0, 'ytick.major.size': 0,
    'grid.color': GRID, 'grid.linewidth': 1, 'grid.linestyle': '-',
})


# ---------------------------------------------------------------- data

KEYS = ['acc', 'wf1', 'macro_f1'] + ['{}_{}'.format(c, k) for c in ('fea', 'dis') for k in ('precision', 'recall', 'f1')]
# Column names used by MELD/analyze_imbalance.py summary.
PER_RUN_COLUMNS = {'acc': 'Acc', 'wf1': 'wF1', 'macro_f1': 'macro-F1',
                   'fea_precision': 'Fear P', 'fea_recall': 'Fear R', 'fea_f1': 'Fear F1',
                   'dis_precision': 'Disgust P', 'dis_recall': 'Disgust R', 'dis_f1': 'Disgust F1'}
PAIRED_COLUMNS = {'acc': 'ΔAcc', 'wf1': 'ΔwF1', 'macro_f1': 'Δmacro-F1', 'fea_f1': 'ΔFear F1', 'dis_f1': 'ΔDisgust F1'}


def load_data(results_dir):
    """(runs, means, paired):
    runs[(variant, seed)][split][key] per-seed values;
    means[(split, variant, key)] mean over seeds; paired[(split, variant, key)] mean per-seed difference from B0."""
    if os.path.isdir(os.path.join(results_dir, 'stage2', 'runs')):
        return load_run_files(results_dir)
    return load_summaries(results_dir)


def check_complete(runs, source):
    missing = [(v, s) for v in VARIANTS for s in SEEDS if (v, s) not in runs]
    if missing:
        sys.exit('missing Stage 2 runs in {}: {}'.format(
            source, ', '.join('{}_s{}'.format(v, s) for v, s in missing)))


def load_run_files(results_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, 'stage2', 'runs', '*', 'metrics.json'))):
        with open(path) as f:
            m = json.load(f)
        if m.get('variant') in VARIANTS:
            runs[(m['variant'], m['seed'])] = {split: flatten(m[split]) for split in ('valid', 'test')}
    check_complete(runs, os.path.join(results_dir, 'stage2', 'runs'))
    means, paired = {}, {}
    for split in ('valid', 'test'):
        for v in VARIANTS:
            for key in KEYS:
                means[(split, v, key)] = statistics.mean(values(runs, v, split, key))
                paired[(split, v, key)] = statistics.mean(paired_deltas(runs, v, split, key))
    return runs, means, paired


def flatten(split_metrics):
    out = {'acc': split_metrics['accuracy'], 'wf1': split_metrics['weighted_f1'],
           'macro_f1': split_metrics['macro_f1']}
    for c in ('fea', 'dis'):
        for k in ('precision', 'recall', 'f1'):
            out['{}_{}'.format(c, k)] = split_metrics['per_class'][c][k]
    return out


def read_csv(path):
    if not os.path.isfile(path):
        sys.exit('missing {} (expected an analyze_imbalance.py summary folder)'.format(path))
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def leading_number(cell):
    """'66.87 ± 0.65' -> 66.87; '-0.64 ± 0.59 (1/3 up)' -> -0.64."""
    return float(cell.split()[0].replace('−', '-'))


def load_summaries(results_dir):
    runs, means, paired = {}, {}, {}
    for split in ('valid', 'test'):
        folder = os.path.join(results_dir, 'stage2', 'summary_' + split)
        for row in read_csv(os.path.join(folder, 'per_run.csv')):
            if row['variant'] in VARIANTS:
                runs.setdefault((row['variant'], int(row['seed'])), {})[split] = {
                    key: float(row[col]) for key, col in PER_RUN_COLUMNS.items()}
        for row in read_csv(os.path.join(folder, 'per_variant.csv')):
            for key, col in (('acc', 'Acc'), ('wf1', 'wF1'), ('macro_f1', 'macro-F1')):
                means[(split, row['variant'], key)] = leading_number(row[col])
        for row in read_csv(os.path.join(folder, 'fear_disgust.csv')):
            for key, col in PER_RUN_COLUMNS.items():
                if key.startswith(('fea_', 'dis_')):
                    means[(split, row['variant'], key)] = leading_number(row[col])
        for row in read_csv(os.path.join(folder, 'paired_vs_B0.csv')):
            for key, col in PAIRED_COLUMNS.items():
                paired[(split, row['variant'], key)] = leading_number(row[col])
    runs = {k: r for k, r in runs.items() if len(r) == 2}
    check_complete(runs, os.path.join(results_dir, 'stage2', 'summary_{valid,test}', 'per_run.csv'))
    return runs, means, paired


def values(runs, variant, split, key):
    return [runs[(variant, s)][split][key] for s in SEEDS]


def paired_deltas(runs, variant, split, key):
    return [runs[(variant, s)][split][key] - runs[('B0', s)][split][key] for s in SEEDS]


# ---------------------------------------------------------------- shared chrome

def style_axes(ax, grid_axis='y'):
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)


def title(fig, text, subtitle):
    fig.text(0.012, 0.985, text, ha='left', va='top', fontsize=11, color=INK, weight='semibold')
    fig.text(0.012, 0.925, subtitle, ha='left', va='top', fontsize=8, color=INK_SECONDARY)


def dot(ax, x, y, color, size=34, **kw):
    ax.scatter(x, y, s=size, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3, **kw)


# ---------------------------------------------------------------- figure 1

def fig_overall(runs, means, paired, path):
    metrics = [('acc', 'Accuracy'), ('wf1', 'Weighted F1'), ('macro_f1', 'Macro-F1')]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (key, label) in zip(axes, metrics):
        b0_mean = means[('test', 'B0', key)]
        ax.axhline(b0_mean, color=INK_MUTED, linewidth=1, zorder=1)
        ax.text(len(VARIANTS) - 0.45, b0_mean, 'B0 mean', ha='right', va='bottom', fontsize=7, color=INK_MUTED)
        for i, v in enumerate(VARIANTS):
            ys = values(runs, v, 'test', key)
            dot(ax, [i] * len(ys), ys, SERIES_1)
            mean = means[('test', v, key)]
            ax.plot([i - 0.22, i + 0.22], [mean, mean], color=INK, linewidth=2, solid_capstyle='round', zorder=4)
            ax.text(i + 0.27, mean, '{:.2f}'.format(mean), va='center', fontsize=7, color=INK_SECONDARY)
        ax.set_xticks(range(len(VARIANTS)))
        ax.set_xticklabels(VARIANTS)
        ax.set_xlim(-0.5, len(VARIANTS) - 0.3)
        ax.set_title(label + ' (%)', loc='left', fontsize=9, color=INK)
        style_axes(ax)
    title(fig, 'Stage 2 test results: no configuration has a higher 3-seed mean than B0',
          'Dots: individual seeds (0, 1, 2). Black ticks: mean of 3 seeds. Grey line: B0 mean. '
          'Each panel has its own y-scale.')
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2

# Label placement (offset points) where two configurations' means sit close together.
LABEL_OFFSETS = {('dis', 'FL2'): (-6, -11, 'right'), ('dis', 'FL2-OS2'): (-6, -11, 'right')}

def fig_minority_pr(runs, means, paired, path):
    colors = {'B0': SERIES_1, 'CB0.999': SERIES_2}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, (cls, name, support) in zip(axes, [('fea', 'Fear', 50), ('dis', 'Disgust', 68)]):
        pts = {v: (values(runs, v, 'test', cls + '_recall'), values(runs, v, 'test', cls + '_precision'))
               for v in VARIANTS}
        all_r = [r for rs, _ in pts.values() for r in rs]
        all_p = [p for _, ps in pts.values() for p in ps]
        lo = 5 * int(min(all_r + all_p) // 5) - 5
        hi = 5 * int(max(all_r + all_p) // 5) + 10
        # Iso-F1 curves: precision = F1 * R / (2R - F1).
        for f1 in range(10, 60, 5):
            rs = [r / 10 for r in range(int(10 * max(lo, f1 / 2 + 0.5)), 10 * hi + 1)]
            curve = [(r, f1 * r / (2 * r - f1)) for r in rs if 2 * r > f1]
            curve = [(r, p) for r, p in curve if lo <= p <= hi]
            if len(curve) < 2:
                continue
            ax.plot(*zip(*curve), color=GRID, linewidth=1, zorder=1)
            r_end, p_end = curve[-1] if curve[-1][0] >= hi - 0.5 else curve[0]
            ax.text(r_end, p_end, ' F1 {}'.format(f1), fontsize=6.5, color=INK_MUTED, va='bottom',
                    ha='right' if r_end >= hi - 0.5 else 'left')
        for v in reversed(VARIANTS):
            color = colors.get(v, INK_MUTED)
            rs, ps = pts[v]
            ax.scatter(rs, ps, s=14, color=color, alpha=0.45, linewidths=0, zorder=2)
            mr, mp = means[('test', v, cls + '_recall')], means[('test', v, cls + '_precision')]
            dot(ax, mr, mp, color, size=60 if v in colors else 40)
            dx, dy, ha = LABEL_OFFSETS.get((cls, v), (6, 4, 'left'))
            ax.annotate(v, (mr, mp), xytext=(dx, dy), textcoords='offset points', fontsize=7.5, ha=ha,
                        color=INK if v in colors else INK_SECONDARY)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_xlabel('Recall (%)')
        ax.set_ylabel('Precision (%)')
        ax.set_title('{} (test support {})'.format(name, support), loc='left', fontsize=9, color=INK)
        ax.grid(False)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    handles = [plt.Line2D([], [], marker='o', linestyle='', markersize=7, color=c, label=l)
               for c, l in [(SERIES_1, 'B0'), (SERIES_2, 'CB0.999'), (INK_MUTED, 'FL2, OS2, FL2-OS2')]]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.995), ncol=3, frameon=False,
               fontsize=7.5, labelcolor=INK_SECONDARY)
    title(fig, 'Fear and Disgust on test: class-balanced weighting trades precision for recall',
          'Large dots: mean of 3 seeds. Small dots: individual seeds. Grey curves: constant F1.')
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3

def fig_valid_vs_test(runs, means, paired, path):
    variants = VARIANTS[1:]
    metrics = [('wf1', 'Δ weighted F1 vs B0 (points)'), ('macro_f1', 'Δ macro-F1 vs B0 (points)')]
    splits = [('valid', 'Validation', SERIES_1, -0.14), ('test', 'Test', SERIES_2, 0.14)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (key, label) in zip(axes, metrics):
        ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=1)
        for row, v in enumerate(variants):
            for split, _, color, offset in splits:
                ds = paired_deltas(runs, v, split, key)
                y = row + offset
                ax.scatter(ds, [y] * len(ds), s=14, color=color, alpha=0.45, linewidths=0, zorder=2)
                mean = paired[(split, v, key)]
                dot(ax, mean, y, color, size=50)
                ax.text(mean, y - 0.13 if offset < 0 else y + 0.13, '{:+.2f}'.format(mean), fontsize=6.5,
                        color=INK_SECONDARY, ha='center', va='bottom' if offset < 0 else 'top')
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(variants)
        ax.set_ylim(len(variants) - 0.5, -0.6)
        ax.set_xlabel(label)
        style_axes(ax, grid_axis='x')
    handles = [plt.Line2D([], [], marker='o', linestyle='', markersize=7, color=c, label=l)
               for _, l, c, _ in splits]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 0.995), ncol=2, frameon=False,
               fontsize=7.5, labelcolor=INK_SECONDARY)
    title(fig, 'Small validation gains did not carry over to the test set',
          'Difference from B0 with the same seed. Large dots: mean of 3 seeds; small dots: individual seeds.')
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Summary figures for the Stage 2 experiment.')
    parser.add_argument('--results', default='results', help='curated results folder (default: results)')
    parser.add_argument('--out', default=os.path.join('docs', 'figures'), help='output folder')
    args = parser.parse_args()
    runs, means, paired = load_data(args.results)
    os.makedirs(args.out, exist_ok=True)
    for name, fn in [('fig1_test_overall.png', fig_overall), ('fig2_minority_pr.png', fig_minority_pr),
                     ('fig3_valid_vs_test_delta.png', fig_valid_vs_test)]:
        fn(runs, means, paired, os.path.join(args.out, name))
        print('wrote', os.path.join(args.out, name))


if __name__ == '__main__':
    main()
