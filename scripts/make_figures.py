"""Summary figures for the Stage 2 MELD class-imbalance experiment.

Reads the 15 Stage 2 runs from results/stage2/runs/*/metrics.json (the unrounded metrics written by
MELD/train_imbalance.py) and writes three PNGs to docs/figures/:

    fig1_test_overall.png          test accuracy, weighted F1 and macro-F1 per configuration (3 seeds each)
    fig2_minority_pr.png           test precision vs recall for Fear and Disgust, with iso-F1 curves
    fig3_valid_vs_test_delta.png   per-seed difference from B0 on validation and on test

    python scripts/make_figures.py                     # from the repository root
    python scripts/make_figures.py --results results --out docs/figures

Only matplotlib is needed. No number is computed here that is not also in the summary tables.
"""
import argparse
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

def load_runs(results_dir):
    """{(variant, seed): {'valid': {...}, 'test': {...}}} for the Stage 2 runs."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, 'stage2', 'runs', '*', 'metrics.json'))):
        with open(path) as f:
            m = json.load(f)
        if m.get('variant') in VARIANTS:
            runs[(m['variant'], m['seed'])] = {split: flatten(m[split]) for split in ('valid', 'test')}
    missing = [(v, s) for v in VARIANTS for s in SEEDS if (v, s) not in runs]
    if missing:
        sys.exit('missing Stage 2 runs: {}'.format(', '.join('{}_s{}'.format(v, s) for v, s in missing)))
    return runs


def flatten(split_metrics):
    out = {'acc': split_metrics['accuracy'], 'wf1': split_metrics['weighted_f1'],
           'macro_f1': split_metrics['macro_f1']}
    for c in ('fea', 'dis'):
        for k in ('precision', 'recall', 'f1'):
            out['{}_{}'.format(c, k)] = split_metrics['per_class'][c][k]
    return out


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

def fig_overall(runs, path):
    metrics = [('acc', 'Accuracy'), ('wf1', 'Weighted F1'), ('macro_f1', 'Macro-F1')]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (key, label) in zip(axes, metrics):
        b0_mean = statistics.mean(values(runs, 'B0', 'test', key))
        ax.axhline(b0_mean, color=INK_MUTED, linewidth=1, zorder=1)
        ax.text(len(VARIANTS) - 0.45, b0_mean, 'B0 mean', ha='right', va='bottom', fontsize=7, color=INK_MUTED)
        for i, v in enumerate(VARIANTS):
            ys = values(runs, v, 'test', key)
            dot(ax, [i] * len(ys), ys, SERIES_1)
            mean = statistics.mean(ys)
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

def fig_minority_pr(runs, path):
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
            mr, mp = statistics.mean(rs), statistics.mean(ps)
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

def fig_valid_vs_test(runs, path):
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
                mean = statistics.mean(ds)
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
    runs = load_runs(args.results)
    os.makedirs(args.out, exist_ok=True)
    for name, fn in [('fig1_test_overall.png', fig_overall), ('fig2_minority_pr.png', fig_minority_pr),
                     ('fig3_valid_vs_test_delta.png', fig_valid_vs_test)]:
        fn(runs, os.path.join(args.out, name))
        print('wrote', os.path.join(args.out, name))


if __name__ == '__main__':
    main()
