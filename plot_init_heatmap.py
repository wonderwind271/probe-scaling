"""Heatmap of average MDL (over seeds and layers) for gamma_second_zero.

Axes: gamma (rows) x width (columns).
Width values follow the 2^{n/2} schedule for n = 4..22, rounded to integers.
Gamma values: 0.3, 0.4, ..., 2.0 (step 0.1).

Output: figures/init_heatmap_mdl.png
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import os
import matplotlib
matplotlib.use('Agg')

BASE = 'init_results/llama/somo'
SEEDS = [42, 142, 242, 342, 442]

# 2^{n/2} rounded for n = 4..22
WIDTHS = [round(2 ** (n / 2)) for n in range(4, 23)]
GAMMAS = [g/10 for g in range(3, 21)]  # 0.3 to 2.0 in steps of 0.1

os.makedirs('figures', exist_ok=True)


def load_mdl() -> dict:
    """Return {(width, gamma, seed): avg_mdl_over_layers}.

    Merges old format (no opt-adamw subdir) and new format (with opt-adamw).
    Latest timestamp wins for duplicates.
    """
    # (width, gamma, seed) -> {timestamp: avg_mdl}
    best: dict = {}

    def _record(f):
        parts = f.split('/')
        width = int(parts[3].replace('hidden-', ''))
        seed = int(parts[4].replace('seed-', ''))
        gamma = float(parts[6].replace('gamma-', ''))
        ts = parts[-2]
        key = (width, gamma, seed)
        if key in best and ts <= best[key][0]:
            print(f'  Skipping {f}, (timestamp {ts} <= {best[key][0]})')
            return
        with open(f) as fh:
            r = json.load(fh)
        avg_mdl = float(np.mean(list(r['mdl_sum_ce'].values())))
        best[key] = (ts, avg_mdl)

    for f in glob.glob(f'{BASE}/hidden-*/seed-*/strategy-gamma_second_zero/gamma-*/opt-adamw/*/results.json'):
        _record(f)

    for f in glob.glob(f'{BASE}/hidden-*/seed-*/strategy-gamma_second_zero/gamma-*/*/results.json'):
        if f.split('/')[7] == 'opt-adamw':
            continue
        _record(f)

    return {k: v[1] for k, v in best.items()}


def build_matrix(data: dict) -> np.ndarray:
    """Shape: (len(GAMMAS), len(WIDTHS)); NaN where seeds are missing."""
    mat = np.full((len(GAMMAS), len(WIDTHS)), np.nan)
    for i, g in enumerate(GAMMAS):
        for j, w in enumerate(WIDTHS):
            vals = [data[(w, g, s)] for s in SEEDS if (w, g, s) in data]
            assert len(vals) == len(SEEDS), f'Missing seeds for (width={w}, gamma={g})'
            mat[i, j] = np.mean(vals)
    return mat


def _width_xticklabels():
    labels = []
    for n in range(4, 23):
        if n % 2 == 0:
            labels.append(f'$2^{{{n // 2}}}$')
        else:
            labels.append(f'$2^{{{n}/2}}$')
    return labels


def _draw_heatmap(mat: np.ndarray, highlight: np.ndarray, out: str) -> None:
    """Draw one heatmap; highlighted cells use bold black text, others adaptive."""
    fig, ax = plt.subplots(figsize=(max(8, len(WIDTHS) * 0.7 + 1.5), 6))

    im = ax.imshow(mat, aspect='auto', cmap='viridis_r')
    plt.colorbar(im, ax=ax, label='MDL sum CE (avg over layers & seeds)')

    ax.set_xticks(range(len(WIDTHS)))
    ax.set_xticklabels(_width_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(GAMMAS)))
    ax.set_yticklabels([f'{g:.1f}' for g in GAMMAS], fontsize=9)
    ax.set_xlabel('Hidden width', fontsize=11)
    ax.set_ylabel('γ', fontsize=11)

    vmin, vmax = np.nanmin(mat), np.nanmax(mat)
    for i in range(len(GAMMAS)):
        for j in range(len(WIDTHS)):
            v = mat[i, j]
            if np.isnan(v):
                continue
            is_best = bool(highlight[i, j])
            brightness = (v - vmin) / (vmax - vmin + 1e-9)
            if is_best:
                color, weight = 'black', 'bold'
            else:
                color = 'white' if brightness < 0.5 else 'black'
                weight = 'normal'
            ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight=weight)

    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


def plot_heatmaps(mat: np.ndarray) -> None:
    best_row = np.nanargmin(mat, axis=0)  # best gamma per width column
    best_col = np.nanargmin(mat, axis=1)  # best width per gamma row

    # mask: best gamma for each width
    hl_gamma = np.zeros(mat.shape, dtype=bool)
    for j, i in enumerate(best_row):
        hl_gamma[i, j] = True

    # mask: best width for each gamma
    hl_width = np.zeros(mat.shape, dtype=bool)
    for i, j in enumerate(best_col):
        hl_width[i, j] = True

    _draw_heatmap(mat, hl_gamma, 'figures/init_heatmap_best_gamma.pdf')
    _draw_heatmap(mat, hl_width, 'figures/init_heatmap_best_width.pdf')


if __name__ == '__main__':
    print('Loading MDL results...')
    data = load_mdl()
    print(f'  Loaded {len(data)} (width, gamma, seed) entries')

    mat = build_matrix(data)
    n_missing = np.sum(np.isnan(mat))
    if n_missing:
        print(f'  WARNING: {int(n_missing)} cells have no data')
    else:
        print('  All cells populated')

    print('Plotting heatmaps...')
    plot_heatmaps(mat)
