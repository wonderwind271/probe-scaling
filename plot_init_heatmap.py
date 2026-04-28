"""Heatmap + best-gamma regression for gamma_second_zero and gamma strategies.

For each strategy:
  1. Completeness check (50 widths x 18 gammas x 5 seeds).
  2. Heatmap of avg MDL over seeds and layers (gamma rows x width columns).
     Best gamma per width highlighted; best width per gamma highlighted.
  3. Linear regression: best_gamma ~ log2(width), with scatter + fit plot.

Outputs saved to figures/.
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')

BASE = 'init_results/llama/somo'
SEEDS = [42, 142, 242, 342, 442]
GAMMAS = [round(g / 10, 1) for g in range(3, 21)]  # 0.3 .. 2.0

os.makedirs('figures', exist_ok=True)


# ── data loading ──────────────────────────────────────────────────────────────

def load_strategy(strategy: str) -> tuple[list[int], dict]:
    """Return (sorted_widths, data) where data maps (width, gamma, seed) -> avg_mdl."""
    best: dict = {}

    def _record(f: str) -> None:
        parts = f.split('/')
        width = int(parts[3].replace('hidden-', ''))
        seed = int(parts[4].replace('seed-', ''))
        gamma = float(parts[6].replace('gamma-', ''))
        ts = parts[-2]
        key = (width, gamma, seed)
        if key in best and ts <= best[key][0]:
            return
        with open(f) as fh:
            r = json.load(fh)
        avg_mdl = float(np.mean(list(r['mdl_sum_ce'].values())))
        best[key] = (ts, avg_mdl)

    for f in glob.glob(
            f'{BASE}/hidden-*/seed-*/strategy-{strategy}/gamma-*/opt-adamw/*/results.json'):
        _record(f)
    for f in glob.glob(
            f'{BASE}/hidden-*/seed-*/strategy-{strategy}/gamma-*/*/results.json'):
        if f.split('/')[7] == 'opt-adamw':
            continue
        _record(f)

    data = {k: v[1] for k, v in best.items()}
    widths = sorted(set(w for w, g, s in data))
    return widths, data


def check_completeness(strategy: str, widths: list[int], data: dict) -> None:
    missing = []
    for w in widths:
        for g in GAMMAS:
            miss = [s for s in SEEDS if (w, g, s) not in data]
            if miss:
                missing.append((w, g, miss))
    if missing:
        print(f'  [INCOMPLETE] {len(missing)} (width, gamma) pairs missing seeds:')
        for w, g, ms in missing[:10]:
            print(f'    width={w}  gamma={g}  missing={ms}')
    else:
        total = len(widths) * len(GAMMAS) * len(SEEDS)
        print(f'  [COMPLETE] {total} results for {strategy}')


def build_matrix(widths: list[int], data: dict) -> np.ndarray:
    """Shape: (len(GAMMAS), len(widths)); NaN where any seed is missing."""
    mat = np.full((len(GAMMAS), len(widths)), np.nan)
    for i, g in enumerate(GAMMAS):
        for j, w in enumerate(widths):
            vals = [data[(w, g, s)] for s in SEEDS if (w, g, s) in data]
            assert len(vals) == len(SEEDS)
            if vals:
                mat[i, j] = np.mean(vals)
    return mat


def _draw_heatmap(mat: np.ndarray, highlight: np.ndarray,
                  widths: list[int], out: str) -> None:
    """Single heatmap; highlighted cells get a black border + bold text."""
    from matplotlib.patches import Rectangle

    fig_w = max(12, len(widths) * 0.42 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    im = ax.imshow(mat, aspect='auto', cmap='viridis_r')
    plt.colorbar(im, ax=ax, label='MDL sum CE (avg over layers & seeds)')

    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels([str(w) for w in widths], rotation=90, fontsize=6)
    ax.set_yticks(range(len(GAMMAS)))
    ax.set_yticklabels([f'{g:.1f}' for g in GAMMAS], fontsize=8)
    ax.set_xlabel('Hidden width', fontsize=11)
    ax.set_ylabel('γ', fontsize=11)

    for i in range(len(GAMMAS)):
        for j in range(len(widths)):
            v = mat[i, j]
            if np.isnan(v):
                continue
            if highlight[i, j]:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                       fill=False, linewidth=1.5))
                ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                        fontsize=6, color='black', fontweight='bold')
            else:
                ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                        fontsize=6, color='white')

    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out}')


def plot_heatmaps(mat: np.ndarray, widths: list[int], strategy: str) -> None:
    """Two separate heatmaps: best gamma per width, and best width per gamma."""
    best_row = np.nanargmin(mat, axis=0)
    best_col = np.nanargmin(mat, axis=1)

    hl_gamma = np.zeros(mat.shape, dtype=bool)
    for j, i in enumerate(best_row):
        hl_gamma[i, j] = True

    hl_width = np.zeros(mat.shape, dtype=bool)
    for i, j in enumerate(best_col):
        hl_width[i, j] = True

    _draw_heatmap(mat, hl_gamma, widths,
                  f'figures/heatmap_{strategy}_best_gamma.png')
    _draw_heatmap(mat, hl_width, widths,
                  f'figures/heatmap_{strategy}_best_width.png')


def _ols_scatter(x: np.ndarray, y: np.ndarray,
                 xlabel: str, ylabel: str, title: str, out: str) -> None:
    """Scatter + OLS fit with R^2 and p-value annotation."""
    import statsmodels.api as sm

    x_col = x.reshape(-1, 1)
    pred = LinearRegression().fit(x_col, y).predict(x_col)
    ols = sm.OLS(y, sm.add_constant(x_col)).fit()
    coef, intercept = ols.params[1], ols.params[0]
    r2, pval = ols.rsquared, ols.pvalues[1]

    print(f'\n  {title}:')
    print(f'    coef={coef:.4f}  intercept={intercept:.4f}  R^2={r2:.3f}  p={pval:.3e}')
    print(ols.summary())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, color='steelblue', s=40, zorder=3)
    ax.plot(x, pred, color='tomato', linewidth=2,
            label=f'OLS: {ylabel} = {intercept:.3f} + {coef:.3f}·{xlabel}\n'
            f'R^2={r2:.3f},  p={pval:.3e}')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out}')


def plot_regressions(mat: np.ndarray, widths: list[int], strategy: str) -> None:
    """Two regression plots: best_gamma ~ log2(width), best_width ~ gamma."""
    best_row = np.nanargmin(mat, axis=0)
    best_col = np.nanargmin(mat, axis=1)

    # 1. best gamma per width vs log2(width)
    best_gammas = np.array([GAMMAS[i] for i in best_row])
    log2_widths = np.log2(np.array(widths, dtype=float))
    _ols_scatter(log2_widths, best_gammas,
                 xlabel='log(width)', ylabel='best gamma',
                 title=f'{strategy}, best gamma per width',
                 out=f'figures/regression_{strategy}_best_gamma.png')

    # 2. best width per gamma vs gamma
    best_log2w = np.array([np.log2(widths[j]) for j in best_col])
    gammas_arr = np.array(GAMMAS, dtype=float)
    _ols_scatter(gammas_arr, best_log2w,
                 xlabel='gamma', ylabel='log(best width)',
                 title=f'{strategy}, best width per gamma',
                 out=f'figures/regression_{strategy}_best_width.png')


def plot_seed_variance(widths: list[int], data: dict, strategy: str) -> None:
    """Plot std across seeds vs gamma.
    """
    # std_mat[i, j] = std over seeds for (gamma_i, width_j)
    std_mat = np.full((len(GAMMAS), len(widths)), np.nan)
    for i, g in enumerate(GAMMAS):
        for j, w in enumerate(widths):
            vals = [data[(w, g, s)] for s in SEEDS if (w, g, s) in data]
            if len(vals) == len(SEEDS):
                std_mat[i, j] = np.std(vals, ddof=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(std_mat, aspect='auto')
    plt.colorbar(im, ax=ax, label='MDL std (over seeds)')

    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels([str(w) for w in widths], rotation=90, fontsize=6)
    ax.set_yticks(range(len(GAMMAS)))
    ax.set_yticklabels([f'{g:.1f}' for g in GAMMAS], fontsize=8)
    ax.set_xlabel('Hidden width', fontsize=11)
    ax.set_ylabel('gamma', fontsize=11)

    out = f'figures/std_{strategy}.png'
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out}')


def plot_gamma_range(mat: np.ndarray, widths: list[int],
                     data: dict, strategy: str) -> None:
    """For each width: find best gamma, record its seed stderr, then mark all
    gammas whose CI [avg_i ± se_i] overlaps [best_mdl ± se_best] as the 'safe range'.

    Overlap condition: avg_i − se_i ≤ best_mdl + se_best  AND  avg_i + se_i ≥ best_mdl − se_best.

    Plots:
      - Scatter of best gamma per width (same as regression plot).
      - Shaded band [min_safe_gamma, max_safe_gamma] per width.
      - Linear fits for both the lower and upper safe boundaries.
    """
    import statsmodels.api as sm

    # build std and se matrices
    n_seeds = len(SEEDS)
    std_mat = np.full((len(GAMMAS), len(widths)), np.nan)
    se_mat  = np.full((len(GAMMAS), len(widths)), np.nan)
    for i, g in enumerate(GAMMAS):
        for j, w in enumerate(widths):
            vals = [data[(w, g, s)] for s in SEEDS if (w, g, s) in data]
            if len(vals) == n_seeds:
                std_mat[i, j] = np.std(vals, ddof=1)
                se_mat[i, j]  = std_mat[i, j] / np.sqrt(n_seeds)

    best_gammas, lo_gammas, hi_gammas = [], [], []
    log2_widths = np.log2(np.array(widths, dtype=float))

    for j in range(len(widths)):
        col     = mat[:, j]
        best_i  = int(np.nanargmin(col))
        best_mdl = col[best_i]
        se_best  = se_mat[best_i, j] if not np.isnan(se_mat[best_i, j]) else 0.0

        # safe if CI [avg_i - se_i, avg_i + se_i] overlaps [best_mdl - se_best, best_mdl + se_best]
        # overlap iff avg_i - se_i <= best_mdl + se_best  AND  avg_i + se_i >= best_mdl - se_best
        safe = [i for i in range(len(GAMMAS))
                if not np.isnan(col[i])
                and not np.isnan(se_mat[i, j])
                and col[i] - se_mat[i, j] <= best_mdl + se_best
                and col[i] + se_mat[i, j] >= best_mdl - se_best]
        best_gammas.append(GAMMAS[best_i])
        lo_gammas.append(GAMMAS[min(safe)] if safe else GAMMAS[best_i])
        hi_gammas.append(GAMMAS[max(safe)] if safe else GAMMAS[best_i])

    best_gammas = np.array(best_gammas)
    lo_gammas   = np.array(lo_gammas)
    hi_gammas   = np.array(hi_gammas)

    def _fit(x, y):
        ols = sm.OLS(y, sm.add_constant(x)).fit()
        return ols.params[0], ols.params[1], ols.rsquared, ols.pvalues[1]

    b0, b1, r2b, pb = _fit(log2_widths, best_gammas)
    l0, l1, r2l, pl = _fit(log2_widths, lo_gammas)
    h0, h1, r2h, ph = _fit(log2_widths, hi_gammas)

    print(f'\n  [{strategy}] gamma range analysis:')
    print(f'    best  gamma: coef={b1:.4f}  R²={r2b:.3f}  p={pb:.3e}')
    print(f'    lower bound: coef={l1:.4f}  R²={r2l:.3f}  p={pl:.3e}')
    print(f'    upper bound: coef={h1:.4f}  R²={r2h:.3f}  p={ph:.3e}')

    x_line = np.linspace(log2_widths.min(), log2_widths.max(), 200)

    fig, ax = plt.subplots(figsize=(9, 5))

    # shaded safe range
    ax.fill_between(log2_widths, lo_gammas, hi_gammas,
                    alpha=0.25, color='steelblue', label='safe range (best ± std)')

    # scatter points for bounds
    ax.scatter(log2_widths, lo_gammas, color='steelblue', s=20, zorder=3)
    ax.scatter(log2_widths, hi_gammas, color='steelblue', s=20, zorder=3)
    ax.scatter(log2_widths, best_gammas, color='tomato', s=30, zorder=4,
               label='best γ')

    # OLS fit lines
    ax.plot(x_line, l0 + l1 * x_line, color='steelblue', linestyle='--', linewidth=1.5,
            label=f'lower fit: {l1:+.3f}·log₂w  R²={r2l:.2f}')
    ax.plot(x_line, h0 + h1 * x_line, color='steelblue', linestyle=':',  linewidth=1.5,
            label=f'upper fit: {h1:+.3f}·log₂w  R²={r2h:.2f}')
    ax.plot(x_line, b0 + b1 * x_line, color='tomato',    linestyle='-',  linewidth=2,
            label=f'best fit:  {b1:+.3f}·log₂w  R²={r2b:.2f}')

    ax.set_xlabel('log₂(hidden width)', fontsize=11)
    ax.set_ylabel('γ', fontsize=11)
    ax.set_title(f'{strategy} — safe γ range per width (tolerance = seed std)',
                 fontsize=11)
    ax.set_ylim(GAMMAS[0] - 0.1, GAMMAS[-1] + 0.1)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f'figures/gamma_range_{strategy}.png'
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'  Saved {out}')


if __name__ == '__main__':
    for strategy in ('gamma_second_zero', 'gamma'):
        print(f'\n=== {strategy} ===')

        widths, data = load_strategy(strategy)
        print(f'  Widths ({len(widths)}): {widths[:5]} ... {widths[-5:]}')
        print(f'  Gammas ({len(GAMMAS)}): {GAMMAS}')

        check_completeness(strategy, widths, data)
        mat = build_matrix(widths, data)

        print('  Plotting heatmaps...')
        plot_heatmaps(mat, widths, strategy)

        print('  Running regressions...')
        plot_regressions(mat, widths, strategy)

        print('  Plotting seed variance...')
        plot_seed_variance(widths, data, strategy)

        print('  Plotting gamma safe range...')
        plot_gamma_range(mat, widths, data, strategy)
