"""
analyze_init_study.py
=====================
Analysis of probe initialization strategies for the somo task.

Data sources
------------
- outputs/llama/somo           : gamma (both layers), widths 16/64/256/1024, γ 0.3–0.8
- outputs/llama/somo-large-gamma: gamma (both layers), same widths, γ 1.0/1.5/2.0
- init_results/llama/somo      : all other strategies (gamma_first_zero,
                                  gamma_second_zero, ntk, mf, mup, standard)
                                  widths 4–2048, 5 seeds each

Strategies
----------
- gamma            : both W1 and W2 scaled by γ
- gamma_first_zero : W1=0, W2 scaled by γ   (network dead — no gradient through W1)
- gamma_second_zero: W1 scaled by γ, W2=0   (readout starts zero, trains fine)
- ntk              : prefactor 1/√n, LR = base_lr
- mf               : prefactor 1/n,  LR = base_lr × n
- mup              : no prefactor, Adam LR base_lr for W1/biases, base_lr/n for W2
- standard         : PyTorch default init

Figures written to figures/.
"""

from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')

# ── constants ──────────────────────────────────────────────────────────────

INIT_BASE = 'init_results/llama/somo'
GAMMA_BASE = 'outputs/llama/somo'
LGAMMA_BASE = 'outputs/llama/somo-large-gamma'

WIDTHS_INIT = [4, 16, 32, 128, 256, 512, 1024, 2048]
WIDTHS_GAMMA = sorted(set([16, 64, 256, 1024]) | set(WIDTHS_INIT))  # union of old + new
SEEDS = [42, 142, 242, 342, 442]
GAMMAS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0]
os.makedirs('figures', exist_ok=True)


def layer_avg(d: dict) -> float:
    """Average a layer-keyed dict over layers >= skip."""
    vals = [v for k, v in d.items()]
    return float(np.mean(vals))


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _record(r: dict) -> dict:
    """Extract scalar acc and MDL from a results.json dict."""
    return {
        'acc': layer_avg(r['test_acc']),
        'mdl': layer_avg(r['mdl_sum_ce']),
    }


def load_gamma_results() -> dict:
    """Load gamma (both-layers) results from the old outputs/ directories.

    Covers widths [16, 64, 256, 1024] and gammas [0.3, …, 2.0].
    Keys: (width, gamma, seed).
    If duplicate timestamps exist for the same key, the last one wins.
    """
    data = {}
    for base in (GAMMA_BASE, LGAMMA_BASE):
        pat = os.path.join(base, 'hidden-*', 'seed-*', 'gamma-*', '*', 'results.json')
        for f in sorted(glob.glob(pat)):
            parts = f.split('/')
            width = int(parts[-5].replace('hidden-', ''))
            seed = int(parts[-4].replace('seed-', ''))
            gamma = float(parts[-3].replace('gamma-', ''))
            data[(width, gamma, seed)] = _record(_load_json(f))
    return data


def load_init_gamma_results(strategy: str) -> dict:
    """Load gamma or gamma_second_zero from init_results/.

    Covers widths [4, …, 2048] and gammas [0.3, …, 2.0].
    Keys: (width, gamma, seed).
    Handles two path formats:
      old: strategy-X/gamma-G/timestamp/results.json
      new: strategy-X/gamma-G/opt-adamw/timestamp/results.json
    """
    data = {}
    # old format (gamma_second_zero): gamma-G/timestamp/results.json
    pat = os.path.join(INIT_BASE, 'hidden-*', 'seed-*',
                       f'strategy-{strategy}', 'gamma-*', '*', 'results.json')
    for f in sorted(glob.glob(pat)):
        parts = f.split('/')
        if parts[-3].startswith('opt-'):
            continue  # new format; handled below
        width = int(parts[-6].replace('hidden-', ''))
        seed = int(parts[-5].replace('seed-', ''))
        gamma = float(parts[-3].replace('gamma-', ''))
        data[(width, gamma, seed)] = _record(_load_json(f))
    # new format (gamma): gamma-G/opt-adamw/timestamp/results.json
    pat2 = os.path.join(INIT_BASE, 'hidden-*', 'seed-*',
                        f'strategy-{strategy}', 'gamma-*', 'opt-adamw', '*', 'results.json')
    for f in sorted(glob.glob(pat2)):
        parts = f.split('/')
        width = int(parts[-7].replace('hidden-', ''))
        seed = int(parts[-6].replace('seed-', ''))
        gamma = float(parts[-4].replace('gamma-', ''))
        data[(width, gamma, seed)] = _record(_load_json(f))
    return data


def load_fixed_strategy(strategy: str, optimizer: str = 'adamw') -> dict:
    """Load a fixed-parametrization strategy (ntk, mf, mup, standard) from init_results/.

    For optimizer='adamw' loads old-format paths (no opt-* subdir).
    Keys: (width, seed).

    For other optimizers loads new-format paths with opt-{optimizer} subdir.
    Multiple lr values share the same opt-* dir; lr is read from the config
    embedded in results.json.  Keys: (width, seed, lr).
    """
    data = {}
    if optimizer == 'adamw':
        pat = os.path.join(INIT_BASE, 'hidden-*', 'seed-*',
                           f'strategy-{strategy}', 'gamma-na', '*', 'results.json')
        for f in sorted(glob.glob(pat)):
            parts = f.split('/')
            width = int(parts[-6].replace('hidden-', ''))
            seed = int(parts[-5].replace('seed-', ''))
            data[(width, seed)] = _record(_load_json(f))
    else:
        pat = os.path.join(INIT_BASE, 'hidden-*', 'seed-*',
                           f'strategy-{strategy}', 'gamma-na',
                           f'opt-{optimizer}', '*', 'results.json')
        for f in sorted(glob.glob(pat)):
            parts = f.split('/')
            width = int(parts[-7].replace('hidden-', ''))
            seed = int(parts[-6].replace('seed-', ''))
            r = _load_json(f)
            lr = float(r['config']['mdl']['lr'])
            data[(width, seed, lr)] = _record(r)
    return data


def seed_mean_std(records: dict, keys: list):
    """Aggregate mean and std of acc and MDL over a list of keys.

    Returns (acc_mean, acc_std, mdl_mean, mdl_std), or (None,)*4 if no data.
    """
    vals_acc, vals_mdl = [], []
    for k in keys:
        if k in records:
            vals_acc.append(records[k]['acc'])
            vals_mdl.append(records[k]['mdl'])
    if not vals_acc:
        return None, None, None, None
    return (float(np.mean(vals_acc)), float(np.std(vals_acc)),
            float(np.mean(vals_mdl)), float(np.std(vals_mdl)))


def _pick_metric(acc_m, acc_s, mdl_m, mdl_s, metric: str):
    return (acc_m, acc_s) if metric == 'acc' else (mdl_m, mdl_s)


print("Loading data...")
gamma_data = load_gamma_results()
gamma_init_data = load_init_gamma_results('gamma')   # new: all WIDTHS_INIT
# merge: new data takes priority for same (w, gamma, seed) keys
gamma_data = {**gamma_data, **gamma_init_data}
gsz_data = load_init_gamma_results('gamma_second_zero')
ntk_adam_data = load_fixed_strategy('ntk',      optimizer='adamw')
mf_adam_data = load_fixed_strategy('mf',       optimizer='adamw')
mup_data = load_fixed_strategy('mup',      optimizer='adamw')
embed()
std_data = load_fixed_strategy('standard', optimizer='adamw')
ntk_sgd_data = load_fixed_strategy('ntk', optimizer='sgd')
mf_sgd_data = load_fixed_strategy('mf',  optimizer='sgd')
SGD_LRS = sorted(set(k[2] for k in ntk_sgd_data) | set(k[2] for k in mf_sgd_data))
print(f"  SGD lr values: {SGD_LRS}")
print("Done.\n")

GAMMA_STYLE = {
    'gamma':             ('tab:blue',  '-',  'o', 'gamma (W1=γ, W2=γ)'),
    'gamma_second_zero': ('tab:green', ':',  '^', 'gamma_second_zero (W1=γ, W2=0)'),
}

FIXED_STYLE = {
    'ntk':      ('tab:red',    '-',  's', 'NTK (Adam)'),
    'mf':       ('tab:brown',  '-',  '^', 'MF (Adam)'),
    'mup':      ('tab:purple', '-',  'o', 'μP (Adam)'),
    'standard': ('tab:gray',   '-.', 'D', 'Standard (Adam)'),
}

SGD_LR_LS = {0.001: '--', 0.005: ':', 0.05: (0, (3, 1, 1, 1))}


def _collect_legend_handles(axes):
    """Collect unique (handle, label) pairs across all axes, preserving order."""
    seen, handles, labels = set(), [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    return handles, labels


def _errorbar(ax, xs, ys, errs, color, ls, marker, label, **kw):
    """Thin wrapper around ax.errorbar with shared defaults."""
    ax.errorbar(xs, ys, yerr=errs, color=color, linestyle=ls,
                marker=marker, markersize=5, capsize=3,
                label=label, linewidth=1.5, **kw)


def _width_axis(ax, widths):
    """Configure a log2-scaled x-axis for hidden width."""
    ax.set_xscale('log', base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_xlabel('Hidden width')


def plot_gamma_comparison(metric: str, ylabel: str, fname: str) -> None:
    """Plot acc or MDL vs gamma value for all three gamma strategies.

    One subplot per hidden width; y-axis is shared for direct comparison.
    Widths from the old gamma experiments (16, 64, 256, 1024) and the new
    init_results (4–2048) are combined; gamma (both layers) is only plotted
    at widths where its data exist.
    """
    gamma_widths = sorted(set(k[0] for k in gamma_data))
    widths = sorted(set(gamma_widths) | set(WIDTHS_INIT))
    ncols = (len(widths) + 1) // 2
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 7), sharey=True)
    axes = axes.flatten()

    src_map = {'gamma': gamma_data, 'gamma_second_zero': gsz_data}
    for idx, w in enumerate(widths):
        ax = axes[idx]
        for strat, (color, ls, marker, label) in GAMMA_STYLE.items():
            xs, ys, errs = [], [], []
            for g in GAMMAS:
                stats = seed_mean_std(src_map[strat], [(w, g, s) for s in SEEDS])
                if stats[0] is None:
                    continue
                y, e = _pick_metric(*stats, metric)
                xs.append(g)
                ys.append(y)
                errs.append(e)
            if xs:
                _errorbar(ax, xs, ys, errs, color, ls, marker, label)
        ax.set_title(f'width = {w}', fontsize=10)
        ax.set_xlabel('γ')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(GAMMAS)
        ax.tick_params(axis='x', labelrotation=45)

    for i in range(len(widths), len(axes)):
        axes[i].set_visible(False)

    handles, labels = _collect_legend_handles(axes[:len(widths)])
    fig.legend(handles, labels, loc='lower right', fontsize=9)
    fig.suptitle(f'Gamma strategies — {ylabel} (mean ± std, avg over layers)', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'figures/{fname}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/{fname}")


# Plot 3: Adam vs SGD for NTK and MF
def plot_sgd_vs_adam(metric: str, ylabel: str, fname: str) -> None:
    """Compare Adam and SGD optimizers for NTK and MF parametrizations.

    Two subplots (NTK | MF) with shared y-axis.  The Adam baseline uses a
    single line; each SGD lr value gets its own line so the effect of lr
    scaling can be assessed alongside the optimizer difference.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    configs = [
        ('NTK', ntk_adam_data, ntk_sgd_data),
        ('MF',  mf_adam_data,  mf_sgd_data),
    ]
    for ax, (title, adam_data, sgd_data) in zip(axes, configs):
        # Adam baseline
        xs, ys, errs = [], [], []
        for w in WIDTHS_INIT:
            stats = seed_mean_std(adam_data, [(w, s) for s in SEEDS])
            if stats[0] is None:
                continue
            y, e = _pick_metric(*stats, metric)
            xs.append(w)
            ys.append(y)
            errs.append(e)
        base_color = FIXED_STYLE['ntk'][0] if title == 'NTK' else FIXED_STYLE['mf'][0]
        _errorbar(ax, xs, ys, errs, base_color, '-', 'o', f'{title} (Adam)')

        # SGD, one line per lr — same color as Adam, linestyle encodes lr
        for lr in SGD_LRS:
            ls = SGD_LR_LS.get(lr, ':')
            xs, ys, errs = [], [], []
            for w in WIDTHS_INIT:
                stats = seed_mean_std(sgd_data, [(w, s, lr) for s in SEEDS])
                if stats[0] is None:
                    continue
                y, e = _pick_metric(*stats, metric)
                xs.append(w)
                ys.append(y)
                errs.append(e)
            if xs:
                _errorbar(ax, xs, ys, errs, base_color, ls, 's',
                          f'{title} SGD (lr={lr})')

        _width_axis(ax, WIDTHS_INIT)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Adam vs SGD — {ylabel} (mean ± std, avg over layers)', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'figures/{fname}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/{fname}")


# Plot 4: All strategies together
def _best_gamma(src: dict, widths: list, metric: str) -> dict:
    """Return {width: (mean, std)} using the gamma that maximises acc or minimises MDL."""
    result = {}
    for w in widths:
        best_val, best_mean, best_std = None, None, None
        for g in GAMMAS:
            stats = seed_mean_std(src, [(w, g, s) for s in SEEDS])
            if stats[0] is None:
                continue
            y, e = _pick_metric(*stats, metric)
            val = y if metric == 'acc' else -y
            if best_val is None or val > best_val:
                best_val, best_mean, best_std = val, y, e
        if best_mean is not None:
            result[w] = (best_mean, best_std)
    return result


def plot_all_strategies(metric: str, ylabel: str, fname: str,
                        gamma_agg='best') -> None:
    """Compare all six strategies on a single width-vs-metric plot.

    gamma_agg='best' selects the best gamma per width for gamma strategies.
    Pass a float (e.g. 1.0) to fix gamma across all widths instead.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))

    src_map = {'ntk': ntk_adam_data, 'mf': mf_adam_data,
               'mup': mup_data, 'standard': std_data}
    for strat, (color, ls, marker, label) in FIXED_STYLE.items():
        xs, ys, errs = [], [], []
        for w in WIDTHS_INIT:
            stats = seed_mean_std(src_map[strat], [(w, s) for s in SEEDS])
            if stats[0] is None:
                continue
            y, e = _pick_metric(*stats, metric)
            xs.append(w)
            ys.append(y)
            errs.append(e)
        _errorbar(ax, xs, ys, errs, color, ls, marker, label)

    gamma_src_map = {
        'gamma':             (gamma_data, WIDTHS_GAMMA),
        'gamma_second_zero': (gsz_data,  WIDTHS_INIT),
    }
    for strat, (src, widths_avail) in gamma_src_map.items():
        color, ls, marker, label = GAMMA_STYLE[strat]
        if isinstance(gamma_agg, float):
            xs, ys, errs = [], [], []
            for w in widths_avail:
                stats = seed_mean_std(src, [(w, gamma_agg, s) for s in SEEDS])
                if stats[0] is None:
                    continue
                y, e = _pick_metric(*stats, metric)
                xs.append(w)
                ys.append(y)
                errs.append(e)
            suffix = f' (γ={gamma_agg})'
        else:
            best = _best_gamma(src, widths_avail, metric)
            xs = sorted(best)
            ys = [best[w][0] for w in xs]
            errs = [best[w][1] for w in xs]
            suffix = ' (best γ)'
        if xs:
            _errorbar(ax, xs, ys, errs, color, ls, marker, label + suffix)

    all_widths = sorted(set(WIDTHS_INIT) | set(WIDTHS_GAMMA))
    _width_axis(ax, all_widths)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    agg_str = f'γ={gamma_agg}' if isinstance(gamma_agg, float) else 'best γ'
    ax.set_title(f'All strategies — {ylabel} vs width ({agg_str}, avg over layers)')
    plt.tight_layout()
    plt.savefig(f'figures/{fname}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/{fname}")


# Plot 5: MDL per layer (layer profile)
def _load_layer_vectors(strategy: str, width: int,
                        optimizer: str = 'adamw', gamma: float = 1.0,
                        lr: float = None):
    """Return (layers, mean_mdl_per_layer) for one strategy/width combination.

    For gamma strategies uses the specified gamma value as reference.
    For SGD results, filters files by lr read from the embedded config.
    Returns (None, None) if no matching files are found.
    """
    if strategy in ('ntk', 'mf', 'mup', 'standard'):
        if optimizer == 'adamw':
            pat = os.path.join(INIT_BASE, f'hidden-{width}', 'seed-*',
                               f'strategy-{strategy}', 'gamma-na', '*', 'results.json')
        else:
            pat = os.path.join(INIT_BASE, f'hidden-{width}', 'seed-*',
                               f'strategy-{strategy}', 'gamma-na',
                               f'opt-{optimizer}', '*', 'results.json')
        files = glob.glob(pat)
    elif strategy in ('gamma_first_zero', 'gamma_second_zero'):
        pat = os.path.join(INIT_BASE, f'hidden-{width}', 'seed-*',
                           f'strategy-{strategy}', f'gamma-{gamma}', '*', 'results.json')
        files = glob.glob(pat)
    else:  # gamma (old outputs/)
        files = (glob.glob(os.path.join(GAMMA_BASE, f'hidden-{width}', 'seed-*',
                                        f'gamma-{gamma}', '*', 'results.json')) +
                 glob.glob(os.path.join(LGAMMA_BASE, f'hidden-{width}', 'seed-*',
                                        f'gamma-{gamma}', '*', 'results.json')))

    # Deduplicate: keep only the latest timestamp per seed dir
    seed_to_latest: dict = {}
    for f in files:
        seed_dir = os.path.dirname(os.path.dirname(f))  # strip timestamp + filename
        current = seed_to_latest.get(seed_dir)
        if current is None or f > current:
            seed_to_latest[seed_dir] = f
    files = list(seed_to_latest.values())

    layer_vals = defaultdict(list)
    for f in files:
        r = _load_json(f)
        if lr is not None and float(r['config']['mdl']['lr']) != lr:
            continue
        for k, v in r['mdl_sum_ce'].items():
            layer_vals[int(k)].append(v)

    if not layer_vals:
        return None, None
    layers = sorted(layer_vals)
    return layers, [np.mean(layer_vals[l]) for l in layers]


def plot_layer_profiles(width: int = 256) -> None:
    """Plot MDL sum CE per layer for all strategies at a fixed hidden width.

    All gamma strategies use γ=1.0 as a neutral reference point.
    SGD strategies appear as dashed lines, one per lr value.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5))

    adam_configs = [
        ('gamma',             'tab:blue',   'gamma (γ=1.0, Adam)'),
        ('gamma_second_zero', 'tab:green',  'gamma_second_zero (γ=1.0, Adam)'),
        ('ntk',               'tab:red',    'NTK (Adam)'),
        ('mf',                'tab:brown',  'MF (Adam)'),
        ('mup',               'tab:purple', 'μP (Adam)'),
        ('standard',          'tab:gray',   'Standard (Adam)'),
    ]
    for strat, color, label in adam_configs:
        layers, means = _load_layer_vectors(strat, width, optimizer='adamw')
        if layers is None:
            continue
        ax.plot(layers, means,
                color=color, label=label, linewidth=1.5)

    for strat in ('ntk', 'mf'):
        base_color = FIXED_STYLE[strat][0]
        for lr in SGD_LRS:
            ls = SGD_LR_LS.get(lr, ':')
            layers, means = _load_layer_vectors(strat, width, optimizer='sgd', lr=lr)
            if layers is None:
                continue
            ax.plot(layers, means,
                    color=base_color, linestyle=ls,
                    label=f'{strat.upper()} SGD (lr={lr})', linewidth=1.2)

    ax.set_xlabel('Layer')
    ax.set_ylabel('MDL sum CE (mean over seeds)')
    ax.set_title(f'MDL per layer — width={width} (γ=1.0 for gamma strategies)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'figures/layer_profile_w{width}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/layer_profile_w{width}.png")


# Plot 6: Heatmap — metric vs (gamma, width)
def plot_gamma_heatmap(strat: str, src: dict, widths: list,
                       metric: str, fname: str) -> None:
    """Heatmap of acc or MDL averaged over seeds, with axes (gamma × width).

    Green = good (high acc / low MDL); cells are annotated with values.
    """
    mat = np.full((len(GAMMAS), len(widths)), np.nan)
    for i, g in enumerate(GAMMAS):
        for j, w in enumerate(widths):
            stats = seed_mean_std(src, [(w, g, s) for s in SEEDS])
            if stats[0] is not None:
                mat[i, j], _ = _pick_metric(*stats, metric)

    fig, ax = plt.subplots(figsize=(max(5, len(widths) * 0.9 + 1), 4))
    im = ax.imshow(mat, aspect='auto')
    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_yticks(range(len(GAMMAS)))
    ax.set_yticklabels([str(g) for g in GAMMAS])
    ax.set_xlabel('Width')
    ax.set_ylabel('γ')
    fmt = '.3f' if metric == 'acc' else '.0f'
    for i in range(len(GAMMAS)):
        for j in range(len(widths)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f'{mat[i,j]:{fmt}}', ha='center', va='center', fontsize=7)
    ylabel = 'Test Acc' if metric == 'acc' else 'MDL sum CE'
    ax.set_title(f'{strat} — {ylabel} heatmap (avg over layers & seeds)')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


# Drift analysis: NTK vs MF under Adam and SGD
def load_drift_data(strategy: str, optimizer: str = 'adamw') -> dict:
    """Load drift_final_total_l2 (layer-averaged) from init_results/.

    Keys: (width, seed) for adamw; (width, seed, lr) for sgd.
    """
    data = {}
    if optimizer == 'adamw':
        pat = os.path.join(INIT_BASE, 'hidden-*', 'seed-*',
                           f'strategy-{strategy}', 'gamma-na', '*', 'results.json')
        for f in sorted(glob.glob(pat)):
            parts = f.split('/')
            width = int(parts[-6].replace('hidden-', ''))
            seed = int(parts[-5].replace('seed-', ''))
            r = _load_json(f)
            if 'drift_final_total_l2' in r:
                data[(width, seed)] = layer_avg(r['drift_final_total_l2'])
    else:
        pat = os.path.join(INIT_BASE, 'hidden-*', 'seed-*',
                           f'strategy-{strategy}', 'gamma-na',
                           f'opt-{optimizer}', '*', 'results.json')
        for f in sorted(glob.glob(pat)):
            parts = f.split('/')
            width = int(parts[-7].replace('hidden-', ''))
            seed = int(parts[-6].replace('seed-', ''))
            r = _load_json(f)
            if 'drift_final_total_l2' in r:
                lr = float(r['config']['mdl']['lr'])
                data[(width, seed, lr)] = layer_avg(r['drift_final_total_l2'])
    return data


def _drift_mean_std(records: dict, keys: list):
    """Return (mean, std) of drift over a list of keys, or (None, None)."""
    vals = [records[k] for k in keys if k in records]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def plot_drift_sgd_vs_adam() -> None:
    """Compare parameter drift (L2 from init, avg over layers) for NTK and MF.

    Two subplots (NTK | MF) with shared y-axis.  Each panel shows the Adam
    baseline alongside one line per SGD lr value, plotted against hidden width.
    Theory predicts NTK drift < MF drift, and SGD should make this gap more
    visible than Adam (whose gradient normalisation reduces the difference).
    """
    ntk_adam_drift = load_drift_data('ntk', optimizer='adamw')
    mf_adam_drift = load_drift_data('mf',  optimizer='adamw')
    ntk_sgd_drift = load_drift_data('ntk', optimizer='sgd')
    mf_sgd_drift = load_drift_data('mf',  optimizer='sgd')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    configs = [
        ('NTK', ntk_adam_drift, ntk_sgd_drift, FIXED_STYLE['ntk'][0]),
        ('MF',  mf_adam_drift,  mf_sgd_drift,  FIXED_STYLE['mf'][0]),
    ]
    for ax, (title, adam_drift, sgd_drift, base_color) in zip(axes, configs):
        xs, ys, errs = [], [], []
        for w in WIDTHS_INIT:
            m, s = _drift_mean_std(adam_drift, [(w, seed) for seed in SEEDS])
            if m is None:
                continue
            xs.append(w)
            ys.append(m)
            errs.append(s)
        _errorbar(ax, xs, ys, errs, base_color, '-', 'o', f'{title} (Adam)')

        for lr in SGD_LRS:
            ls = SGD_LR_LS.get(lr, ':')
            xs, ys, errs = [], [], []
            for w in WIDTHS_INIT:
                m, s = _drift_mean_std(sgd_drift, [(w, seed, lr) for seed in SEEDS])
                if m is None:
                    continue
                xs.append(w)
                ys.append(m)
                errs.append(s)
            if xs:
                _errorbar(ax, xs, ys, errs, base_color, ls, 's',
                          f'{title} SGD (lr={lr})')

        _width_axis(ax, WIDTHS_INIT)
        ax.set_ylabel('Drift L2 (avg over layers)')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Parameter drift (L2 from init) — NTK vs MF, Adam vs SGD\n'
                 '(mean ± std across seeds, avg over layers)', fontsize=11)
    plt.tight_layout()
    plt.savefig('figures/drift_ntk_mf.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figures/drift_ntk_mf.png")


def plot_drift_all_adam() -> None:
    """Drift vs width for all four Adam strategies on one axis.

    Complements the SGD drift plot by showing how μP and Standard
    compare to NTK and MF under the same Adam optimizer.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for strat, (color, ls, marker, label) in FIXED_STYLE.items():
        drift = load_drift_data(strat, optimizer='adamw')
        xs, ys, errs = [], [], []
        for w in WIDTHS_INIT:
            m, s = _drift_mean_std(drift, [(w, seed) for seed in SEEDS])
            if m is None:
                continue
            xs.append(w)
            ys.append(m)
            errs.append(s)
        if xs:
            _errorbar(ax, xs, ys, errs, color, ls, marker, label)

    _width_axis(ax, WIDTHS_INIT)
    ax.set_ylabel('Drift L2 (avg over layers)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Parameter drift — all Adam strategies vs width\n'
                 '(mean ± std across seeds, avg over layers)')
    plt.tight_layout()
    plt.savefig('figures/drift_all_adam.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figures/drift_all_adam.png")


def plot_drift_two_panel() -> None:
    """Drift figure with two subfigures: SGD (left) and Adam (right).

    Left panel: NTK and MF under SGD, one line per lr value, distinguished
    by color (NTK vs MF) and linestyle (lr value).
    Right panel: NTK, MF, μP, and Standard under Adam.
    Shared y-axis so the scale is directly comparable across optimizers.
    """
    ntk_adam_drift = load_drift_data('ntk', optimizer='adamw')
    mf_adam_drift = load_drift_data('mf',  optimizer='adamw')
    mup_drift = load_drift_data('mup', optimizer='adamw')
    std_drift = load_drift_data('standard', optimizer='adamw')
    ntk_sgd_drift = load_drift_data('ntk', optimizer='sgd')
    mf_sgd_drift = load_drift_data('mf',  optimizer='sgd')

    fig, (ax_sgd, ax_adam) = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    # Left: SGD (NTK and MF, one line per lr)
    for strat_name, sgd_drift, base_color in [
        ('NTK', ntk_sgd_drift, FIXED_STYLE['ntk'][0]),
        ('MF',  mf_sgd_drift,  FIXED_STYLE['mf'][0]),
    ]:
        for lr in SGD_LRS:
            ls = SGD_LR_LS.get(lr, ':')
            xs, ys, errs = [], [], []
            for w in WIDTHS_INIT:
                m, s = _drift_mean_std(sgd_drift, [(w, seed, lr) for seed in SEEDS])
                if m is None:
                    continue
                xs.append(w)
                ys.append(m)
                errs.append(s)
            if xs:
                _errorbar(ax_sgd, xs, ys, errs, base_color, ls, 's',
                          f'{strat_name} (lr={lr})')

    _width_axis(ax_sgd, WIDTHS_INIT)
    ax_sgd.set_ylabel('Drift L2 (avg over layers)')
    ax_sgd.set_title('SGD — NTK and MF')
    ax_sgd.legend(fontsize=8)
    ax_sgd.grid(True, alpha=0.3)

    # Right: Adam (NTK, MF, μP, Standard)
    adam_configs = [
        ('ntk',      ntk_adam_drift),
        ('mf',       mf_adam_drift),
        ('mup',      mup_drift),
        ('standard', std_drift),
    ]
    for strat, drift in adam_configs:
        color, ls, marker, label = FIXED_STYLE[strat]
        xs, ys, errs = [], [], []
        for w in WIDTHS_INIT:
            m, s = _drift_mean_std(drift, [(w, seed) for seed in SEEDS])
            if m is None:
                continue
            xs.append(w)
            ys.append(m)
            errs.append(s)
        if xs:
            _errorbar(ax_adam, xs, ys, errs, color, ls, marker, label)

    _width_axis(ax_adam, WIDTHS_INIT)
    ax_adam.set_title('Adam — NTK, MF, μP, Standard')
    ax_adam.legend(fontsize=8)
    ax_adam.grid(True, alpha=0.3)

    fig.suptitle('Parameter drift (L2 from init, avg over layers) vs hidden width\n'
                 '(mean ± std across seeds)', fontsize=11)
    plt.tight_layout()
    plt.savefig('figures/drift_two_panel.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figures/drift_two_panel.png")


def plot_standard_min_mdl_vs_width() -> None:
    """Plot min-over-LLM-layers MDL vs hidden width for the standard strategy.

    For each (width, seed): take the minimum MDL sum CE over all LLM layers.
    Then compute mean ± standard error over seeds and plot vs hidden width.
    This shows how the best achievable MDL (at the optimal LLM layer) scales
    with probe width under standard initialisation.
    """
    # For each (width, seed): min MDL over LLM layers
    width_seed_min: dict = {}   # (width, seed) -> min mdl over layers
    for w in WIDTHS_INIT:
        pat = os.path.join(INIT_BASE, f'hidden-{w}', 'seed-*',
                           'strategy-standard', 'gamma-na', '*', 'results.json')
        seed_to_file: dict = {}
        for f in sorted(glob.glob(pat)):
            parts = f.split('/')
            seed = int(parts[-5].replace('seed-', ''))
            seed_to_file[seed] = f  # latest timestamp wins (sorted)
        for seed, f in seed_to_file.items():
            r = _load_json(f)
            width_seed_min[(w, seed)] = min(r['mdl_sum_ce'].values())

    if not width_seed_min:
        print("  No standard data found — skipping.")
        return

    xs, ys, errs = [], [], []
    for w in WIDTHS_INIT:
        vals = [width_seed_min[(w, s)] for s in SEEDS if (w, s) in width_seed_min]
        if not vals:
            continue
        arr = np.array(vals)
        xs.append(w)
        ys.append(float(np.mean(arr)))
        errs.append(float(np.std(arr) / np.sqrt(len(arr))))

    fig, ax = plt.subplots(figsize=(7, 4))
    _errorbar(ax, xs, ys, errs,
              FIXED_STYLE['standard'][0], '-', 'D',
              'Standard (min over layers)')
    _width_axis(ax, xs)
    ax.set_ylabel('MDL sum CE')
    ax.set_title('Standard init — min MDL over LLM layers vs hidden width\n'
                 '(mean ± SE across seeds)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/standard_min_mdl_vs_width.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved figures/standard_min_mdl_vs_width.png")


# Run all plots
print("[1] Gamma strategy comparison...")
plot_gamma_comparison('acc', 'Test Accuracy', 'gamma_cmp_acc.png')
plot_gamma_comparison('mdl', 'MDL (sum CE)',  'gamma_cmp_mdl.png')

print("\n[3] Adam vs SGD comparison...")
plot_sgd_vs_adam('acc', 'Test Accuracy', 'sgd_vs_adam_acc.png')
plot_sgd_vs_adam('mdl', 'MDL (sum CE)',  'sgd_vs_adam_mdl.png')

print("\n[4] All strategies together...")
plot_all_strategies('acc', 'Test Accuracy', 'all_strat_acc_best_gamma.png', gamma_agg='best')
plot_all_strategies('mdl', 'MDL (sum CE)',  'all_strat_mdl_best_gamma.png', gamma_agg='best')
plot_all_strategies('acc', 'Test Accuracy', 'all_strat_acc_gamma1.png',     gamma_agg=1.0)
plot_all_strategies('mdl', 'MDL (sum CE)',  'all_strat_mdl_gamma1.png',     gamma_agg=1.0)

print("\n[5] Layer profiles...")
for w in [16, 256, 1024]:
    plot_layer_profiles(w)

print("\n[6] Heatmaps...")
plot_gamma_heatmap('gamma',             gamma_data, WIDTHS_GAMMA, 'acc', 'figures/heatmap_gamma_acc.png')
plot_gamma_heatmap('gamma',             gamma_data, WIDTHS_GAMMA, 'mdl', 'figures/heatmap_gamma_mdl.png')
plot_gamma_heatmap('gamma_second_zero', gsz_data,   WIDTHS_INIT,  'acc', 'figures/heatmap_gamma_second_zero_acc.png')
plot_gamma_heatmap('gamma_second_zero', gsz_data,   WIDTHS_INIT,  'mdl', 'figures/heatmap_gamma_second_zero_mdl.png')

print("\nAll figures saved to figures/")

print("\n[8] Drift analysis...")
plot_drift_two_panel()


# ══════════════════════════════════════════════════════════════════════════════
# New Plot A: NTK / MF / μP — all variants (Adam + SGD) together
# ══════════════════════════════════════════════════════════════════════════════

def plot_ntk_mf_mup_all(metric: str, ylabel: str, fname: str) -> None:
    """Width-scaling plot combining all NTK, MF, and μP variants.

    Shows Adam baselines for NTK, MF, and μP, plus one line per SGD lr for
    NTK and MF.  Standard is omitted so the three parametrisations are the
    focus.  Both subplots (acc, mdl) can be generated by calling twice.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))

    # μP (Adam only — no SGD run)
    color_mup, ls_mup, marker_mup, label_mup = FIXED_STYLE['mup']
    xs, ys, errs = [], [], []
    for w in WIDTHS_INIT:
        stats = seed_mean_std(mup_data, [(w, s) for s in SEEDS])
        if stats[0] is None:
            continue
        y, e = _pick_metric(*stats, metric)
        xs.append(w)
        ys.append(y)
        errs.append(e)
    _errorbar(ax, xs, ys, errs, color_mup, ls_mup, marker_mup, label_mup)

    # NTK and MF: Adam + each SGD lr
    for strat, adam_src, sgd_src, base_color, base_label in [
        ('NTK', ntk_adam_data, ntk_sgd_data, FIXED_STYLE['ntk'][0], 'NTK'),
        ('MF',  mf_adam_data,  mf_sgd_data,  FIXED_STYLE['mf'][0],  'MF'),
    ]:
        # Adam
        xs, ys, errs = [], [], []
        for w in WIDTHS_INIT:
            stats = seed_mean_std(adam_src, [(w, s) for s in SEEDS])
            if stats[0] is None:
                continue
            y, e = _pick_metric(*stats, metric)
            xs.append(w)
            ys.append(y)
            errs.append(e)
        _errorbar(ax, xs, ys, errs, base_color, '-', 'o', f'{base_label} (Adam)')

        # SGD — same color as Adam, linestyle encodes lr
        for lr in SGD_LRS:
            ls = SGD_LR_LS.get(lr, ':')
            xs, ys, errs = [], [], []
            for w in WIDTHS_INIT:
                stats = seed_mean_std(sgd_src, [(w, s, lr) for s in SEEDS])
                if stats[0] is None:
                    continue
                y, e = _pick_metric(*stats, metric)
                xs.append(w)
                ys.append(y)
                errs.append(e)
            if xs:
                _errorbar(ax, xs, ys, errs, base_color, ls, 's',
                          f'{base_label} (SGD lr={lr})')

    _width_axis(ax, WIDTHS_INIT)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'NTK / MF / μP — {ylabel} vs width\n'
                 '(mean ± std across seeds, avg over layers)')
    plt.tight_layout()
    plt.savefig(f'figures/{fname}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figures/{fname}")


# New Plot B: Avg MDL vs width — per-setting comparison against Standard
def _plot_vs_standard_width(setting_label: str,
                            get_setting_stats,   # callable(w) -> (mean, std) or (None, None)
                            widths: list,
                            fname: str) -> None:
    """Single plot: avg MDL vs width for one setting against Standard.

    Both lines are drawn on the same axes so the gap to the standard baseline
    is immediately visible.  Error bars show std across seeds.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    # Standard baseline
    xs_s, ys_s, errs_s = [], [], []
    for w in widths:
        stats = seed_mean_std(std_data, [(w, s) for s in SEEDS])
        y, e = _pick_metric(*stats, 'mdl')
        if y is not None:
            xs_s.append(w)
            ys_s.append(y)
            errs_s.append(e)
    if xs_s:
        color_s, ls_s, marker_s, label_s = FIXED_STYLE['standard']
        _errorbar(ax, xs_s, ys_s, errs_s, color_s, ls_s, marker_s, label_s)

    # Target setting
    xs_t, ys_t, errs_t = [], [], []
    for w in widths:
        m, e = get_setting_stats(w)
        if m is not None:
            xs_t.append(w)
            ys_t.append(m)
            errs_t.append(e)
    if xs_t:
        _errorbar(ax, xs_t, ys_t, errs_t, 'tab:blue', '-', 'o', setting_label)

    _width_axis(ax, widths)
    ax.set_ylabel('MDL sum CE (avg over layers, mean ± std)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Avg MDL vs width: {setting_label} vs Standard')
    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


def plot_all_vs_standard() -> None:
    """Generate one avg-MDL-vs-width comparison per setting against Standard.

    Excluded: MF+Adam (already known to diverge at large width).
    Figures saved in figures/vs_standard/.
    """
    out = 'figures/vs_standard'
    os.makedirs(out, exist_ok=True)

    # ── gamma (both layers) ────────────────────────────────────────────────
    gamma_widths = sorted(set(k[0] for k in gamma_data))
    for g in GAMMAS:
        def _gs_gamma(w, _g=g):
            return _pick_metric(*seed_mean_std(gamma_data, [(w, _g, s) for s in SEEDS]), 'mdl')
        _plot_vs_standard_width(
            f'gamma (W1=γ, W2=γ)  γ={g}',
            _gs_gamma, gamma_widths,
            f'{out}/gamma_g{g}.png')

    # ── gamma_second_zero ──────────────────────────────────────────────────
    for g in GAMMAS:
        def _gs_gsz(w, _g=g):
            return _pick_metric(*seed_mean_std(gsz_data, [(w, _g, s) for s in SEEDS]), 'mdl')
        _plot_vs_standard_width(
            f'gamma_second_zero (W1=γ, W2=0)  γ={g}',
            _gs_gsz, WIDTHS_INIT,
            f'{out}/gamma_second_zero_g{g}.png')

    # ── NTK (Adam) ─────────────────────────────────────────────────────────
    _plot_vs_standard_width(
        'NTK (Adam)',
        lambda w: _pick_metric(*seed_mean_std(ntk_adam_data, [(w, s) for s in SEEDS]), 'mdl'),
        WIDTHS_INIT, f'{out}/ntk_adam.png')

    # ── μP (Adam) ──────────────────────────────────────────────────────────
    _plot_vs_standard_width(
        'μP (Adam)',
        lambda w: _pick_metric(*seed_mean_std(mup_data, [(w, s) for s in SEEDS]), 'mdl'),
        WIDTHS_INIT, f'{out}/mup_adam.png')

    # ── NTK (SGD) and MF (SGD) — one figure per lr ─────────────────────────
    sgd_data_map = {'ntk': ntk_sgd_data, 'mf': mf_sgd_data}
    for strat_name, strat_key in [('NTK', 'ntk'), ('MF', 'mf')]:
        for lr in SGD_LRS:
            def _gs_sgd(w, _d=sgd_data_map[strat_key], _lr=lr):
                return _pick_metric(*seed_mean_std(_d, [(w, s, _lr) for s in SEEDS]), 'mdl')
            _plot_vs_standard_width(
                f'{strat_name} (SGD lr={lr})',
                _gs_sgd, WIDTHS_INIT,
                f'{out}/{strat_key}_sgd_lr{lr}.png')


print("\n[A] NTK / MF / μP all variants together...")
plot_ntk_mf_mup_all('acc', 'Test Accuracy', 'ntk_mf_mup_acc.png')
plot_ntk_mf_mup_all('mdl', 'MDL (sum CE)',  'ntk_mf_mup_mdl.png')

print("\n[B] Per-setting MDL curve vs Standard...")
plot_all_vs_standard()

print("\nDone.")
