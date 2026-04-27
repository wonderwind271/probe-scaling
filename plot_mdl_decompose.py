"""MDL decomposition plot: total MDL, model codelength, data codelength.

For each (model, dataset, task), plot three curves vs probe hidden size:
  - total MDL       = mdl_sum_ce  (online prequential encoding cost)
  - model codelength = train_loss × n_train  (final probe's compression of training data)
  - data codelength  = total MDL − model codelength

Averaged over layers and seeds. Only results with train_loss present (i.e. from
runs with train_epochs=30) are included in the decomposition; results without it
still appear as MDL-only curves.

Usage:
    cd probe-scaling/
    python plot_mdl_decompose.py
    python plot_mdl_decompose.py --epochs 30        # filter by train_epochs
    python plot_mdl_decompose.py --out-dir plots/decompose
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')


COLORS = {
    'mdl':   '#333333',
    'model': '#E87722',   # orange
    'data':  '#0072B2',   # blue
}
SEEDS = [42, 142, 242, 342, 442]


def load_results(result_dir: Path, filter_epochs: Optional[int] = None) -> list[dict]:
    """Load results, keeping the highest train_epochs available per setting.

    For each (model, dataset, task, hidden, seed): if multiple epoch counts exist (e.g. 30 and 60), only the largest is kept. Within the same epoch count, the latest timestamp wins.
    filter_epochs: if set, skip records whose train_epochs != this value.
    """
    # best[(model, dataset, task, hidden, seed)] = (max_epochs, latest_ts, record_dict)
    best: dict = {}

    for rfile in sorted(result_dir.rglob('results.json')):
        with open(rfile) as fh:
            r = json.load(fh)

        cfg = r['config']
        mdl_cfg = cfg['mdl']

        if mdl_cfg.get('probe_type', 'mlp') != 'mlp':
            continue
        if 'train_loss' not in r:
            continue

        hidden = int(mdl_cfg['probe_hidden_size'][0])
        seed = int(mdl_cfg['seed'])
        model = str(cfg.get('model_short', ''))
        task = str(cfg.get('task_name', ''))
        dataset = str(cfg.get('dataset', {}).get('name', '') or
                      cfg.get('dataset_name', ''))
        epochs = int(mdl_cfg['train_epochs'])

        if filter_epochs is not None and epochs != filter_epochs:
            continue

        ts = rfile.parts[-2]
        key = (model, dataset, task, hidden, seed)

        prev = best.get(key)
        if prev is not None:
            prev_ep, prev_ts, _ = prev
            # keep whichever has more epochs; break ties by timestamp
            if ts <= prev_ts:
                continue

        splits = r.get('splits', [0])
        mdl_layers = {int(k): float(v) for k, v in r['mdl_sum_ce'].items()}
        train_layers = {int(k): float(v) for k, v in r['train_loss'].items()}

        best[key] = (epochs, ts, {
            'model':      model,
            'dataset':    dataset,
            'task':       task,
            'hidden':     hidden,
            'seed':       seed,
            'epochs':     epochs,
            'n_first':    splits[0],
            'n_train':    sum(splits),
            'mdl':        mdl_layers,
            'train_loss': train_layers,
        })

    return [rec for _, _, rec in best.values()]


def _build_bucket(records: list[dict]) -> dict:
    """Flatten records into bucket[(model, dataset, task, hidden, seed, layer)] -> single value."""
    bucket: dict = {}
    for rec in records:
        uniform = rec['n_first'] * np.log(2)
        for layer in sorted(rec['mdl'].keys()):
            if layer not in rec['train_loss']:
                continue
            key = (rec['model'], rec['dataset'], rec['task'],
                   rec['hidden'], rec['seed'], layer)
            mdl_val = rec['mdl'][layer] + uniform
            data_val = rec['train_loss'][layer] * rec['n_train']
            bucket[key] = {'mdl': mdl_val, 'model_cl': mdl_val - data_val, 'data_cl': data_val}
    return bucket


def _stats(vals: list[dict]) -> dict:
    """Compute mean ± stderr (over seeds) for each metric."""
    n = len(vals)
    leaf = {}
    for metric, key_m, key_se in [
        ('mdl', 'mdl_mean', 'mdl_stderr'),
        ('model_cl', 'model_cl_mean', 'model_cl_stderr'),
            ('data_cl', 'data_cl_mean', 'data_cl_stderr')]:
        arr = np.array([v[metric] for v in vals])
        leaf[key_m] = float(np.mean(arr))
        leaf[key_se] = float(np.std(arr) / np.sqrt(n)) if n > 1 else 0.0
    return leaf


def aggregate_per_layer(bucket: dict) -> dict:
    """Average over seeds per (model, dataset, task, hidden, layer).

    Returns nested dict:
        result[model][dataset][task][hidden][layer] = {mdl_mean, mdl_stderr, ...}
    """
    # group values by (model, dataset, task, hidden, layer) across seeds
    by_layer: dict = defaultdict(list)
    for (model, dataset, task, hidden, _seed, layer), val in bucket.items():
        by_layer[(model, dataset, task, hidden, layer)].append(val)

    result: dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for (model, dataset, task, hidden, layer), vals in by_layer.items():
        result[model][dataset][task][hidden][layer] = _stats(vals)
    return result


def aggregate(bucket: dict) -> dict:
    """Average over layers and seeds; stderr is over seeds only.

    For each (model, dataset, task, hidden, seed) the layer values are averaged
    first, giving one scalar per seed.  stderr is then computed over those
    n_seeds scalars.

    Returns nested dict:
        result[model][dataset][task][hidden] = {mdl_mean, mdl_stderr, ...}
    """
    # step 1: per-seed layer average
    # seed_avgs[(model, dataset, task, hidden, seed)] = list of layer values
    seed_vals: dict = defaultdict(list)
    for (model, dataset, task, hidden, seed, _layer), val in bucket.items():
        seed_vals[(model, dataset, task, hidden, seed)].append(val)

    # step 2: average each seed over layers → one dict per seed
    per_seed: dict = defaultdict(list)
    for (model, dataset, task, hidden, seed), vals in seed_vals.items():
        mean_val = {m: float(np.mean([v[m] for v in vals]))
                    for m in ('mdl', 'model_cl', 'data_cl')}
        per_seed[(model, dataset, task, hidden)].append(mean_val)

    # step 3: stderr over seeds
    result: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for (model, dataset, task, hidden), vals in per_seed.items():
        result[model][dataset][task][hidden] = _stats(vals)

    return result


def plot_decompose_per_layer(agg_pl: dict, out_dir: Path) -> None:
    """Per-layer stacked bar plots.

    One figure per (model, dataset, task): grid of 12 subplots (one per layer),
    each showing stacked bars (data CL + model CL) vs hidden size.
    """
    datasets = sorted({ds for m in agg_pl for ds in agg_pl[m]})
    models = sorted(agg_pl.keys())

    for dataset in datasets:
        tasks = sorted({t for m in agg_pl for t in agg_pl[m].get(dataset, {})})
        for task in tasks:
            for model in models:
                cell = agg_pl.get(model, {}).get(dataset, {}).get(task, {})
                if not cell:
                    continue
                hiddens = sorted(cell.keys())
                layers = sorted({l for h in hiddens for l in cell[h]})
                if not layers:
                    continue

                n_cols = 4
                n_rows = (len(layers) + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols,
                                         figsize=(4.5 * n_cols, 3.5 * n_rows),
                                         squeeze=False)

                x_idx = np.arange(len(hiddens))
                bar_w = 0.6

                for li, layer in enumerate(layers):
                    ax = axes[li // n_cols][li % n_cols]

                    dcl_m = np.array([cell[h][layer]['data_cl_mean'] if layer in cell[h] else np.nan for h in hiddens])
                    mcl_m = np.array([cell[h][layer]['model_cl_mean'] if layer in cell[h] else np.nan for h in hiddens])
                    mdl_m = np.array([cell[h][layer]['mdl_mean'] if layer in cell[h] else np.nan for h in hiddens])
                    dcl_s = np.array([cell[h][layer]['data_cl_stderr'] if layer in cell[h] else 0.0 for h in hiddens])
                    mcl_s = np.array([cell[h][layer]['model_cl_stderr'] if layer in cell[h] else 0.0 for h in hiddens])

                    ax.bar(x_idx, dcl_m, bar_w, color=COLORS['data'],  alpha=0.8, yerr=dcl_s, capsize=2, error_kw={'elinewidth': 0.8},
                           label='Data CL')
                    ax.bar(x_idx, mcl_m, bar_w, bottom=dcl_m,
                           color=COLORS['model'], alpha=0.8,
                           yerr=mcl_s, capsize=2, error_kw={'elinewidth': 0.8},
                           label='Model CL')
                    ax.plot(x_idx, mdl_m, color=COLORS['mdl'],
                            marker='o', markersize=3, linewidth=1.2)

                    ax.set_xticks(x_idx)
                    ax.set_xticklabels([str(h) for h in hiddens],
                                       rotation=90, fontsize=5)
                    ax.set_title(f'Layer {layer}', fontsize=9)
                    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
                    if li == 0:
                        ax.legend(fontsize=6)

                # hide unused subplots
                for li in range(len(layers), n_rows * n_cols):
                    axes[li // n_cols][li % n_cols].set_visible(False)

                fig.suptitle(
                    f'{model.upper()} | {dataset} | {task} : MDL decomposition per layer\n'
                    '(mean ± stderrerr over seeds; bars = data CL + model CL)',
                    fontsize=10)
                plt.tight_layout()
                out = out_dir / f'mdl_per_layer_{model}_{dataset}_{task}_60.png'
                plt.savefig(out, dpi=160, bbox_inches='tight')
                plt.close()
                print(f'Saved {out}')


def plot_mdl_per_layer(agg_pl: dict, out_dir: Path) -> None:
    """Grid of MDL vs hidden-size curves, one subplot per layer.

    One figure per (dataset, task); both models on each subplot.
    No codelength decomposition.
    """
    MODEL_COLORS = {'clip': '#E87722', 'dino': '#0072B2'}
    datasets = sorted({ds for m in agg_pl for ds in agg_pl[m]})
    models = sorted(agg_pl.keys())

    for dataset in datasets:
        tasks = sorted({t for m in agg_pl for t in agg_pl[m].get(dataset, {})})
        for task in tasks:
            layers = sorted({l for m in models
                             for h in agg_pl.get(m, {}).get(dataset, {}).get(task, {})
                             for l in agg_pl[m][dataset][task][h]})
            hiddens = sorted({h for m in models
                              for h in agg_pl.get(m, {}).get(dataset, {}).get(task, {})})
            if not layers or not hiddens:
                continue

            n_cols = 4
            n_rows = (len(layers) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(4.5 * n_cols, 3.5 * n_rows),
                                     squeeze=False)
            xs_log = np.log2(np.array(hiddens, dtype=float))

            for li, layer in enumerate(layers):
                ax = axes[li // n_cols][li % n_cols]
                for model in models:
                    cell = agg_pl.get(model, {}).get(dataset, {}).get(task, {})
                    ys = np.array([cell[h][layer]['mdl_mean']
                                   if h in cell and layer in cell[h] else np.nan
                                   for h in hiddens])
                    ers = np.array([cell[h][layer]['mdl_stderr']
                                    if h in cell and layer in cell[h] else 0.0
                                    for h in hiddens])
                    color = MODEL_COLORS.get(model, 'gray')
                    ax.plot(xs_log, ys, color=color, marker='o', markersize=3,
                            linewidth=1.5, label=model.upper())
                    ax.fill_between(xs_log, ys - ers, ys + ers,
                                    color=color, alpha=0.15)

                ax.set_xticks(xs_log)
                ax.set_xticklabels([str(h) for h in hiddens], rotation=90, fontsize=5)
                ax.set_title(f'Layer {layer}', fontsize=9)
                ax.grid(True, linestyle='--', alpha=0.3)
                if li == 0:
                    ax.legend(fontsize=7)

            for li in range(len(layers), n_rows * n_cols):
                axes[li // n_cols][li % n_cols].set_visible(False)

            fig.suptitle(f'{dataset} | {task} : MDL vs width per layer\n(mean ± stderrerr over seeds)', fontsize=10)
            plt.tight_layout()
            out = out_dir / f'mdl_per_layer_{dataset}_{task}_60.png'
            plt.savefig(out, dpi=160, bbox_inches='tight')
            plt.close()
            print(f'Saved {out}')


def _errbar(ax, xs, ys, errs, color, label, ls='-', marker='o', alpha_fill=0.15):
    ys = np.array(ys)
    errs = np.array(errs)
    ax.plot(xs, ys, color=color, linestyle=ls, marker=marker,
            markersize=5, linewidth=1.8, label=label)
    ax.fill_between(xs, ys - errs, ys + errs, color=color, alpha=alpha_fill)


def plot_decompose(agg: dict, out_dir: Path) -> None:
    """One figure per (dataset, task): stacked bars (data + model codelength)
    with total-MDL line overlaid.  Rows = models."""
    datasets = sorted({ds for m in agg for ds in agg[m]})
    models = sorted(agg.keys())

    for dataset in datasets:
        tasks = sorted({t for m in agg for t in agg[m].get(dataset, {})})
        for task in tasks:
            n_rows = len(models)
            fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), sharex=False)
            if n_rows == 1:
                axes = [axes]

            for ax, model in zip(axes, models):
                cell = agg.get(model, {}).get(dataset, {}).get(task, {})
                if not cell:
                    ax.set_title(f'{model.upper()} : no data')
                    continue

                hiddens = sorted(cell.keys())
                x_idx = np.arange(len(hiddens))
                bar_w = 0.6
                dcl_m = np.array([cell[h]['data_cl_mean'] for h in hiddens])
                mcl_m = np.array([cell[h]['model_cl_mean'] for h in hiddens])
                mdl_m = np.array([cell[h]['mdl_mean'] for h in hiddens])
                dcl_s = np.array([cell[h]['data_cl_stderr'] for h in hiddens])
                mcl_s = np.array([cell[h]['model_cl_stderr'] for h in hiddens])

                # stacked bars: data codelength (bottom) + model codelength (top)
                ax.bar(x_idx, dcl_m, bar_w,
                       color=COLORS['data'], alpha=0.8,
                       yerr=dcl_s, capsize=3, error_kw={'elinewidth': 1},
                       label='Data codelength (CE with known params)')
                ax.bar(x_idx, mcl_m, bar_w,
                       bottom=dcl_m, color=COLORS['model'], alpha=0.8,
                       yerr=mcl_s, capsize=3, error_kw={'elinewidth': 1},
                       label='Model codelength (online MDL − data CL)')

                # total MDL line for reference
                ax.plot(x_idx, mdl_m, color=COLORS['mdl'],
                        marker='o', markersize=5, linewidth=1.8,
                        label='Total MDL (check: should match bar tops)')

                ax.set_xticks(x_idx)
                ax.set_xticklabels([str(h) for h in hiddens],
                                   rotation=45, fontsize=7)
                ax.set_xlabel('Hidden size', fontsize=10)
                ax.set_ylabel('Sum CE / nats (avg over layers)', fontsize=9)
                ax.set_title(f'{model.upper()} | {dataset} | {task}', fontsize=11)
                ax.legend(fontsize=8)
                ax.grid(True, axis='y', linestyle='--', alpha=0.35)

            fig.suptitle(
                f'MDL decomposition : {dataset} / {task}\n'
                '(mean ± stderrerr across seeds; averaged over layers)',
                fontsize=11)
            plt.tight_layout()
            out = out_dir / f'mdl_decompose_{dataset}_{task}_60.png'
            plt.savefig(out, dpi=160, bbox_inches='tight')
            plt.close()
            print(f'Saved {out}')


def plot_summary_grid(agg: dict, out_dir: Path) -> None:
    """Grid: rows=tasks, cols=datasets; one line per model.  Total MDL only."""
    datasets = sorted({ds for m in agg for ds in agg[m]})
    tasks = sorted({t for m in agg for ds in agg[m] for t in agg[m][ds]})
    models = sorted(agg.keys())
    colors = {'clip': '#E87722', 'dino': '#0072B2'}

    n_rows, n_cols = len(tasks), len(datasets)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4 * n_rows),
                             squeeze=False)

    for ri, task in enumerate(tasks):
        for ci, dataset in enumerate(datasets):
            ax = axes[ri][ci]
            for model in models:
                cell = agg.get(model, {}).get(dataset, {}).get(task, {})
                if not cell:
                    continue
                hiddens = sorted(cell.keys())
                xs = np.log2(np.array(hiddens, dtype=float))
                ys = [cell[h]['mdl_mean'] for h in hiddens]
                ers = [cell[h]['mdl_stderr'] for h in hiddens]
                _errbar(ax, xs, ys, ers,
                        colors.get(model, 'gray'), model.upper())
            ax.set_xticks(np.log2(sorted({h for m in agg
                                          for ds in agg[m]
                                          for t in agg[m][ds]
                                          for h in agg[m][ds][t]})))
            ax.set_xticklabels(
                [str(int(2**x)) for x in ax.get_xticks()],
                rotation=45, fontsize=7)
            ax.set_title(f'{dataset} / {task}', fontsize=10)
            ax.set_xlabel('Hidden size', fontsize=9)
            ax.set_ylabel('Total MDL', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.35)

    fig.suptitle('Total MDL vs hidden size : CLIP vs DINO', fontsize=12)
    plt.tight_layout()
    out = out_dir / 'mdl_summary_grid.png'
    plt.savefig(out, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mscoco-dir', type=Path, default=Path('outputs/mscoco'))
    p.add_argument('--openimage-dir', type=Path, default=Path('outputs/openimage'))
    p.add_argument('--epochs', type=int, default=None, help='Filter to runs with this train_epochs value. Default: include all.')
    p.add_argument('--out-dir', type=Path, default=Path('plots/decompose'))
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for label, d in [('mscoco', args.mscoco_dir),
                     ('openimage', args.openimage_dir)]:
        if not d.exists():
            print(f'  [skip] {d} not found')
            continue
        recs = load_results(d, filter_epochs=args.epochs)
        # stamp the dataset name if not in config
        for r in recs:
            if not r['dataset']:
                r['dataset'] = label
        all_records.extend(recs)
        print(f'  Loaded {len(recs)} records from {label}')

    # report coverage
    from collections import Counter
    cov = Counter((r['model'], r['dataset'], r['task'], r['hidden'])
                  for r in all_records)
    models = sorted({r['model'] for r in all_records})
    datasets = sorted({r['dataset'] for r in all_records})
    tasks = sorted({r['task'] for r in all_records})
    hiddens = sorted({r['hidden'] for r in all_records})
    print(f'\nModels:   {models}')
    print(f'Datasets: {datasets}')
    print(f'Tasks:    {tasks}')
    print(f'Hiddens:  {hiddens}')
    has_train = sum(1 for r in all_records if r['train_loss'])
    print(f'Records with train_loss: {has_train}/{len(all_records)}')

    bucket = _build_bucket(all_records)
    # per layer
    agg_pl = aggregate_per_layer(bucket)
    # aggregate over layer and seed
    agg = aggregate(bucket)

    print('\nPlotting decomposition figures...')
    plot_decompose(agg, args.out_dir)

    print('\nPlotting MDL per layer (no decomposition)...')
    plot_mdl_per_layer(agg_pl, args.out_dir)

    print('\nPlotting per-layer decomposition...')
    plot_decompose_per_layer(agg_pl, args.out_dir)

    # print('\nPlotting summary grid...')
    # plot_summary_grid(agg, args.out_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
