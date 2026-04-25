#!/usr/bin/env python3
"""Plot per-layer MDL vs hidden size for a given model/task directory.

For each layer, draw one curve across hidden sizes (mean ± stderr over seeds),
producing a single figure with all layers overlaid.

Usage:
    python plot_mdl_per_layer.py --task-dir outputs/dino/car
    python plot_mdl_per_layer.py --task-dir outputs/clip/car --out-dir plots/per_layer
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


@dataclass
class SeedRun:
    hidden_size: int
    seed: int
    # layer_idx (int) -> mdl_sum_ce (float)
    layer_mdl: Dict[int, float] = field(default_factory=dict)
    layer_test_acc: Dict[int, float] = field(default_factory=dict)


def parse_hidden_size(path: Path) -> int:
    return int(path.name.replace('hidden-', ''))


def parse_seed(path: Path) -> int:
    return int(path.name.replace('seed-', ''))


def latest_results_by_seed(hidden_dir: Path) -> List[Tuple[int, Path]]:
    """Pick the latest timestamped results.json for each seed dir."""
    results = []
    for seed_dir in sorted(hidden_dir.glob('seed-*')):
        if not seed_dir.is_dir():
            continue
        run_dirs = sorted(p for p in seed_dir.iterdir() if p.is_dir())
        if not run_dirs:
            continue
        results_path = run_dirs[-1] / 'results.json'
        if results_path.exists():
            results.append((parse_seed(seed_dir), results_path))
    return results


def collect_runs(task_dir: Path) -> List[SeedRun]:
    """Return one SeedRun per (hidden_size, seed) with per-layer MDL values."""
    runs: List[SeedRun] = []
    for hidden_dir in sorted(task_dir.glob('hidden-*'), key=parse_hidden_size):
        if not hidden_dir.is_dir():
            continue
        hidden_size = parse_hidden_size(hidden_dir)
        for seed, results_path in latest_results_by_seed(hidden_dir):
            with results_path.open('r', encoding='utf-8') as fh:
                data = json.load(fh)
            layer_mdl = {int(k): float(v) for k, v in data['mdl_sum_ce'].items()}
            layer_acc = {int(k): float(v) for k, v in data['test_acc'].items()}
            runs.append(SeedRun(
                hidden_size=hidden_size,
                seed=seed,
                layer_mdl=layer_mdl,
                layer_test_acc=layer_acc,
            ))
    return runs


def summarize(runs: List[SeedRun]) -> Dict[int, List[Dict]]:
    """
    Returns:
        layer_summary[layer] = list of dicts sorted by hidden_size, each with:
            hidden_size, mean_mdl, stderr, mean_test_acc, acc_stderr
    """
    # Collect raw values: grouped[layer][hidden_size] = [mdl, ...]
    layers = sorted({l for r in runs for l in r.layer_mdl})
    hidden_sizes = sorted({r.hidden_size for r in runs})

    layer_summary: Dict[int, List[Dict]] = {}
    for layer in layers:
        rows = []
        for h in hidden_sizes:
            mdl_vals = [r.layer_mdl[layer] for r in runs
                        if r.hidden_size == h and layer in r.layer_mdl]
            acc_vals = [r.layer_test_acc[layer] for r in runs
                        if r.hidden_size == h and layer in r.layer_test_acc]
            if not mdl_vals:
                continue
            mean_mdl = statistics.mean(mdl_vals)
            stderr = (statistics.stdev(mdl_vals) / math.sqrt(len(mdl_vals))
                      if len(mdl_vals) > 1 else 0.0)
            mean_acc = statistics.mean(acc_vals) if acc_vals else float('nan')
            acc_stderr = (statistics.stdev(acc_vals) / math.sqrt(len(acc_vals))
                          if len(acc_vals) > 1 else 0.0)
            rows.append(dict(
                hidden_size=h,
                mean_mdl=mean_mdl,
                stderr=stderr,
                mean_test_acc=mean_acc,
                acc_stderr=acc_stderr,
                n_seeds=len(mdl_vals),
            ))
        layer_summary[layer] = rows
    return layer_summary


def plot_per_layer(
    task_dir: Path,
    out_dir: Path,
    metric: str = 'mdl',   # 'mdl' or 'acc'
) -> Path:
    model_name = task_dir.parent.name
    task_name = task_dir.name

    runs = collect_runs(task_dir)
    if not runs:
        raise FileNotFoundError(f'No results.json files found under {task_dir}')

    summary = summarize(runs)
    layers = sorted(summary)
    n_layers = len(layers)
    # One color per layer using a perceptually uniform colormap
    colors = cm.viridis(np.linspace(0, 1, n_layers))

    fig, ax = plt.subplots(figsize=(11, 7))

    for layer, color in zip(layers, colors):
        rows = summary[layer]
        xs = [r['hidden_size'] for r in rows]
        if metric == 'mdl':
            ys = [r['mean_mdl'] for r in rows]
            errs = [r['stderr'] for r in rows]
            ylabel = 'MDL (sum CE per layer)'
        else:
            ys = [r['mean_test_acc'] for r in rows]
            errs = [r['acc_stderr'] for r in rows]
            ylabel = 'Test Accuracy'

        ax.errorbar(
            xs, ys, yerr=errs,
            fmt='o-',
            color=color,
            capsize=3,
            linewidth=1.5,
            markersize=4,
            label=f'Layer {layer}',
        )

    ax.set_xscale('log', base=2)
    hidden_sizes = sorted({r.hidden_size for run in runs for r in [run]})
    ax.set_xticks(hidden_sizes)
    ax.set_xticklabels([str(h) for h in hidden_sizes], rotation=45)
    ax.set_xlabel('MLP Hidden Size')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{model_name} / {task_name}: Per-Layer {metric.upper()} vs Hidden Size')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=8, ncol=2)

    # Colorbar to show layer depth
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=layers[0], vmax=layers[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label('Layer index')

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'per_layer_{metric}_{model_name}_{task_name}.png'
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Per-layer MDL (or accuracy) vs hidden size plot.',
    )
    parser.add_argument('--task-dir', type=Path, required=True,
                        help='e.g. outputs/dino/car')
    parser.add_argument('--out-dir', type=Path, default=None,
                        help='Output directory (default: plots/per_layer/)')
    parser.add_argument('--metric', choices=['mdl', 'acc'], default='mdl',
                        help='Which metric to plot (default: mdl)')
    return parser.parse_args()


def main():
    args = parse_args()
    task_dir = args.task_dir.resolve()
    out_dir = (args.out_dir or Path(__file__).resolve().parent / 'plots' / 'per_layer').resolve()
    out_path = plot_per_layer(task_dir, out_dir, metric=args.metric)
    print(out_path)


if __name__ == '__main__':
    main()
