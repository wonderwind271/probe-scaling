#!/usr/bin/env python3
"""Plot linear-probe separability (accuracy and MDL) versus layer.

This script scans `outputs/` for finished linear-probe runs, averages metrics
across seeds, and writes plots with both line and grouped-bar views.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


COLORS = {'clip': '#E87722', 'dino': '#0072B2'}
MODELS = ('clip', 'dino')


@dataclass
class SeedLayerMetrics:
    """Store one seed's per-layer metrics for one model/task run."""

    model: str
    task: str
    seed: int
    timestamp: str
    run_dir: Path
    layer_acc: Dict[int, float]
    layer_mdl: Dict[int, float]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Aggregate linear probe outputs and plot separability by layer.',
    )
    parser.add_argument(
        '--outputs-dir',
        type=Path,
        default=Path(__file__).resolve().parent / 'outputs',
        help='Root outputs directory containing model/task/seed runs.',
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=['chair', 'person', 'table'],
        help='Tasks to include.',
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=list(MODELS),
        help='Models to include.',
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path(__file__).resolve().parent / 'plots' / 'linear',
        help='Directory to save plots and summary JSON.',
    )
    return parser.parse_args()


def parse_layer_dict(raw: Dict[str, float]) -> Dict[int, float]:
    """Convert layer-keyed dict from JSON into int->float dict."""
    return {int(key): float(value) for key, value in raw.items()}


def iter_linear_results(outputs_dir: Path) -> Iterable[Path]:
    """Yield all results.json paths for linear-probe runs."""
    for path in outputs_dir.glob('*/*/hidden-*/seed-*/*/results.json'):
        yield path


def collect_latest_linear_runs(
    outputs_dir: Path,
    models: List[str],
    tasks: List[str],
) -> List[SeedLayerMetrics]:
    """Collect the latest linear run per (model, task, seed)."""
    wanted_models = set(models)
    wanted_tasks = set(tasks)
    latest: Dict[Tuple[str, str, int], Tuple[str, Path, dict]] = {}

    for path in iter_linear_results(outputs_dir):
        parts = path.parts
        model = parts[-6]
        task = parts[-5]
        if model not in wanted_models or task not in wanted_tasks:
            continue

        seed_str = parts[-3].replace('seed-', '')
        if not seed_str.isdigit():
            continue
        seed = int(seed_str)
        timestamp = parts[-2]

        with path.open('r', encoding='utf-8') as handle:
            data = json.load(handle)

        if str(data.get('probe_type', '')).lower() != 'linear':
            continue

        key = (model, task, seed)
        prev = latest.get(key)
        if prev is None or timestamp > prev[0]:
            latest[key] = (timestamp, path, data)

    runs: List[SeedLayerMetrics] = []
    for (model, task, seed), (timestamp, path, data) in sorted(latest.items()):
        runs.append(
            SeedLayerMetrics(
                model=model,
                task=task,
                seed=seed,
                timestamp=timestamp,
                run_dir=path.parent,
                layer_acc=parse_layer_dict(data['test_acc']),
                layer_mdl=parse_layer_dict(data['mdl_sum_ce']),
            ),
        )
    return runs


def stderr(values: List[float]) -> float:
    """Return standard error of a sample list."""
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def summarize_by_layer(runs: List[SeedLayerMetrics]) -> Dict[str, Dict[int, dict]]:
    """Return summary stats grouped by model then layer."""
    grouped: Dict[str, Dict[int, Dict[str, List[float]]]] = {}
    for run in runs:
        model_group = grouped.setdefault(run.model, {})
        all_layers = sorted(set(run.layer_acc) & set(run.layer_mdl))
        for layer in all_layers:
            layer_group = model_group.setdefault(layer, {'acc': [], 'mdl': []})
            layer_group['acc'].append(run.layer_acc[layer])
            layer_group['mdl'].append(run.layer_mdl[layer])

    summary: Dict[str, Dict[int, dict]] = {}
    for model, layer_map in grouped.items():
        summary[model] = {}
        for layer, metric_map in sorted(layer_map.items()):
            acc_values = metric_map['acc']
            mdl_values = metric_map['mdl']
            summary[model][layer] = {
                'mean_acc': float(np.mean(acc_values)),
                'stderr_acc': stderr(acc_values),
                'mean_mdl': float(np.mean(mdl_values)),
                'stderr_mdl': stderr(mdl_values),
                'num_seeds': len(acc_values),
            }
    return summary


def plot_one_task(
    task: str,
    task_summary: Dict[str, Dict[int, dict]],
    out_path: Path,
) -> None:
    """Draw 2x2 figure: line and bar plots for acc/MDL versus layer."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex='col')
    line_acc_ax, bar_acc_ax = axes[0]
    line_mdl_ax, bar_mdl_ax = axes[1]

    for model in MODELS:
        if model not in task_summary:
            continue
        layers = sorted(task_summary[model].keys())
        mean_acc = [task_summary[model][layer]['mean_acc'] for layer in layers]
        se_acc = [task_summary[model][layer]['stderr_acc'] for layer in layers]
        mean_mdl = [task_summary[model][layer]['mean_mdl'] for layer in layers]
        se_mdl = [task_summary[model][layer]['stderr_mdl'] for layer in layers]

        line_acc_ax.errorbar(
            layers,
            mean_acc,
            yerr=se_acc,
            fmt='o-',
            capsize=3,
            linewidth=2,
            color=COLORS[model],
            label=model.upper(),
        )
        line_mdl_ax.errorbar(
            layers,
            mean_mdl,
            yerr=se_mdl,
            fmt='o-',
            capsize=3,
            linewidth=2,
            color=COLORS[model],
            label=model.upper(),
        )

    all_layers = sorted(
        set().union(*(set(layer_map.keys()) for layer_map in task_summary.values())),
    )
    x = np.arange(len(all_layers), dtype=float)
    width = 0.36
    for idx, model in enumerate(MODELS):
        if model not in task_summary:
            continue
        layer_map = task_summary[model]
        mean_acc = [layer_map[layer]['mean_acc'] for layer in all_layers]
        mean_mdl = [layer_map[layer]['mean_mdl'] for layer in all_layers]
        offset = (idx - 0.5) * width
        bar_acc_ax.bar(
            x + offset,
            mean_acc,
            width=width,
            color=COLORS[model],
            alpha=0.85,
            label=model.upper(),
        )
        bar_mdl_ax.bar(
            x + offset,
            mean_mdl,
            width=width,
            color=COLORS[model],
            alpha=0.85,
            label=model.upper(),
        )

    line_acc_ax.set_title('Accuracy vs layer (mean ± s.e.)')
    line_mdl_ax.set_title('MDL vs layer (mean ± s.e.)')
    bar_acc_ax.set_title('Accuracy vs layer (bar)')
    bar_mdl_ax.set_title('MDL vs layer (bar)')

    line_acc_ax.set_ylabel('Test accuracy')
    line_mdl_ax.set_ylabel('MDL (sum CE)')
    line_mdl_ax.set_xlabel('Layer')
    bar_mdl_ax.set_xlabel('Layer')
    bar_acc_ax.set_ylabel('Test accuracy')
    bar_mdl_ax.set_ylabel('MDL (sum CE)')

    for ax in (line_acc_ax, line_mdl_ax):
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.legend()

    for ax in (bar_acc_ax, bar_mdl_ax):
        ax.set_xticks(x, [str(layer) for layer in all_layers])
        ax.grid(True, axis='y', linestyle='--', alpha=0.35)
        ax.legend()

    fig.suptitle(f'Linear separability by layer: {task}', fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_summary_payload(
    runs: List[SeedLayerMetrics],
    per_task_summary: Dict[str, Dict[str, Dict[int, dict]]],
) -> dict:
    """Build JSON payload with source runs and aggregated statistics."""
    run_rows = [
        {
            'model': run.model,
            'task': run.task,
            'seed': run.seed,
            'timestamp': run.timestamp,
            'run_dir': str(run.run_dir),
        }
        for run in runs
    ]
    return {
        'runs': run_rows,
        'summary': per_task_summary,
    }


def main() -> None:
    """Run linear-result aggregation and emit plots plus JSON summary."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_latest_linear_runs(
        outputs_dir=args.outputs_dir,
        models=args.models,
        tasks=args.tasks,
    )
    if not runs:
        raise FileNotFoundError('No linear-probe results found in outputs directory.')

    by_task: Dict[str, List[SeedLayerMetrics]] = {}
    for run in runs:
        by_task.setdefault(run.task, []).append(run)

    per_task_summary: Dict[str, Dict[str, Dict[int, dict]]] = {}
    for task in args.tasks:
        task_runs = by_task.get(task, [])
        if not task_runs:
            continue
        task_summary = summarize_by_layer(task_runs)
        per_task_summary[task] = task_summary
        plot_one_task(
            task=task,
            task_summary=task_summary,
            out_path=args.out_dir / f'linear_separability_{task}.png',
        )

    overall_summary = summarize_by_layer(runs)
    per_task_summary['overall'] = overall_summary
    plot_one_task(
        task='overall',
        task_summary=overall_summary,
        out_path=args.out_dir / 'linear_separability_overall.png',
    )

    summary_payload = build_summary_payload(runs, per_task_summary)
    summary_path = args.out_dir / 'linear_separability_summary.json'
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary_payload, handle, indent=2)

    print(f'Saved {summary_path}')
    for task in sorted(per_task_summary):
        print(f'[done] {task}')


if __name__ == '__main__':
    main()
