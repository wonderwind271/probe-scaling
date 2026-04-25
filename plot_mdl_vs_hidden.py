#!/usr/bin/env python3
'''Plot mean layer-averaged MDL and accuracy versus hidden size.'''

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


@dataclass
class SeedRun:
    '''Store one seed-specific aggregated run result.'''

    hidden_size: int
    seed: int
    run_dir: Path
    mean_layer_mdl: float
    mean_layer_test_acc: float


def parse_args():
    '''Parse command-line arguments for plotting one model/task directory.'''
    parser = argparse.ArgumentParser(
        description='Plot MDL versus hidden size for one outputs folder.',
    )
    parser.add_argument(
        '--task-dir',
        type=Path,
        required=True,
        help='Path to one model/task directory such as outputs/clip/car.',
    )
    return parser.parse_args()


def parse_hidden_size(path: Path) -> int:
    '''Extract hidden size from a folder name like hidden-128.'''
    return int(path.name.replace('hidden-', ''))


def parse_seed(path: Path) -> int:
    '''Extract seed value from a folder name like seed-42.'''
    return int(path.name.replace('seed-', ''))


def latest_results_by_seed(hidden_dir: Path) -> List[Tuple[int, Path]]:
    '''Pick the latest timestamped results file for each seed directory.'''
    results = []
    for seed_dir in sorted(hidden_dir.glob('seed-*')):
        if not seed_dir.is_dir():
            continue
        run_dirs = sorted(path for path in seed_dir.iterdir() if path.is_dir())
        if not run_dirs:
            continue
        latest_run = run_dirs[-1]
        results_path = latest_run / 'results.json'
        if results_path.exists():
            results.append((parse_seed(seed_dir), results_path))
    return results


def load_run_metrics(results_path: Path) -> Tuple[float, float]:
    '''Load one results.json and average MDL and test accuracy across layers.'''
    with results_path.open('r', encoding='utf-8') as handle:
        results = json.load(handle)
    layer_mdls = list(results['mdl_sum_ce'].values())
    layer_accs = list(results['test_acc'].values())
    return (
        float(sum(layer_mdls) / len(layer_mdls)),
        float(sum(layer_accs) / len(layer_accs)),
    )


def collect_runs(task_dir: Path) -> List[SeedRun]:
    '''Collect one aggregated MDL value per hidden size and seed.'''
    runs: List[SeedRun] = []
    for hidden_dir in sorted(task_dir.glob('hidden-*'), key=parse_hidden_size):
        if not hidden_dir.is_dir():
            continue
        hidden_size = parse_hidden_size(hidden_dir)
        for seed, results_path in latest_results_by_seed(hidden_dir):
            mean_layer_mdl, mean_layer_test_acc = load_run_metrics(results_path)
            runs.append(
                SeedRun(
                    hidden_size=hidden_size,
                    seed=seed,
                    run_dir=results_path.parent,
                    mean_layer_mdl=mean_layer_mdl,
                    mean_layer_test_acc=mean_layer_test_acc,
                ),
            )
    return runs


def summarize_runs(runs: List[SeedRun]) -> List[Dict]:
    '''Compute mean and standard error across seeds for each hidden size.'''
    grouped: Dict[int, List[SeedRun]] = {}
    for run in runs:
        grouped.setdefault(run.hidden_size, []).append(run)

    summary = []
    for hidden_size in sorted(grouped):
        mdl_values = [run.mean_layer_mdl for run in grouped[hidden_size]]
        acc_values = [run.mean_layer_test_acc for run in grouped[hidden_size]]
        mean_value = statistics.mean(mdl_values)
        mean_acc = statistics.mean(acc_values)
        if len(mdl_values) > 1:
            stderr = statistics.stdev(mdl_values) / math.sqrt(len(mdl_values))
            acc_stderr = statistics.stdev(acc_values) / math.sqrt(len(acc_values))
        else:
            stderr = 0.0
            acc_stderr = 0.0
        summary.append(
            {
                'hidden_size': hidden_size,
                'mean_mdl': mean_value,
                'stderr': stderr,
                'mean_test_acc': mean_acc,
                'test_acc_stderr': acc_stderr,
                'num_seeds': len(mdl_values),
                'seed_values': {
                    str(run.seed): run.mean_layer_mdl
                    for run in sorted(grouped[hidden_size], key=lambda item: item.seed)
                },
                'seed_test_accs': {
                    str(run.seed): run.mean_layer_test_acc
                    for run in sorted(grouped[hidden_size], key=lambda item: item.seed)
                },
            },
        )
    return summary


def build_output_paths(task_dir: Path) -> Tuple[Path, Path, Path]:
    '''Build root-level output paths under probe-scaling.'''
    probe_root = Path(__file__).resolve().parent
    model_name = task_dir.parent.name
    task_name = task_dir.name
    mdl_figure_path = probe_root / 'plots' / f'mdl_vs_hidden_{model_name}_{task_name}.png'
    acc_figure_path = probe_root / 'plots' / f'acc_vs_hidden_{model_name}_{task_name}.png'
    summary_path = probe_root / 'plots'/ f'mdl_vs_hidden_{model_name}_{task_name}.json'
    return mdl_figure_path, acc_figure_path, summary_path


def plot_task_dir(task_dir: Path) -> Tuple[Path, Path]:
    '''Aggregate one model/task folder and write MDL/accuracy plots plus summary JSON.'''
    model_name = task_dir.parent.name
    task_name = task_dir.name
    runs = collect_runs(task_dir)
    if not runs:
        raise FileNotFoundError(f'No results.json files found under {task_dir}')

    summary = summarize_runs(runs)
    hidden_sizes = [row['hidden_size'] for row in summary]
    mean_mdls = [row['mean_mdl'] for row in summary]
    mdl_stderrs = [row['stderr'] for row in summary]
    mean_accs = [row['mean_test_acc'] for row in summary]
    acc_stderrs = [row['test_acc_stderr'] for row in summary]
    mdl_figure_path, acc_figure_path, summary_path = build_output_paths(task_dir)

    plt.figure(figsize=(9, 6))
    plt.errorbar(
        hidden_sizes,
        mean_mdls,
        yerr=mdl_stderrs,
        fmt='o-',
        capsize=4,
        linewidth=2,
        markersize=6,
    )
    plt.xscale('log', base=2)
    plt.xticks(hidden_sizes, [str(size) for size in hidden_sizes], rotation=45)
    plt.xlabel('Hidden Size')
    plt.ylabel('Mean MDL Across Layers')
    plt.title(f'{model_name} / {task_name}: Mean Layer MDL vs Hidden Size')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(mdl_figure_path, dpi=200)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.errorbar(
        hidden_sizes,
        mean_accs,
        yerr=acc_stderrs,
        fmt='o-',
        capsize=4,
        linewidth=2,
        markersize=6,
    )
    plt.xscale('log', base=2)
    plt.xticks(hidden_sizes, [str(size) for size in hidden_sizes], rotation=45)
    plt.xlabel('Hidden Size')
    plt.ylabel('Mean Test Accuracy Across Layers')
    plt.title(f'{model_name} / {task_name}: Mean Layer Accuracy vs Hidden Size')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(acc_figure_path, dpi=200)
    plt.close()

    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    return mdl_figure_path, acc_figure_path


def main():
    '''Generate MDL/accuracy plots for the requested model/task directory.'''
    args = parse_args()
    mdl_plot, acc_plot = plot_task_dir(args.task_dir.resolve())
    print(mdl_plot)
    print(acc_plot)


if __name__ == '__main__':
    main()
