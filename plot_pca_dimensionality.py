#!/usr/bin/env python3
"""Plot delta-signal and total-variance compactness for CLIP and DINO.

This script computes only two outputs:
1) `k90`: smallest number of PCA dimensions needed to explain 90% of
   `delta = mean(X_pos) - mean(X_neg)`.
2) For the first `max_k` dimensions (default 300), the cumulative explained
   fraction of `delta`.

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


COLORS = {'clip': '#E87722', 'dino': '#0072B2'}
MODELS = ('clip', 'dino')


def load_cls_tokens(
    pos_dir: Path,
    neg_dir: Path,
    layer_idx: int,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return feature matrix and labels from cached layer tensors."""
    pos_path = pos_dir / f'layer_{layer_idx:02d}.pt'
    neg_path = neg_dir / f'layer_{layer_idx:02d}.pt'

    pos_npy = pos_dir / f'layer_{layer_idx:02d}_cls.npy'
    neg_npy = neg_dir / f'layer_{layer_idx:02d}_cls.npy'
    if pos_npy.exists() and neg_npy.exists():
        pos_cls = np.load(pos_npy)
        neg_cls = np.load(neg_npy)
    else:
        try:
            pos_tensor = torch.load(pos_path, map_location='cpu', mmap=True)
            neg_tensor = torch.load(neg_path, map_location='cpu', mmap=True)
        except TypeError:
            pos_tensor = torch.load(pos_path, map_location='cpu')
            neg_tensor = torch.load(neg_path, map_location='cpu')
        pos_cls = pos_tensor[:, 0, :].float().numpy()
        neg_cls = neg_tensor[:, 0, :].float().numpy()
        del pos_tensor, neg_tensor
        gc.collect()

    rng = np.random.default_rng(seed)
    if max_samples is not None:
        n_each = max_samples // 2
        if len(pos_cls) > n_each:
            pos_cls = pos_cls[rng.choice(len(pos_cls), n_each, replace=False)]
        if len(neg_cls) > n_each:
            neg_cls = neg_cls[rng.choice(len(neg_cls), n_each, replace=False)]

    features = np.concatenate([pos_cls, neg_cls], axis=0)
    labels = np.concatenate([
        np.ones(len(pos_cls), dtype=np.int64),
        np.zeros(len(neg_cls), dtype=np.int64),
    ])
    return features, labels


def delta_cumulative_fraction(
    features: np.ndarray,
    labels: np.ndarray,
    max_k: int,
) -> np.ndarray:
    """Return cumulative explained fraction of delta in top-k PCA directions."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    mu_pos = scaled[labels == 1].mean(axis=0)
    mu_neg = scaled[labels == 0].mean(axis=0)
    delta = mu_pos - mu_neg
    delta_norm_sq = float(np.dot(delta, delta))

    if delta_norm_sq <= 0.0:
        n_components = min(max_k, scaled.shape[0] - 1, scaled.shape[1])
        return np.zeros(max(n_components, 1), dtype=np.float64)

    n_components = min(max_k, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    pca.fit(scaled)

    components = pca.components_
    proj_sq = (components @ delta) ** 2
    cumulative = np.cumsum(proj_sq / delta_norm_sq)
    return np.clip(cumulative, 0.0, 1.0)


def total_cumulative_variance(
    features: np.ndarray,
    max_k: int,
) -> np.ndarray:
    """Return cumulative explained variance ratio for all features."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    n_components = min(max_k, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    pca.fit(scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return np.clip(cumulative, 0.0, 1.0)


def k_at_threshold(cumulative: np.ndarray, threshold: float = 0.9) -> int:
    """Return 1-based k where cumulative[k-1] first reaches threshold."""
    hit = np.where(cumulative >= threshold)[0]
    return int(hit[0]) + 1 if len(hit) else int(len(cumulative))


def collect_model_task(
    vlm_lens_dir: Path,
    model: str,
    task: str,
    layers: List[int],
    max_samples: int,
    max_k: int,
) -> Dict[int, dict]:
    """Collect per-layer cumulative delta fractions and k90 values."""
    pos_dir = vlm_lens_dir / f'{model}-{task}-pos'
    neg_dir = vlm_lens_dir / f'{model}-{task}-neg'
    if not pos_dir.exists() or not neg_dir.exists():
        print(f'[skip] {model}/{task}: cache dirs missing')
        return {}

    results: Dict[int, dict] = {}
    for layer in layers:
        if not (pos_dir / f'layer_{layer:02d}.pt').exists():
            continue
        print(f'  {model}/{task} layer {layer:02d}...', end=' ', flush=True)
        features, labels = load_cls_tokens(
            pos_dir=pos_dir,
            neg_dir=neg_dir,
            layer_idx=layer,
            max_samples=max_samples,
        )
        cumulative_delta = delta_cumulative_fraction(features, labels, max_k=max_k)
        cumulative_total = total_cumulative_variance(features, max_k=max_k)
        results[layer] = {
            'cum_delta': cumulative_delta,
            'k90_delta': k_at_threshold(cumulative_delta, threshold=0.9),
            'cum_total': cumulative_total,
            'k90_total': k_at_threshold(cumulative_total, threshold=0.9),
        }
        print(
            f'N={len(features)} '
            f'k90_delta={results[layer]["k90_delta"]} '
            f'k90_total={results[layer]["k90_total"]}',
        )
    return results


def plot_cumulative_grid(
    all_results: Dict[str, Dict[str, Dict[int, dict]]],
    tasks: List[str],
    layers: List[int],
    out_path: Path,
    metric_key: str,
    title: str,
    ylabel: str,
    k90_key: str,
) -> None:
    """Plot cumulative explained fraction/variance for first `max_k` dimensions."""
    n_rows = len(layers)
    n_cols = len(tasks)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 3.0 * n_rows),
        sharex=True,
        sharey=True,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, layer in enumerate(layers):
        for col, task in enumerate(tasks):
            ax = axes[row, col]
            for model in MODELS:
                data = all_results.get(model, {}).get(task, {}).get(layer)
                if data is None:
                    continue
                cumulative = data[metric_key]
                x = np.arange(1, len(cumulative) + 1)
                ax.plot(
                    x,
                    cumulative,
                    linewidth=1.8,
                    color=COLORS[model],
                    label=f'{model.upper()} (k90={data[k90_key]})',
                )

            ax.axhline(0.9, color='gray', linestyle='--', linewidth=0.8)
            ax.set_ylim(0.0, 1.02)
            ax.grid(True, linestyle='--', alpha=0.35)
            if row == 0:
                ax.set_title(task)
            if col == 0:
                ax.set_ylabel(f'Layer {layer}\n{ylabel}')
            if row == n_rows - 1:
                ax.set_xlabel('PCA dimension index (1..300)')
            if row == 0 and col == n_cols - 1:
                ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f'Saved {out_path}')


def plot_k90_vs_layer(
    all_results: Dict[str, Dict[str, Dict[int, dict]]],
    tasks: List[str],
    out_path: Path,
    k90_key: str,
    title: str,
    ylabel: str,
) -> None:
    """Plot k90 (number of dimensions for 90% delta) versus layer."""
    task_markers = ['o', 's', '^', 'D', 'v']
    task_linestyles = ['-', '--', '-.', ':']

    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    for model in MODELS:
        for idx, task in enumerate(tasks):
            layer_map = all_results.get(model, {}).get(task, {})
            if not layer_map:
                continue
            layers = sorted(layer_map.keys())
            k90_values = [layer_map[layer][k90_key] for layer in layers]
            ax.plot(
                layers,
                k90_values,
                color=COLORS[model],
                linestyle=task_linestyles[idx % len(task_linestyles)],
                marker=task_markers[idx % len(task_markers)],
                linewidth=1.8,
                markersize=5,
                label=f'{model.upper()} / {task}',
            )

    ax.set_xlabel('Layer')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f'Saved {out_path}')


def build_summary(
    all_results: Dict[str, Dict[str, Dict[int, dict]]],
    tasks: List[str],
    layers: List[int],
) -> dict:
    """Build JSON summary containing only k90 and cumulative curves."""
    summary: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for model in MODELS:
        summary[model] = {}
        for task in tasks:
            summary[model][task] = {}
            for layer in layers:
                data = all_results.get(model, {}).get(task, {}).get(layer)
                if data is None:
                    continue
                summary[model][task][str(layer)] = {
                    'k90_delta': int(data['k90_delta']),
                    'k90_total': int(data['k90_total']),
                    'cum_delta': [float(x) for x in data['cum_delta']],
                    'cum_total': [float(x) for x in data['cum_total']],
                }
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line args."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--vlm-lens-dir',
        type=Path,
        default=Path('/home/xxluo/projects/aip-fredashi/xxluo/myproject/vlm-lens'),
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=['chair', 'person', 'table'],
    )
    parser.add_argument(
        '--layers',
        nargs='+',
        type=int,
        default=list(range(12)),
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--max-k',
        type=int,
        default=300,
        help='Number of PCA dimensions for cumulative curve.',
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path(__file__).resolve().parent / 'plots' / 'pca',
    )
    return parser.parse_args()


def main() -> None:
    """Run simplified delta-dimensionality analysis and plotting."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict[str, Dict[int, dict]]] = {}
    for model in MODELS:
        all_results[model] = {}
        for task in args.tasks:
            print(f'\n=== {model.upper()} / {task} ===')
            all_results[model][task] = collect_model_task(
                vlm_lens_dir=args.vlm_lens_dir,
                model=model,
                task=task,
                layers=args.layers,
                max_samples=args.max_samples,
                max_k=args.max_k,
            )

    plot_cumulative_grid(
        all_results=all_results,
        tasks=args.tasks,
        layers=args.layers,
        out_path=args.out_dir / 'pca_delta_cumvar_grid.png',
        metric_key='cum_delta',
        title='Cumulative explained fraction of delta = X_pos - X_neg',
        ylabel='Cumulative delta fraction',
        k90_key='k90_delta',
    )
    plot_cumulative_grid(
        all_results=all_results,
        tasks=args.tasks,
        layers=args.layers,
        out_path=args.out_dir / 'pca_total_cumvar_grid.png',
        metric_key='cum_total',
        title='Cumulative explained total variance of X_pos + X_neg',
        ylabel='Cumulative total variance',
        k90_key='k90_total',
    )
    plot_k90_vs_layer(
        all_results=all_results,
        tasks=args.tasks,
        out_path=args.out_dir / 'pca_delta_k90_vs_layer.png',
        k90_key='k90_delta',
        title='Dimensions needed to explain 90% of delta = X_pos - X_neg',
        ylabel='k90 delta (dimensions)',
    )
    plot_k90_vs_layer(
        all_results=all_results,
        tasks=args.tasks,
        out_path=args.out_dir / 'pca_total_k90_vs_layer.png',
        k90_key='k90_total',
        title='Dimensions needed to explain 90% of total variance',
        ylabel='k90 total (dimensions)',
    )

    summary = build_summary(
        all_results=all_results,
        tasks=args.tasks,
        layers=args.layers,
    )
    summary_path = args.out_dir / 'pca_delta_summary.json'
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    print(f'Saved {summary_path}')


if __name__ == '__main__':
    main()
