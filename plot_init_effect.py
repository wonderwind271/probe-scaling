from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")

RESULT_RE = re.compile(
    r"(?P<task>[^/]+)/hidden-(?P<hidden>[^/]+)/seed-(?P<seed>[^/]+)/gamma-(?P<gamma>[^/]+)/(?P<timestamp>[^/]+)/results\.json$"
)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent / "outputs" / "llama"
    parser = argparse.ArgumentParser(
        description="Plot initialization effects on probe MDL across hidden sizes."
    )
    parser.add_argument("--root", type=Path, default=default_root, help="Root directory containing task outputs.")
    parser.add_argument("--task", type=str, default="somo", help="Task subdirectory under the root.")
    parser.add_argument(
        "--metric",
        choices=["mdl_sum_ce", "mdl_nll"],
        default="mdl_sum_ce",
        help="Metric to average over layers before aggregating across seeds.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,142,242,342,442",
        help="Comma-separated list of seeds to include.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the figure. Defaults to <root>/<task>/init_effect_<metric>_by_hidden.png",
    )
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=None,
        help="Optional path to save aggregated summary statistics as JSON.",
    )
    return parser.parse_args()


def parse_seed_list(seed_text: str) -> List[int]:
    return [int(seed.strip()) for seed in seed_text.split(",") if seed.strip()]


def find_latest_results(task_root: Path) -> Dict[Tuple[int, int, float], Path]:
    latest: Dict[Tuple[int, int, float], Tuple[str, Path]] = {}
    for result_path in task_root.glob("hidden-*/seed-*/gamma-*/*/results.json"):
        rel = result_path.relative_to(task_root.parent).as_posix()
        match = RESULT_RE.match(rel)
        if not match:
            continue
        hidden = int(match.group("hidden"))
        seed = int(match.group("seed"))
        gamma = float(match.group("gamma"))
        timestamp = match.group("timestamp")
        key = (hidden, seed, gamma)
        prev = latest.get(key)
        if prev is None or timestamp > prev[0]:
            latest[key] = (timestamp, result_path)
    return {key: path for key, (_, path) in latest.items()}


def average_series_over_layers(result_path: Path, key: str) -> float:
    with result_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    series_by_layer = data[key]
    values = [float(value) for value in series_by_layer.values()]
    if not values:
        raise ValueError(f"No layer values found for key {key} in {result_path}")
    return float(np.mean(values))


def aggregate(
    latest_results: Dict[Tuple[int, int, float], Path],
    metric: str,
    seeds: Iterable[int],
) -> Dict[int, Dict[float, Dict[str, List[float]]]]:
    seed_set = set(seeds)
    grouped: Dict[int, Dict[float, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"metric": [], "acc": []})
    )
    for (hidden, seed, gamma), result_path in sorted(latest_results.items()):
        if seed not in seed_set:
            continue
        grouped[hidden][gamma]["metric"].append(average_series_over_layers(result_path, metric))
        grouped[hidden][gamma]["acc"].append(average_series_over_layers(result_path, "test_acc"))
    return grouped


def compute_se(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    std = float(np.std(values, ddof=1))
    return std / math.sqrt(len(values))


def build_summary(
    grouped: Dict[int, Dict[float, Dict[str, List[float]]]],
    expected_seeds: int,
) -> Dict[int, Dict[float, Dict[str, float]]]:
    summary: Dict[int, Dict[float, Dict[str, float]]] = {}
    for hidden, gamma_map in grouped.items():
        summary[hidden] = {}
        for gamma, stats in gamma_map.items():
            metric_values = stats["metric"]
            acc_values = stats["acc"]
            summary[hidden][gamma] = {
                "mean": float(np.mean(metric_values)),
                "se": compute_se(metric_values),
                "acc_mean": float(np.mean(acc_values)),
                "acc_se": compute_se(acc_values),
                "num_seeds": len(metric_values),
                "expected_seeds": expected_seeds,
            }
    return summary


def plot_summary(
    summary: Dict[int, Dict[float, Dict[str, float]]],
    metric: str,
    output_path: Path,
) -> None:
    hidden_sizes = sorted(summary)
    if not hidden_sizes:
        raise ValueError("No summary data available to plot.")

    n_panels = len(hidden_sizes)
    ncols = min(2, n_panels)
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6.2 * ncols, 4.3 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.2, 0.85, n_panels))

    ylabel = "Average MDL over layers" if metric == "mdl_sum_ce" else "Average MDL NLL over layers"

    for idx, hidden in enumerate(hidden_sizes):
        ax = axes[idx // ncols][idx % ncols]
        gamma_values = sorted(summary[hidden])
        means = [summary[hidden][gamma]["mean"] for gamma in gamma_values]
        errors = [summary[hidden][gamma]["se"] for gamma in gamma_values]
        gamma_arr = np.asarray(gamma_values, dtype=float)
        mean_arr = np.asarray(means, dtype=float)
        err_arr = np.asarray(errors, dtype=float)
        ax.plot(
            gamma_arr,
            mean_arr,
            marker="o",
            linewidth=2,
            color=colors[idx],
        )
        ax.fill_between(
            gamma_arr,
            mean_arr - err_arr,
            mean_arr + err_arr,
            color=colors[idx],
            alpha=0.22,
            linewidth=0,
        )
        ax.set_title(f"Hidden size = {hidden}")
        ax.set_xlabel("Initialization parameter gamma")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Initialization effect on probing (SE over seeds)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    task_ls = args.task.split(",") if "," in args.task else [args.task]

    latest_results: Dict[Tuple[int, int, float], Path] = {}
    for task in task_ls:
        task_root = args.root / task
        if not task_root.exists():
            raise FileNotFoundError(f"Task directory does not exist: {task_root}")

        seeds = parse_seed_list(args.seeds)
        latest_result = find_latest_results(task_root)
        if not latest_result:
            raise FileNotFoundError(f"No results.json files found under {task_root}")
        latest_results.update(latest_result)

    grouped = aggregate(latest_results, args.metric, seeds)
    expected_seeds = len(seeds)
    summary = build_summary(grouped, expected_seeds=expected_seeds)

    missing = []
    for hidden, gamma_map in sorted(summary.items()):
        for gamma, stats in sorted(gamma_map.items()):
            if stats["num_seeds"] != expected_seeds:
                missing.append((hidden, gamma, stats["num_seeds"]))
    if missing:
        print("Warning: missing seed runs for:")
        for hidden, gamma, count in missing:
            print(f"  hidden={hidden}, gamma={gamma}: found {count}/{expected_seeds}")

    output_path = args.output
    if output_path is None:
        output_path = args.root / task_ls[-1] / f"init_effect_{args.metric}_by_hidden.png"

    plot_summary(summary, metric=args.metric, output_path=output_path)
    print(f"Saved figure to {output_path}")

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            str(hidden): {
                str(gamma): stats for gamma, stats in sorted(gamma_map.items())
            }
            for hidden, gamma_map in sorted(summary.items())
        }
        args.summary_json.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        print(f"Saved summary to {args.summary_json}")


if __name__ == "__main__":
    main()
