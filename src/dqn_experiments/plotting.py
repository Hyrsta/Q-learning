from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


ALGORITHM_ORDER = ("dqn", "double_dqn", "dueling_dqn")
ALGORITHM_LABELS = {
    "dqn": "DQN",
    "double_dqn": "Double-DQN",
    "dueling_dqn": "Dueling-DQN",
}
ALGORITHM_COLORS = {
    "dqn": "#1f77b4",
    "double_dqn": "#ff7f0e",
    "dueling_dqn": "#2ca02c",
}

ENV_PANELS = (
    "CartPole-v1",
    "LunarLander-v3",
    "BreakoutNoFrameskip-v4",
    "PongNoFrameskip-v4",
)
ENV_DISPLAY_NAMES = {
    "CartPole-v1": "CartPole",
    "LunarLander-v3": "LunarLander",
    "BreakoutNoFrameskip-v4": "Breakout",
    "PongNoFrameskip-v4": "Pong",
}

DEFAULT_TARGET_SCORES = {
    "CartPole-v1": 475.0,
    "LunarLander-v3": 200.0,
    "BreakoutNoFrameskip-v4": 400.0,
    "PongNoFrameskip-v4": 18.0,
}


@dataclass
class RunRecord:
    env_id: str
    algo: str
    seed: Optional[int]
    run_dir: Path
    training_metrics: List[Dict[str, float]]
    evaluations: List[Dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot aggregated metrics for DQN experiments")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Directory containing training outputs")
    parser.add_argument("--output-dir", type=str, default="figures", help="Where to save the generated figures")
    parser.add_argument(
        "--target-score",
        action="append",
        default=[],
        metavar="ENV=VALUE",
        help="Override default target scores used for the sample efficiency plot",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure resolution when saving to disk")
    parser.add_argument("--show", action="store_true", help="Display the figures interactively after saving")
    return parser.parse_args()


def parse_target_overrides(overrides: Iterable[str]) -> Dict[str, float]:
    targets = DEFAULT_TARGET_SCORES.copy()
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid target override '{override}'. Expected format ENV=VALUE")
        env_id, value = override.split("=", 1)
        env_id = env_id.strip()
        try:
            targets[env_id] = float(value)
        except ValueError as exc:
            raise ValueError(f"Could not parse target value in '{override}'") from exc
    return targets


def load_metadata(run_dir: Path) -> Dict[str, object]:
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    checkpoint_path = run_dir / "model.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "metadata" in checkpoint:
            return checkpoint["metadata"]
    raise FileNotFoundError(f"Could not find metadata for run at {run_dir}")


def load_json(path: Path) -> List[Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, found {type(data)}")
    return data


def collect_runs(root: Path) -> List[RunRecord]:
    runs: List[RunRecord] = []
    if not root.exists():
        return runs
    for training_path in sorted(root.rglob("training_metrics.json")):
        run_dir = training_path.parent
        try:
            metadata = load_metadata(run_dir)
        except FileNotFoundError:
            continue
        env_id = metadata.get("env_id")
        algo = metadata.get("algo")
        seed_value = metadata.get("seed")
        seed: Optional[int]
        if isinstance(seed_value, (int, float)):
            seed = int(seed_value)
        elif isinstance(seed_value, str):
            try:
                seed = int(seed_value)
            except ValueError:
                seed = None
        else:
            seed = None
        if not env_id or not algo:
            continue
        if algo not in ALGORITHM_ORDER:
            continue
        training_metrics = load_json(training_path)
        eval_path = run_dir / "evaluation_metrics.json"
        evaluations: List[Dict[str, float]] = []
        if eval_path.exists():
            evaluations = load_json(eval_path)
        runs.append(
            RunRecord(
                env_id=str(env_id),
                algo=str(algo),
                seed=seed,
                run_dir=run_dir,
                training_metrics=training_metrics,
                evaluations=evaluations,
            )
        )
    return runs


def group_runs(runs: Iterable[RunRecord]) -> Dict[str, Dict[str, List[RunRecord]]]:
    grouped: Dict[str, Dict[str, List[RunRecord]]] = defaultdict(lambda: defaultdict(list))
    for run in runs:
        grouped[run.env_id][run.algo].append(run)
    return grouped


def _extract_episode_series(run: RunRecord) -> Optional[tuple[np.ndarray, np.ndarray]]:
    cumulative_steps: List[float] = []
    returns: List[float] = []
    total_steps = 0.0
    for entry in run.training_metrics:
        length_value: Optional[float] = None
        for key in ("l", "length", "episode_length", "steps"):
            if key in entry:
                try:
                    length_value = float(entry[key])
                except (TypeError, ValueError):
                    length_value = None
                break
        if length_value is None or length_value <= 0:
            continue
        return_value: Optional[float] = None
        for key in ("r", "return", "episode_return", "reward"):
            if key in entry:
                try:
                    return_value = float(entry[key])
                except (TypeError, ValueError):
                    return_value = None
                break
        if return_value is None or np.isnan(return_value):
            continue
        total_steps += length_value
        cumulative_steps.append(total_steps)
        returns.append(return_value)
    if not cumulative_steps:
        return None
    steps_array = np.asarray(cumulative_steps, dtype=float)
    returns_array = np.asarray(returns, dtype=float)
    if steps_array[0] > 0:
        steps_array = np.insert(steps_array, 0, 0.0)
        returns_array = np.insert(returns_array, 0, returns_array[0])
    return steps_array, returns_array


def _extract_evaluation_series(run: RunRecord) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if not run.evaluations:
        return None
    filtered = [entry for entry in run.evaluations if "step" in entry and "mean_return" in entry]
    if not filtered:
        return None
    filtered.sort(key=lambda item: float(item["step"]))
    steps = np.asarray([float(entry["step"]) for entry in filtered], dtype=float)
    values = np.asarray([float(entry.get("mean_return", np.nan)) for entry in filtered], dtype=float)
    mask = ~np.isnan(values)
    steps = steps[mask]
    values = values[mask]
    if steps.size == 0:
        return None
    if steps[0] > 0:
        steps = np.insert(steps, 0, 0.0)
        values = np.insert(values, 0, values[0])
    return steps, values


def aggregate_learning_curve(runs: List[RunRecord]) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not runs:
        return None
    series: List[tuple[np.ndarray, np.ndarray]] = []
    for run in runs:
        data = _extract_evaluation_series(run)
        if data is None:
            data = _extract_episode_series(run)
        if data is None:
            continue
        series.append(data)
    if not series:
        return None
    step_grid = sorted({float(step) for steps, _ in series for step in steps})
    if not step_grid:
        return None
    grid_array = np.asarray(step_grid, dtype=float)
    values = np.full((len(series), len(grid_array)), np.nan, dtype=float)
    for row, (steps, returns) in enumerate(series):
        if steps.size == 0:
            continue
        order = np.argsort(steps)
        steps = steps[order]
        returns = returns[order]
        steps, unique_indices = np.unique(steps, return_index=True)
        returns = returns[unique_indices]
        if steps.size == 1:
            interp = np.full_like(grid_array, np.nan, dtype=float)
            interp[grid_array <= steps[0]] = returns[0]
        else:
            interp = np.interp(grid_array, steps, returns)
            interp[grid_array > steps[-1]] = np.nan
        values[row] = interp
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    counts = np.sum(~np.isnan(values), axis=0)
    ci = np.zeros_like(mean)
    valid = counts > 1
    ci[valid] = 1.96 * std[valid] / np.sqrt(counts[valid])
    return grid_array, mean, ci


def steps_to_target(run: RunRecord, target: float) -> float:
    if not run.evaluations:
        return np.nan
    for entry in sorted(run.evaluations, key=lambda item: item.get("step", float("inf"))):
        if float(entry.get("mean_return", float("nan"))) >= target:
            return float(entry.get("step", np.nan))
    return np.nan


def final_returns(runs: List[RunRecord]) -> List[float]:
    finals: List[float] = []
    for run in runs:
        if not run.evaluations:
            continue
        last = max(run.evaluations, key=lambda item: item.get("step", -float("inf")))
        value = float(last.get("mean_return", np.nan))
        if not np.isnan(value):
            finals.append(value)
    return finals


def plot_learning_curves(grouped_runs: Dict[str, Dict[str, List[RunRecord]]], output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    axes = axes.flatten()
    for idx, env_id in enumerate(ENV_PANELS):
        ax = axes[idx]
        env_runs = grouped_runs.get(env_id, {})
        for algo in ALGORITHM_ORDER:
            runs = env_runs.get(algo, [])
            aggregate = aggregate_learning_curve(runs)
            if aggregate is None:
                continue
            steps, mean, ci = aggregate
            color = ALGORITHM_COLORS.get(algo, None)
            ax.plot(steps, mean, label=ALGORITHM_LABELS.get(algo, algo), color=color, linewidth=2.0)
            if np.any(ci > 0):
                ax.fill_between(
                    steps,
                    mean - ci,
                    mean + ci,
                    alpha=0.12,
                    color=color,
                    linewidth=0,
                )
        ax.set_title(ENV_DISPLAY_NAMES.get(env_id, env_id))
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Evaluation return")
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "learning_curves.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_sample_efficiency(
    grouped_runs: Dict[str, Dict[str, List[RunRecord]]],
    output_dir: Path,
    dpi: int,
    targets: Dict[str, float],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    env_indices = np.arange(len(ENV_PANELS))
    width = 0.2
    for offset, algo in enumerate(ALGORITHM_ORDER):
        heights = []
        for env_id in ENV_PANELS:
            runs = grouped_runs.get(env_id, {}).get(algo, [])
            target = targets.get(env_id)
            if target is None or not runs:
                heights.append(np.nan)
                continue
            steps = np.array([steps_to_target(run, target) for run in runs], dtype=float)
            if np.all(np.isnan(steps)):
                heights.append(np.nan)
            else:
                heights.append(float(np.nanmean(steps)))
        positions = env_indices + (offset - 1) * width
        bars = ax.bar(
            positions,
            heights,
            width=width,
            label=ALGORITHM_LABELS.get(algo, algo),
            color=ALGORITHM_COLORS.get(algo),
            alpha=0.85,
        )
        for bar, height in zip(bars, heights):
            if np.isnan(height):
                bar.set_visible(False)
    ax.set_xticks(env_indices)
    ax.set_xticklabels([ENV_DISPLAY_NAMES.get(env, env) for env in ENV_PANELS])
    ax.set_ylabel("Steps to reach target score")
    ax.set_ylim(bottom=0)
    ax.set_title("Sample efficiency across environments")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "sample_efficiency.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_stability(grouped_runs: Dict[str, Dict[str, List[RunRecord]]], output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.flatten()
    for idx, env_id in enumerate(ENV_PANELS):
        ax = axes[idx]
        data = []
        labels = []
        colors = []
        for algo in ALGORITHM_ORDER:
            returns = final_returns(grouped_runs.get(env_id, {}).get(algo, []))
            if returns:
                data.append(returns)
                labels.append(ALGORITHM_LABELS.get(algo, algo))
                colors.append(ALGORITHM_COLORS.get(algo))
        if data:
            box = ax.boxplot(
                data,
                tick_labels=labels,
                patch_artist=True,
                showmeans=True,
                meanline=True,
            )
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            for median in box["medians"]:
                median.set_color("black")
            for mean in box["means"]:
                mean.set_color("black")
        ax.set_title(ENV_DISPLAY_NAMES.get(env_id, env_id))
        ax.set_ylabel("Final evaluation return")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "stability.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs = collect_runs(Path(args.runs_dir))
    if not runs:
        raise SystemExit(f"No training runs with metrics found under {args.runs_dir}")
    grouped = group_runs(runs)
    targets = parse_target_overrides(args.target_score)
    output_dir = Path(args.output_dir)
    plot_learning_curves(grouped, output_dir, args.dpi)
    plot_sample_efficiency(grouped, output_dir, args.dpi, targets)
    plot_stability(grouped, output_dir, args.dpi)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
