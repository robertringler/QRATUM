"""Reporting utilities for benchmark results."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from quasim.ownai.eval.benchmark import BenchmarkResult
from quasim.ownai.train.metrics import compute_stability_margin


def results_to_dict_list(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    """Convert benchmark results to list of dictionaries.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Benchmark results

    Returns
    -------
    list[dict]
        List of result dictionaries
    """
    return [
        {
            "task": r.task,
            "model": r.model_name,
            "dataset": r.dataset,
            "seed": r.seed,
            "primary_metric": r.primary_metric,
            "secondary_metric": r.secondary_metric,
            "latency_p50_ms": r.latency_p50,
            "latency_p95_ms": r.latency_p95,
            "throughput": r.throughput,
            "model_size_mb": r.model_size_mb,
            "energy_proxy": r.energy_proxy,
            "prediction_hash": r.prediction_hash,
        }
        for r in results
    ]


def save_results_csv(results: list[BenchmarkResult], output_path: Path) -> None:
    """Save results to CSV file.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Benchmark results
    output_path : Path
        Output CSV file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dict_list = results_to_dict_list(results)

    if not dict_list:
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dict_list[0].keys())
        writer.writeheader()
        writer.writerows(dict_list)


def save_results_json(results: list[BenchmarkResult], output_path: Path) -> None:
    """Save results to JSON file.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Benchmark results
    output_path : Path
        Output JSON file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dict_list = results_to_dict_list(results)

    with open(output_path, "w") as f:
        json.dump(dict_list, f, indent=2)


def generate_summary_table(results: list[BenchmarkResult]) -> dict[str, dict[str, Any]]:
    """Generate summary statistics aggregated by model and task.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Benchmark results

    Returns
    -------
    dict
        Summary statistics
    """
    # Group by (task, model)
    groups = {}

    for r in results:
        key = (r.task, r.model_name)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    # Compute summary statistics
    summary = {}

    for (task, model), group_results in groups.items():
        key = f"{task}_{model}"

        primary_scores = [r.primary_metric for r in group_results]
        secondary_scores = [r.secondary_metric for r in group_results]
        latencies = [r.latency_p50 for r in group_results]
        energies = [r.energy_proxy for r in group_results]

        # Check determinism
        hashes = [r.prediction_hash for r in group_results]
        deterministic = len(set(hashes)) == 1

        summary[key] = {
            "task": task,
            "model": model,
            "primary_mean": float(np.mean(primary_scores)),
            "primary_std": float(np.std(primary_scores)),
            "secondary_mean": float(np.mean(secondary_scores)),
            "secondary_std": float(np.std(secondary_scores)),
            "latency_mean": float(np.mean(latencies)),
            "energy_mean": float(np.mean(energies)),
            "stability_margin": compute_stability_margin(primary_scores),
            "deterministic": deterministic,
            "n_runs": len(group_results),
        }

    return summary


def generate_markdown_report(
    results: list[BenchmarkResult],
    output_path: Path,
    title: str = "QuASIM-Own Benchmark Results",
) -> None:
    """Generate Markdown report with benchmark results.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Benchmark results
    output_path : Path
        Output markdown file path
    title : str
        Report title
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = generate_summary_table(results)

    # Generate report
    lines = [
        f"# {title}",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        f"Total runs: {len(results)}",
        "",
        "## Summary by Task and Model",
        "",
    ]

    # Group by task
    tasks = sorted(set(s["task"] for s in summary.values()))

    for task in tasks:
        lines.append(f"### {task}")
        lines.append("")
        lines.append(
            "| Model | Primary Metric | Secondary Metric | Latency (ms) | Stability | Deterministic |"
        )
        lines.append(
            "|-------|---------------|------------------|--------------|-----------|---------------|"
        )

        task_summaries = {k: v for k, v in summary.items() if v["task"] == task}

        for key in sorted(task_summaries.keys()):
            s = task_summaries[key]
            det_str = "✅" if s["deterministic"] else "❌"
            lines.append(
                f"| {s['model']} | {s['primary_mean']:.4f} ± {s['primary_std']:.4f} | "
                f"{s['secondary_mean']:.4f} ± {s['secondary_std']:.4f} | "
                f"{s['latency_mean']:.2f} | {s['stability_margin']:.3f} | {det_str} |"
            )

        lines.append("")

    # Add reliability-per-watt ranking
    lines.append("## Reliability-per-Watt Ranking")
    lines.append("")
    lines.append("Computed as: `(stability × primary_metric) / energy_proxy`")
    lines.append("")

    reliabilities = []
    for key, s in summary.items():
        reliability = (s["stability_margin"] * s["primary_mean"]) / (s["energy_mean"] + 1e-10)
        reliabilities.append((reliability, s["task"], s["model"]))

    reliabilities.sort(reverse=True)

    lines.append("| Rank | Task | Model | Reliability-per-Watt |")
    lines.append("|------|------|-------|---------------------|")

    for i, (rel, task, model) in enumerate(reliabilities[:10], 1):
        lines.append(f"| {i} | {task} | {model} | {rel:.6f} |")

    lines.append("")

    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def generate_ascii_chart(values: list[float], labels: list[str], max_width: int = 50) -> str:
    """Generate simple ASCII bar chart.

    Parameters
    ----------
    values : list[float]
        Values to plot
    labels : list[str]
        Labels for each value
    max_width : int
        Maximum bar width in characters

    Returns
    -------
    str
        ASCII chart
    """
    if not values:
        return ""

    max_val = max(values)
    if max_val == 0:
        max_val = 1.0

    lines = []
    for label, value in zip(labels, values):
        bar_len = int((value / max_val) * max_width)
        bar = "█" * bar_len
        lines.append(f"{label:15s} | {bar} {value:.4f}")

    return "\n".join(lines)
