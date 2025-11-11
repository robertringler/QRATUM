"""TERC observables extraction for QuASIM-Own runs."""

from pathlib import Path
from typing import Any

import numpy as np

from quasim.ownai.eval.benchmark import BenchmarkResult
from quasim.ownai.train.metrics import compute_stability_margin


def collect_terc_observables(results: list[BenchmarkResult]) -> dict[str, Any]:
    """Collect TERC observables from benchmark results.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Benchmark results

    Returns
    -------
    dict[str, Any]
        TERC observables including:
        - stability_margin: 1 - CV across repeats
        - qgh_consensus_status: All prediction hashes equal?
        - emergent_complexity: Latent dispersion measure
        - goal_progress: Task completion metric
    """
    if not results:
        return {
            "stability_margin": 0.0,
            "qgh_consensus_status": False,
            "emergent_complexity": 0.0,
            "goal_progress": 0.0,
        }

    # Group by model/task
    groups = {}
    for r in results:
        key = (r.model_name, r.task)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    # Compute aggregated observables
    all_stabilities = []
    all_consensus = []
    all_complexities = []
    all_progress = []

    for group in groups.values():
        # Stability margin
        primary_scores = [r.primary_metric for r in group]
        stability = compute_stability_margin(primary_scores)
        all_stabilities.append(stability)

        # QGH consensus: all hashes equal?
        hashes = [r.prediction_hash for r in group]
        consensus = len(set(hashes)) == 1
        all_consensus.append(consensus)

        # Emergent complexity: variance in latencies (proxy)
        latencies = [r.latency_p50 for r in group]
        complexity = float(np.std(latencies) / (np.mean(latencies) + 1e-10))
        all_complexities.append(complexity)

        # Goal progress: mean primary metric
        progress = float(np.mean(primary_scores))
        all_progress.append(progress)

    return {
        "stability_margin": float(np.mean(all_stabilities)),
        "qgh_consensus_status": all(all_consensus),
        "emergent_complexity": float(np.mean(all_complexities)),
        "goal_progress": float(np.mean(all_progress)),
    }


def save_terc_observables(observables: dict[str, Any], output_path: Path) -> None:
    """Save TERC observables to JSON file.

    Parameters
    ----------
    observables : dict
        TERC observables
    output_path : Path
        Output file path
    """
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(observables, f, indent=2)
