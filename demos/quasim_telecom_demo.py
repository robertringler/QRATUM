#!/usr/bin/env python3
"""

QuASIM × Telecom & Satellite Constellations Demo

LEO constellation network optimization for throughput and latency.
Deterministic, CPU-only, < 60s runtime per profile.

Usage:
    python demos/quasim_telecom_demo.py --profile configs/vertical_profiles/telecom_constellation.json
"""

import argparse
import base64
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from quasim.common import (calculate_fidelity, evolutionary_optimization,
                           generate_report, load_profile)


def simulate_constellation(alpha: float, profile: dict, seed: int = 42) -> dict:
    """

    Simulate satellite constellation network performance.

    Alpha controls resource allocation strategy between satellites.
    """

    np.random.seed(seed)

    targets = profile["targets"]

    # Simulate network over time
    time = np.linspace(0, 100, 100)  # 100 time steps

    # Throughput model (alpha affects load distribution)
    base_throughput = targets["target_throughput_gbps"]
    throughput = base_throughput * (0.8 + 0.4 * alpha) * (1 + 0.1 * np.sin(time / 10))
    throughput += np.random.normal(0, 2, len(time))

    # Latency model (inverse relationship with alpha)
    base_latency = targets["target_latency_ms"]
    latency = base_latency * (1.3 - 0.5 * alpha) + np.random.normal(0, 1, len(time))

    # Coverage model
    base_coverage = targets["target_coverage_pct"]
    coverage = base_coverage * (0.85 + 0.2 * alpha) * np.ones_like(time)
    coverage += np.random.normal(0, 0.5, len(time))

    # Utilization model
    base_util = targets["target_satellite_util_pct"]
    utilization = base_util * (0.7 + 0.4 * alpha) * np.ones_like(time)
    utilization += np.random.normal(0, 2, len(time))

    return {
        "time": time.tolist(),
        "throughput_gbps": throughput.tolist(),
        "latency_ms": latency.tolist(),
        "coverage_pct": coverage.tolist(),
        "satellite_util_pct": utilization.tolist(),
        "avg_throughput_gbps": float(np.mean(throughput)),
        "avg_latency_ms": float(np.mean(latency)),
        "avg_coverage_pct": float(np.mean(coverage)),
        "avg_satellite_util_pct": float(np.mean(utilization)),
    }


def evaluate_fitness(metrics: dict, profile: dict) -> float:
    """Evaluate network fitness against profile targets."""

    targets = profile["targets"]
    tolerances = profile["tolerances"]
    weights = profile["weights"]

    throughput_error = (
        abs(metrics["avg_throughput_gbps"] - targets["target_throughput_gbps"])
        / tolerances["throughput_tolerance_gbps"]
    ) * weights["throughput"]

    latency_error = (
        abs(metrics["avg_latency_ms"] - targets["target_latency_ms"])
        / tolerances["latency_tolerance_ms"]
    ) * weights["latency"]

    coverage_error = (
        abs(metrics["avg_coverage_pct"] - targets["target_coverage_pct"])
        / tolerances["coverage_tolerance_pct"]
    ) * weights["coverage"]

    util_error = (
        abs(metrics["avg_satellite_util_pct"] - targets["target_satellite_util_pct"])
        / tolerances["util_tolerance_pct"]
    ) * weights["utilization"]

    fitness = np.sqrt(throughput_error**2 + latency_error**2 + coverage_error**2 + util_error**2)
    return float(fitness)


def create_visualization(metrics: dict, profile: dict) -> str:
    """Create network performance visualization."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    time = np.array(metrics["time"])

    # Throughput
    ax1.plot(time, metrics["throughput_gbps"], "b-", linewidth=1.5)
    ax1.axhline(
        y=profile["targets"]["target_throughput_gbps"], color="r", linestyle="--", label="Target"
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Throughput (Gbps)")
    ax1.set_title("Network Throughput")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Latency
    ax2.plot(time, metrics["latency_ms"], "g-", linewidth=1.5)
    ax2.axhline(
        y=profile["targets"]["target_latency_ms"], color="r", linestyle="--", label="Target"
    )
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Network Latency")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Coverage
    ax3.plot(time, metrics["coverage_pct"], "orange", linewidth=1.5)
    ax3.axhline(
        y=profile["targets"]["target_coverage_pct"], color="r", linestyle="--", label="Target"
    )
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Coverage (%)")
    ax3.set_title("Network Coverage")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Utilization
    ax4.plot(time, metrics["satellite_util_pct"], "purple", linewidth=1.5)
    ax4.axhline(
        y=profile["targets"]["target_satellite_util_pct"], color="r", linestyle="--", label="Target"
    )
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Utilization (%)")
    ax4.set_title("Satellite Utilization")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()

    return image_base64


def main():
    parser = argparse.ArgumentParser(description="QuASIM × Telecom Demo")
    parser.add_argument("--profile", type=str, required=True, help="Path to telecom profile JSON")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations")
    parser.add_argument("--pop", type=int, default=20, help="Population size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print(f"Loading profile: {args.profile}")
    profile = load_profile(args.profile)
    print(f"Profile: {profile['name']}")

    print("\nRunning evolutionary optimization...")
    best_alpha, best_metrics, best_fitness, history = evolutionary_optimization(
        simulate_constellation,
        evaluate_fitness,
        profile,
        generations=args.generations,
        population_size=args.pop,
        seed=args.seed,
    )

    print("\nOptimization complete!")
    print(f"  Best alpha: {best_alpha:.6f}")
    print(f"  Best fitness: {best_fitness:.6f}")

    viz_base64 = create_visualization(best_metrics, profile)

    fidelity = calculate_fidelity(
        best_metrics,
        profile["targets"],
        ["avg_throughput_gbps", "avg_latency_ms", "avg_coverage_pct"],
    )

    validation_metrics = {"fidelity": fidelity, "fitness_rmse": best_fitness}

    output_file = f"{Path(args.profile).stem}_demo_report.json"

    generate_report(
        profile,
        best_alpha,
        best_fitness,
        best_metrics,
        history,
        {"generations": args.generations, "population_size": args.pop, "seed": args.seed},
        viz_base64,
        validation_metrics,
        profile.get("compliance_tags", []),
        output_file,
    )

    print(f"\nReport saved: {output_file}")
    print(f"  Fidelity: {fidelity:.4f}")
    print("\nDemo complete! ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
