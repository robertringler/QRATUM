#!/usr/bin/env python3
"""

QuASIM × Healthcare Genomics Demo

Genomic pipeline optimization for WGS throughput and accuracy.
Deterministic, CPU-only, < 60s runtime per profile.

Usage:
    python demos/quasim_healthcare_demo.py --profile configs/vertical_profiles/healthcare_genomics.json
"""

import argparse
import base64
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from quasim.common import (
    calculate_fidelity,
    evolutionary_optimization,
    generate_report,
    load_profile,
)


def simulate_genomics_pipeline(alpha: float, profile: dict, seed: int = 42) -> dict:
    """

    Simulate genomic sequencing pipeline performance.

    Alpha controls resource allocation (compute vs accuracy trade-off).
    """

    np.random.seed(seed)

    targets = profile["targets"]
    time = np.linspace(0, 100, 100)

    # Throughput model (alpha affects parallelization)
    base_throughput = targets["target_throughput_samples_per_day"]
    throughput = base_throughput * (0.7 + 0.5 * alpha) * np.ones_like(time)
    throughput += np.random.normal(0, 2, len(time))

    # Accuracy model (higher alpha = more computational rigor)
    base_accuracy = targets["target_accuracy_pct"]
    accuracy = base_accuracy * (0.98 + 0.02 * alpha) * np.ones_like(time)
    accuracy += np.random.normal(0, 0.05, len(time))

    # Sensitivity model
    base_sensitivity = targets["target_sensitivity_pct"]
    sensitivity = base_sensitivity * (0.96 + 0.04 * alpha) * np.ones_like(time)
    sensitivity += np.random.normal(0, 0.3, len(time))

    # Cost model (higher throughput = higher cost)
    base_cost = targets["target_cost_per_sample_usd"]
    cost = base_cost * (1.2 - 0.3 * alpha) * np.ones_like(time)
    cost += np.random.normal(0, 5, len(time))

    return {
        "time": time.tolist(),
        "throughput_samples_per_day": throughput.tolist(),
        "accuracy_pct": accuracy.tolist(),
        "sensitivity_pct": sensitivity.tolist(),
        "cost_per_sample_usd": cost.tolist(),
        "avg_throughput": float(np.mean(throughput)),
        "avg_accuracy": float(np.mean(accuracy)),
        "avg_sensitivity": float(np.mean(sensitivity)),
        "avg_cost": float(np.mean(cost)),
    }


def evaluate_fitness(metrics: dict, profile: dict) -> float:
    """Evaluate pipeline fitness."""

    targets = profile["targets"]
    tolerances = profile["tolerances"]
    weights = profile["weights"]

    throughput_error = (
        abs(metrics["avg_throughput"] - targets["target_throughput_samples_per_day"])
        / tolerances["throughput_tolerance"]
    ) * weights["throughput"]

    accuracy_error = (
        abs(metrics["avg_accuracy"] - targets["target_accuracy_pct"])
        / tolerances["accuracy_tolerance_pct"]
    ) * weights["accuracy"]

    sensitivity_error = (
        abs(metrics["avg_sensitivity"] - targets["target_sensitivity_pct"])
        / tolerances["sensitivity_tolerance_pct"]
    ) * weights["sensitivity"]

    cost_error = (
        abs(metrics["avg_cost"] - targets["target_cost_per_sample_usd"])
        / tolerances["cost_tolerance_usd"]
    ) * weights["cost"]

    fitness = np.sqrt(
        throughput_error**2 + accuracy_error**2 + sensitivity_error**2 + cost_error**2
    )
    return float(fitness)


def create_visualization(metrics: dict, profile: dict) -> str:
    """Create genomics pipeline visualization."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    time = np.array(metrics["time"])

    ax1.plot(time, metrics["throughput_samples_per_day"], "b-", linewidth=1.5)
    ax1.axhline(
        y=profile["targets"]["target_throughput_samples_per_day"],
        color="r",
        linestyle="--",
        label="Target",
    )
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Samples/Day")
    ax1.set_title("Pipeline Throughput")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(time, metrics["accuracy_pct"], "g-", linewidth=1.5)
    ax2.axhline(
        y=profile["targets"]["target_accuracy_pct"], color="r", linestyle="--", label="Target"
    )
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Variant Calling Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(time, metrics["sensitivity_pct"], "orange", linewidth=1.5)
    ax3.axhline(
        y=profile["targets"]["target_sensitivity_pct"], color="r", linestyle="--", label="Target"
    )
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Sensitivity (%)")
    ax3.set_title("Detection Sensitivity")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.plot(time, metrics["cost_per_sample_usd"], "purple", linewidth=1.5)
    ax4.axhline(
        y=profile["targets"]["target_cost_per_sample_usd"],
        color="r",
        linestyle="--",
        label="Target",
    )
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Cost (USD)")
    ax4.set_title("Cost per Sample")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()

    return image_base64


def main():
    parser = argparse.ArgumentParser(description="QuASIM × Healthcare Genomics Demo")
    parser.add_argument("--profile", type=str, required=True)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    print(f"Profile: {profile['name']}")

    print("\nRunning evolutionary optimization...")
    best_alpha, best_metrics, best_fitness, history = evolutionary_optimization(
        simulate_genomics_pipeline,
        evaluate_fitness,
        profile,
        generations=args.generations,
        population_size=args.pop,
        seed=args.seed,
    )

    print("\nOptimization complete!")
    print(f"  Best alpha: {best_alpha:.6f}")

    viz_base64 = create_visualization(best_metrics, profile)
    fidelity = calculate_fidelity(
        best_metrics, profile["targets"], ["avg_throughput", "avg_accuracy", "avg_sensitivity"]
    )

    output_file = f"{Path(args.profile).stem}_demo_report.json"
    generate_report(
        profile,
        best_alpha,
        best_fitness,
        best_metrics,
        history,
        {"generations": args.generations, "population_size": args.pop, "seed": args.seed},
        viz_base64,
        {"fidelity": fidelity, "fitness_rmse": best_fitness},
        profile.get("compliance_tags", []),
        output_file,
    )

    print(f"\nReport saved: {output_file}")
    print(f"  Fidelity: {fidelity:.4f}")
    print("\nDemo complete! ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
