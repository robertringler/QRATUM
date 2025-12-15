#!/usr/bin/env python3
"""
QuASIM × Transportation & Logistics Demo

Multi-modal freight network optimization.
Deterministic, CPU-only, < 60s runtime per profile.

Usage:
    python demos/quasim_transportation_demo.py --profile configs/vertical_profiles/transportation_logistics.json
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


def simulate_logistics(alpha: float, profile: dict, seed: int = 42) -> dict:
    """Simulate logistics network performance."""
    np.random.seed(seed)
    targets = profile["targets"]
    time = np.linspace(0, 100, 100)

    base_cost = targets["target_cost_per_mile_usd"]
    cost = base_cost * (1.3 - 0.4 * alpha) * np.ones_like(time)
    cost += np.random.normal(0, 0.05, len(time))

    base_time = targets["target_delivery_time_hrs"]
    delivery_time = base_time * (1.2 - 0.3 * alpha) * np.ones_like(time)
    delivery_time += np.random.normal(0, 2, len(time))

    base_util = targets["target_utilization_pct"]
    utilization = base_util * (0.8 + 0.3 * alpha) * np.ones_like(time)
    utilization += np.random.normal(0, 2, len(time))

    base_on_time = targets["target_on_time_pct"]
    on_time = base_on_time * (0.88 + 0.14 * alpha) * np.ones_like(time)
    on_time += np.random.normal(0, 1, len(time))

    return {
        "time": time.tolist(),
        "cost_per_mile": cost.tolist(),
        "delivery_time": delivery_time.tolist(),
        "utilization": utilization.tolist(),
        "on_time_pct": on_time.tolist(),
        "avg_cost": float(np.mean(cost)),
        "avg_delivery_time": float(np.mean(delivery_time)),
        "avg_utilization": float(np.mean(utilization)),
        "avg_on_time": float(np.mean(on_time)),
    }


def evaluate_fitness(metrics: dict, profile: dict) -> float:
    """Evaluate logistics fitness."""
    targets = profile["targets"]
    tolerances = profile["tolerances"]
    weights = profile["weights"]

    cost_error = (
        abs(metrics["avg_cost"] - targets["target_cost_per_mile_usd"])
        / tolerances["cost_tolerance_usd"]
    ) * weights["cost"]
    time_error = (
        abs(metrics["avg_delivery_time"] - targets["target_delivery_time_hrs"])
        / tolerances["time_tolerance_hrs"]
    ) * weights["delivery_time"]
    util_error = (
        abs(metrics["avg_utilization"] - targets["target_utilization_pct"])
        / tolerances["util_tolerance_pct"]
    ) * weights["utilization"]
    on_time_error = (
        abs(metrics["avg_on_time"] - targets["target_on_time_pct"])
        / tolerances["on_time_tolerance_pct"]
    ) * weights["on_time"]

    return float(np.sqrt(cost_error**2 + time_error**2 + util_error**2 + on_time_error**2))


def create_visualization(metrics: dict, profile: dict) -> str:
    """Create logistics visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    time = np.array(metrics["time"])

    ax1.plot(time, metrics["cost_per_mile"], "b-", linewidth=1.5)
    ax1.axhline(y=profile["targets"]["target_cost_per_mile_usd"], color="r", linestyle="--")
    ax1.set_ylabel("Cost (USD/mile)")
    ax1.set_title("Cost per Mile")
    ax1.grid(True, alpha=0.3)

    ax2.plot(time, metrics["delivery_time"], "g-", linewidth=1.5)
    ax2.axhline(y=profile["targets"]["target_delivery_time_hrs"], color="r", linestyle="--")
    ax2.set_ylabel("Time (hours)")
    ax2.set_title("Delivery Time")
    ax2.grid(True, alpha=0.3)

    ax3.plot(time, metrics["utilization"], "orange", linewidth=1.5)
    ax3.axhline(y=profile["targets"]["target_utilization_pct"], color="r", linestyle="--")
    ax3.set_ylabel("Utilization (%)")
    ax3.set_title("Fleet Utilization")
    ax3.grid(True, alpha=0.3)

    ax4.plot(time, metrics["on_time_pct"], "purple", linewidth=1.5)
    ax4.axhline(y=profile["targets"]["target_on_time_pct"], color="r", linestyle="--")
    ax4.set_ylabel("On-Time (%)")
    ax4.set_title("On-Time Delivery Rate")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return image_base64


def main():
    parser = argparse.ArgumentParser(description="QuASIM × Transportation Demo")
    parser.add_argument("--profile", type=str, required=True)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    print(f"Profile: {profile['name']}")

    print("\nRunning evolutionary optimization...")
    best_alpha, best_metrics, best_fitness, history = evolutionary_optimization(
        simulate_logistics, evaluate_fitness, profile, args.generations, args.pop, args.seed
    )

    print(f"\nOptimization complete! Best alpha: {best_alpha:.6f}")

    viz_base64 = create_visualization(best_metrics, profile)
    fidelity = calculate_fidelity(
        best_metrics, profile["targets"], ["avg_cost", "avg_delivery_time", "avg_utilization"]
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

    print(f"\nReport saved: {output_file}\n  Fidelity: {fidelity:.4f}\n\nDemo complete! ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
