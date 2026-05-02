#!/usr/bin/env python3
"""

QuASIM × Manufacturing IIoT Demo

Smart factory production line optimization.
Deterministic, CPU-only, < 60s runtime per profile.

Usage:
    python demos/quasim_manufacturing_demo.py --profile configs/vertical_profiles/manufacturing_iiot.json
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


def simulate_production(alpha: float, profile: dict, seed: int = 42) -> dict:
    """Simulate manufacturing production line."""

    np.random.seed(seed)
    targets = profile["targets"]
    time = np.linspace(0, 100, 100)

    base_oee = targets["target_oee_pct"]
    oee = base_oee * (0.8 + 0.25 * alpha) * np.ones_like(time)
    oee += np.random.normal(0, 1.5, len(time))

    base_throughput = targets["target_throughput_units_per_hr"]
    throughput = base_throughput * (0.75 + 0.35 * alpha) * np.ones_like(time)
    throughput += np.random.normal(0, 30, len(time))

    base_defect = targets["target_defect_rate_ppm"]
    defect_rate = base_defect * (1.5 - 0.7 * alpha) * np.ones_like(time)
    defect_rate += np.random.normal(0, 5, len(time))
    defect_rate = np.maximum(defect_rate, 10)

    base_downtime = targets["target_downtime_pct"]
    downtime = base_downtime * (1.8 - alpha) * np.ones_like(time)
    downtime += np.random.normal(0, 0.3, len(time))

    return {
        "time": time.tolist(),
        "oee_pct": oee.tolist(),
        "throughput": throughput.tolist(),
        "defect_rate_ppm": defect_rate.tolist(),
        "downtime_pct": downtime.tolist(),
        "avg_oee": float(np.mean(oee)),
        "avg_throughput": float(np.mean(throughput)),
        "avg_defect_rate": float(np.mean(defect_rate)),
        "avg_downtime": float(np.mean(downtime)),
    }


def evaluate_fitness(metrics: dict, profile: dict) -> float:
    """Evaluate manufacturing fitness."""

    targets = profile["targets"]
    tolerances = profile["tolerances"]
    weights = profile["weights"]

    oee_error = (
        abs(metrics["avg_oee"] - targets["target_oee_pct"]) / tolerances["oee_tolerance_pct"]
    ) * weights["oee"]
    throughput_error = (
        abs(metrics["avg_throughput"] - targets["target_throughput_units_per_hr"])
        / tolerances["throughput_tolerance"]
    ) * weights["throughput"]
    defect_error = (
        abs(metrics["avg_defect_rate"] - targets["target_defect_rate_ppm"])
        / tolerances["defect_tolerance_ppm"]
    ) * weights["defect_rate"]
    downtime_error = (
        abs(metrics["avg_downtime"] - targets["target_downtime_pct"])
        / tolerances["downtime_tolerance_pct"]
    ) * weights["downtime"]

    return float(np.sqrt(oee_error**2 + throughput_error**2 + defect_error**2 + downtime_error**2))


def create_visualization(metrics: dict, profile: dict) -> str:
    """Create manufacturing visualization."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    time = np.array(metrics["time"])

    ax1.plot(time, metrics["oee_pct"], "b-", linewidth=1.5)
    ax1.axhline(y=profile["targets"]["target_oee_pct"], color="r", linestyle="--")
    ax1.set_ylabel("OEE (%)")
    ax1.set_title("Overall Equipment Effectiveness")
    ax1.grid(True, alpha=0.3)

    ax2.plot(time, metrics["throughput"], "g-", linewidth=1.5)
    ax2.axhline(y=profile["targets"]["target_throughput_units_per_hr"], color="r", linestyle="--")
    ax2.set_ylabel("Units/hr")
    ax2.set_title("Production Throughput")
    ax2.grid(True, alpha=0.3)

    ax3.plot(time, metrics["defect_rate_ppm"], "orange", linewidth=1.5)
    ax3.axhline(y=profile["targets"]["target_defect_rate_ppm"], color="r", linestyle="--")
    ax3.set_ylabel("Defects (PPM)")
    ax3.set_title("Defect Rate")
    ax3.grid(True, alpha=0.3)

    ax4.plot(time, metrics["downtime_pct"], "purple", linewidth=1.5)
    ax4.axhline(y=profile["targets"]["target_downtime_pct"], color="r", linestyle="--")
    ax4.set_ylabel("Downtime (%)")
    ax4.set_title("Production Downtime")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return image_base64


def main():
    parser = argparse.ArgumentParser(description="QuASIM × Manufacturing Demo")
    parser.add_argument("--profile", type=str, required=True)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    print(f"Profile: {profile['name']}")

    print("\nRunning evolutionary optimization...")
    best_alpha, best_metrics, best_fitness, history = evolutionary_optimization(
        simulate_production, evaluate_fitness, profile, args.generations, args.pop, args.seed
    )

    print(f"\nOptimization complete! Best alpha: {best_alpha:.6f}")

    viz_base64 = create_visualization(best_metrics, profile)
    fidelity = calculate_fidelity(
        best_metrics, profile["targets"], ["avg_oee", "avg_throughput", "avg_defect_rate"]
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
