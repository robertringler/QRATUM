#!/usr/bin/env python3
"""

QuASIM × Agritech Precision Agriculture Demo

Crop yield optimization with resource efficiency.
Deterministic, CPU-only, < 60s runtime per profile.

Usage:
    python demos/quasim_agritech_demo.py --profile configs/vertical_profiles/agritech_optimization.json
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


def simulate_agriculture(alpha: float, profile: dict, seed: int = 42) -> dict:
    """Simulate precision agriculture operations."""

    np.random.seed(seed)
    targets = profile["targets"]
    time = np.linspace(0, 100, 100)  # Growing season

    base_yield = targets["target_yield_bushels_per_acre"]
    crop_yield = base_yield * (0.8 + 0.3 * alpha) * (1 + 0.1 * np.sin(time * np.pi / 50))
    crop_yield += np.random.normal(0, 5, len(time))

    base_water = targets["target_water_efficiency_gal_per_bushel"]
    water_eff = base_water * (1.4 - 0.5 * alpha) * np.ones_like(time)
    water_eff += np.random.normal(0, 1, len(time))

    base_fert = targets["target_fertilizer_efficiency_lbs_per_acre"]
    fertilizer = base_fert * (1.3 - 0.4 * alpha) * np.ones_like(time)
    fertilizer += np.random.normal(0, 8, len(time))

    base_margin = targets["target_profit_margin_pct"]
    profit_margin = base_margin * (0.7 + 0.5 * alpha) * np.ones_like(time)
    profit_margin += np.random.normal(0, 2, len(time))

    return {
        "time": time.tolist(),
        "yield_bushels_per_acre": crop_yield.tolist(),
        "water_efficiency": water_eff.tolist(),
        "fertilizer_lbs_per_acre": fertilizer.tolist(),
        "profit_margin_pct": profit_margin.tolist(),
        "avg_yield": float(np.mean(crop_yield)),
        "avg_water_eff": float(np.mean(water_eff)),
        "avg_fertilizer": float(np.mean(fertilizer)),
        "avg_profit_margin": float(np.mean(profit_margin)),
    }


def evaluate_fitness(metrics: dict, profile: dict) -> float:
    """Evaluate agriculture fitness."""

    targets = profile["targets"]
    tolerances = profile["tolerances"]
    weights = profile["weights"]

    yield_error = (
        abs(metrics["avg_yield"] - targets["target_yield_bushels_per_acre"])
        / tolerances["yield_tolerance"]
    ) * weights["yield"]
    water_error = (
        abs(metrics["avg_water_eff"] - targets["target_water_efficiency_gal_per_bushel"])
        / tolerances["water_tolerance"]
    ) * weights["water_efficiency"]
    fert_error = (
        abs(metrics["avg_fertilizer"] - targets["target_fertilizer_efficiency_lbs_per_acre"])
        / tolerances["fertilizer_tolerance"]
    ) * weights["fertilizer_efficiency"]
    margin_error = (
        abs(metrics["avg_profit_margin"] - targets["target_profit_margin_pct"])
        / tolerances["margin_tolerance_pct"]
    ) * weights["profit_margin"]

    return float(np.sqrt(yield_error**2 + water_error**2 + fert_error**2 + margin_error**2))


def create_visualization(metrics: dict, profile: dict) -> str:
    """Create agriculture visualization."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    time = np.array(metrics["time"])

    ax1.plot(time, metrics["yield_bushels_per_acre"], "b-", linewidth=1.5)
    ax1.axhline(y=profile["targets"]["target_yield_bushels_per_acre"], color="r", linestyle="--")
    ax1.set_ylabel("Bushels/acre")
    ax1.set_title("Crop Yield")
    ax1.grid(True, alpha=0.3)

    ax2.plot(time, metrics["water_efficiency"], "g-", linewidth=1.5)
    ax2.axhline(
        y=profile["targets"]["target_water_efficiency_gal_per_bushel"], color="r", linestyle="--"
    )
    ax2.set_ylabel("Gal/bushel")
    ax2.set_title("Water Efficiency")
    ax2.grid(True, alpha=0.3)

    ax3.plot(time, metrics["fertilizer_lbs_per_acre"], "orange", linewidth=1.5)
    ax3.axhline(
        y=profile["targets"]["target_fertilizer_efficiency_lbs_per_acre"], color="r", linestyle="--"
    )
    ax3.set_ylabel("Lbs/acre")
    ax3.set_title("Fertilizer Usage")
    ax3.grid(True, alpha=0.3)

    ax4.plot(time, metrics["profit_margin_pct"], "purple", linewidth=1.5)
    ax4.axhline(y=profile["targets"]["target_profit_margin_pct"], color="r", linestyle="--")
    ax4.set_ylabel("Margin (%)")
    ax4.set_title("Profit Margin")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return image_base64


def main():
    parser = argparse.ArgumentParser(description="QuASIM × Agritech Demo")
    parser.add_argument("--profile", type=str, required=True)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    print(f"Profile: {profile['name']}")

    print("\nRunning evolutionary optimization...")
    best_alpha, best_metrics, best_fitness, history = evolutionary_optimization(
        simulate_agriculture, evaluate_fitness, profile, args.generations, args.pop, args.seed
    )

    print(f"\nOptimization complete! Best alpha: {best_alpha:.6f}")

    viz_base64 = create_visualization(best_metrics, profile)
    fidelity = calculate_fidelity(
        best_metrics, profile["targets"], ["avg_yield", "avg_water_eff", "avg_fertilizer"]
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
