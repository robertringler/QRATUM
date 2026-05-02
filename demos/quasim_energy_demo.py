#!/usr/bin/env python3
"""

QuASIM × Energy Grid Optimization Demo

Smart grid load balancing with renewable integration.
Deterministic, CPU-only, < 60s runtime per profile.

Usage:
    python demos/quasim_energy_demo.py --profile configs/vertical_profiles/energy_grid.json
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


def simulate_grid(alpha: float, profile: dict, seed: int = 42) -> dict:
    """Simulate smart grid performance with renewable integration."""

    np.random.seed(seed)

    targets = profile["targets"]
    time = np.linspace(0, 24, 100)  # 24 hours

    # Load balance (alpha affects distribution strategy)
    base_balance = targets["target_load_balance_pct"]
    load_balance = base_balance * (0.85 + 0.2 * alpha) * (1 + 0.05 * np.sin(time * np.pi / 12))
    load_balance += np.random.normal(0, 1, len(time))

    # Renewable penetration
    base_renewable = targets["target_renewable_pct"]
    renewable = base_renewable * (0.8 + 0.3 * alpha) * (1 + 0.15 * np.sin(time * np.pi / 6))
    renewable += np.random.normal(0, 2, len(time))

    # Frequency stability
    base_freq = targets["target_frequency_hz"]
    frequency = base_freq * (1 + 0.001 * (alpha - 0.5)) * np.ones_like(time)
    frequency += np.random.normal(0, 0.02, len(time))

    # Voltage stability
    base_voltage = targets["target_voltage_stability"]
    voltage = base_voltage * (0.97 + 0.04 * alpha) * np.ones_like(time)
    voltage += np.random.normal(0, 0.01, len(time))

    return {
        "time": time.tolist(),
        "load_balance_pct": load_balance.tolist(),
        "renewable_pct": renewable.tolist(),
        "frequency_hz": frequency.tolist(),
        "voltage_stability": voltage.tolist(),
        "avg_load_balance": float(np.mean(load_balance)),
        "avg_renewable": float(np.mean(renewable)),
        "avg_frequency": float(np.mean(frequency)),
        "avg_voltage": float(np.mean(voltage)),
    }


def evaluate_fitness(metrics: dict, profile: dict) -> float:
    """Evaluate grid fitness."""

    targets = profile["targets"]
    tolerances = profile["tolerances"]
    weights = profile["weights"]

    balance_error = (
        abs(metrics["avg_load_balance"] - targets["target_load_balance_pct"])
        / tolerances["balance_tolerance_pct"]
    ) * weights["load_balance"]

    renewable_error = (
        abs(metrics["avg_renewable"] - targets["target_renewable_pct"])
        / tolerances["renewable_tolerance_pct"]
    ) * weights["renewable"]

    freq_error = (
        abs(metrics["avg_frequency"] - targets["target_frequency_hz"])
        / tolerances["frequency_tolerance_hz"]
    ) * weights["frequency"]

    voltage_error = (
        abs(metrics["avg_voltage"] - targets["target_voltage_stability"])
        / tolerances["voltage_tolerance"]
    ) * weights["voltage"]

    return float(np.sqrt(balance_error**2 + renewable_error**2 + freq_error**2 + voltage_error**2))


def create_visualization(metrics: dict, profile: dict) -> str:
    """Create grid visualization."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    time = np.array(metrics["time"])

    ax1.plot(time, metrics["load_balance_pct"], "b-", linewidth=1.5)
    ax1.axhline(y=profile["targets"]["target_load_balance_pct"], color="r", linestyle="--")
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Load Balance (%)")
    ax1.set_title("Grid Load Balance")
    ax1.grid(True, alpha=0.3)

    ax2.plot(time, metrics["renewable_pct"], "g-", linewidth=1.5)
    ax2.axhline(y=profile["targets"]["target_renewable_pct"], color="r", linestyle="--")
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Renewable (%)")
    ax2.set_title("Renewable Energy Penetration")
    ax2.grid(True, alpha=0.3)

    ax3.plot(time, metrics["frequency_hz"], "orange", linewidth=1.5)
    ax3.axhline(y=profile["targets"]["target_frequency_hz"], color="r", linestyle="--")
    ax3.set_xlabel("Time (hours)")
    ax3.set_ylabel("Frequency (Hz)")
    ax3.set_title("Grid Frequency")
    ax3.grid(True, alpha=0.3)

    ax4.plot(time, metrics["voltage_stability"], "purple", linewidth=1.5)
    ax4.axhline(y=profile["targets"]["target_voltage_stability"], color="r", linestyle="--")
    ax4.set_xlabel("Time (hours)")
    ax4.set_ylabel("Voltage Stability")
    ax4.set_title("Voltage Stability Index")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()

    return image_base64


def main():
    parser = argparse.ArgumentParser(description="QuASIM × Energy Demo")
    parser.add_argument("--profile", type=str, required=True)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    print(f"Profile: {profile['name']}")

    print("\nRunning evolutionary optimization...")
    best_alpha, best_metrics, best_fitness, history = evolutionary_optimization(
        simulate_grid, evaluate_fitness, profile, args.generations, args.pop, args.seed
    )

    print(f"\nOptimization complete! Best alpha: {best_alpha:.6f}")

    viz_base64 = create_visualization(best_metrics, profile)
    fidelity = calculate_fidelity(
        best_metrics, profile["targets"], ["avg_load_balance", "avg_renewable", "avg_frequency"]
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
