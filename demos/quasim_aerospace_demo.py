#!/usr/bin/env python3
"""

QuASIM × Aerospace & Defense Demo

Profile-aware trajectory optimization for launch vehicle ascent profiles.
Deterministic, CPU-only, < 60s runtime per profile.

Usage:
    python demos/quasim_aerospace_demo.py --profile configs/vertical_profiles/aerospace_f9.json
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from quasim.common import (calculate_fidelity, evolutionary_optimization,
                           generate_report, load_profile)


def simulate_trajectory(alpha: float, profile: dict, seed: int = 42) -> dict:
    """

    Simulate ascent trajectory with given thrust profile alpha.

    This is a simplified surrogate model for demonstration purposes.
    Real QuASIM would use cuQuantum tensor network simulation.
    """

    np.random.seed(seed)

    targets = profile["targets"]
    t_meco = targets["meco_time_s"]

    # Simplified physics model
    time = np.linspace(0, t_meco, 100)
    thrust_profile = alpha + (1 - alpha) * (time / t_meco) ** 2

    dt = time[1] - time[0]
    velocity = np.zeros_like(time)
    altitude = np.zeros_like(time)

    g0 = 9.81e-3  # km/s²
    drag_coeff = 0.001 * (1 - alpha * 0.3)

    for i in range(1, len(time)):
        accel = thrust_profile[i] * 0.03 - drag_coeff * velocity[i - 1] ** 2 - g0
        velocity[i] = velocity[i - 1] + accel * dt
        altitude[i] = altitude[i - 1] + velocity[i - 1] * dt

    meco_idx = -1
    meco_altitude = altitude[meco_idx]
    meco_velocity = velocity[meco_idx]
    peak_altitude = np.max(altitude)

    return {
        "time": time.tolist(),
        "altitude": altitude.tolist(),
        "velocity": velocity.tolist(),
        "meco_time_s": float(t_meco),
        "meco_altitude_km": float(meco_altitude),
        "meco_velocity_kms": float(meco_velocity),
        "peak_altitude_km": float(peak_altitude),
    }


def evaluate_fitness(metrics: dict, profile: dict) -> float:
    """Evaluate trajectory fitness against profile targets."""

    targets = profile["targets"]
    tolerances = profile["tolerances"]
    weights = profile["weights"]

    alt_error = (
        abs(metrics["peak_altitude_km"] - targets["peak_altitude_km"])
        / tolerances["altitude_tolerance_km"]
    ) * weights["altitude"]

    vel_error = (
        abs(metrics["meco_velocity_kms"] - targets["meco_velocity_kms"])
        / tolerances["velocity_tolerance_kms"]
    ) * weights["velocity"]

    time_error = (
        abs(metrics["meco_time_s"] - targets["meco_time_s"]) / tolerances["time_tolerance_s"]
    ) * weights["time"]

    fitness = np.sqrt(alt_error**2 + vel_error**2 + time_error**2)
    return float(fitness)


def create_visualization(metrics: dict, profile: dict) -> str:
    """Create trajectory visualization and return as base64-encoded PNG."""

    import base64
    import io

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    time = np.array(metrics["time"])
    altitude = np.array(metrics["altitude"])
    velocity = np.array(metrics["velocity"])

    ax1.plot(time, altitude, "b-", linewidth=2, label="Trajectory")
    ax1.axhline(
        y=profile["targets"]["peak_altitude_km"],
        color="r",
        linestyle="--",
        label="Target Altitude",
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Altitude (km)")
    ax1.set_title(f"{profile['name']} - Altitude Profile")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(time, velocity, "g-", linewidth=2, label="Trajectory")
    ax2.axhline(
        y=profile["targets"]["meco_velocity_kms"],
        color="r",
        linestyle="--",
        label="Target MECO Velocity",
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Velocity (km/s)")
    ax2.set_title(f"{profile['name']} - Velocity Profile")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()

    return image_base64


def main():
    parser = argparse.ArgumentParser(description="QuASIM × Aerospace & Defense Demo")
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help="Path to aerospace profile JSON",
    )
    parser.add_argument("--generations", type=int, default=50, help="Number of generations")
    parser.add_argument("--pop", type=int, default=20, help="Population size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print(f"Loading profile: {args.profile}")
    profile = load_profile(args.profile)
    print(f"Profile: {profile['name']}")
    print(f"Description: {profile['description']}")

    print("\nRunning evolutionary optimization...")
    best_alpha, best_metrics, best_fitness, history = evolutionary_optimization(
        simulate_trajectory,
        evaluate_fitness,
        profile,
        generations=args.generations,
        population_size=args.pop,
        seed=args.seed,
    )

    print("\nOptimization complete!")
    print(f"  Best alpha: {best_alpha:.6f}")
    print(f"  Best fitness (RMSE): {best_fitness:.6f}")

    viz_base64 = create_visualization(best_metrics, profile)

    targets = profile["targets"]
    fidelity = calculate_fidelity(
        best_metrics,
        targets,
        ["peak_altitude_km", "meco_velocity_kms"],
    )

    validation_metrics = {
        "fidelity": fidelity,
        "fitness_rmse": best_fitness,
    }

    profile_name = Path(args.profile).stem
    output_file = f"{profile_name}_demo_report.json"

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
