#!/usr/bin/env python3
"""

QuASIM × SpaceX/NASA Pilot Track Demo

Profile-aware evolutionary optimization for MECO and hot-staging trajectories.
Deterministic, CPU-only, < 30s runtime per profile.

Usage:
    python quasim_spacex_demo.py --profile configs/meco_profiles/spacex_f9_stage1.json
    python quasim_spacex_demo.py --profile configs/meco_profiles/starship_hotstaging.json
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend


def load_profile(profile_path: str) -> dict:
    """Load MECO/hot-staging profile from JSON."""

    with open(profile_path) as f:
        return json.load(f)


def simulate_trajectory(alpha: float, profile: dict, seed: int = 42) -> dict:
    """

    Simulate ascent trajectory with given thrust profile alpha.

    This is a simplified surrogate model for demonstration purposes.
    Real QuASIM would use cuQuantum tensor network simulation.

    Args:
        alpha: Thrust shaping parameter (0.0-1.0)
        profile: Mission profile with targets and tolerances
        seed: Random seed for deterministic simulation

    Returns:
        Dictionary with trajectory metrics
    """

    np.random.seed(seed)

    # Extract targets from profile
    targets = profile["targets"]
    t_meco = targets["meco_time_s"]
    targets["peak_altitude_km"]
    targets["meco_velocity_kms"]

    # Simplified physics model (surrogate for actual simulation)
    # Alpha shapes the thrust profile over time
    time = np.linspace(0, t_meco, 100)
    thrust_profile = alpha + (1 - alpha) * (time / t_meco) ** 2

    # Integration of simplified equations of motion
    # v = ∫(thrust - drag - gravity) dt
    # h = ∫v dt
    dt = time[1] - time[0]
    velocity = np.zeros_like(time)
    altitude = np.zeros_like(time)

    g0 = 9.81e-3  # km/s²
    drag_coeff = 0.001 * (1 - alpha * 0.3)  # Alpha affects drag through vehicle orientation

    for i in range(1, len(time)):
        # Simplified acceleration (thrust - drag - gravity)
        accel = thrust_profile[i] * 0.03 - drag_coeff * velocity[i - 1] ** 2 - g0
        velocity[i] = velocity[i - 1] + accel * dt
        altitude[i] = altitude[i - 1] + velocity[i - 1] * dt

    # MECO conditions (at specified time)
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
    """

    Evaluate trajectory fitness against profile targets.

    Lower is better (RMSE-style objective).
    """

    targets = profile["targets"]
    tolerances = profile["tolerances"]
    weights = profile["weights"]

    # Calculate weighted errors
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

    # Combined fitness (RMSE)
    fitness = np.sqrt(alt_error**2 + vel_error**2 + time_error**2)
    return float(fitness)


def evolutionary_optimization(
    profile: dict, generations: int = 50, population_size: int = 20, seed: int = 42
) -> tuple:
    """

    Simple evolutionary algorithm to optimize alpha parameter.

    Returns:
        (best_alpha, best_metrics, best_fitness, optimization_history)
    """

    np.random.seed(seed)

    alpha_min, alpha_max = profile["alpha_bounds"]

    # Initialize population
    population = np.random.uniform(alpha_min, alpha_max, population_size)

    history = []

    for gen in range(generations):
        # Evaluate fitness for all individuals
        fitness_scores = []
        metrics_list = []

        for alpha in population:
            metrics = simulate_trajectory(alpha, profile, seed=seed + gen)
            fitness = evaluate_fitness(metrics, profile)
            fitness_scores.append(fitness)
            metrics_list.append(metrics)

        # Track best
        best_idx = np.argmin(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        best_alpha = population[best_idx]
        metrics_list[best_idx]

        history.append(
            {
                "generation": gen,
                "best_fitness": float(best_fitness),
                "best_alpha": float(best_alpha),
                "mean_fitness": float(np.mean(fitness_scores)),
            }
        )

        # Selection: Keep top 50%
        sorted_indices = np.argsort(fitness_scores)
        elite = population[sorted_indices[: population_size // 2]]

        # Reproduction: Crossover and mutation
        offspring = []
        for _ in range(population_size - len(elite)):
            # Random crossover
            parent1, parent2 = elite[np.random.choice(len(elite), 2, replace=False)]
            child = (parent1 + parent2) / 2
            # Mutation
            child += np.random.normal(0, 0.05 * (alpha_max - alpha_min))
            child = np.clip(child, alpha_min, alpha_max)
            offspring.append(child)

        population = np.concatenate([elite, offspring])

    # Final evaluation
    final_fitness_scores = []
    final_metrics_list = []
    for alpha in population:
        metrics = simulate_trajectory(alpha, profile, seed=seed + generations)
        fitness = evaluate_fitness(metrics, profile)
        final_fitness_scores.append(fitness)
        final_metrics_list.append(metrics)

    best_idx = np.argmin(final_fitness_scores)
    return (
        float(population[best_idx]),
        final_metrics_list[best_idx],
        float(final_fitness_scores[best_idx]),
        history,
    )


def create_visualization(metrics: dict, profile: dict) -> str:
    """

    Create trajectory visualization and return as base64-encoded PNG.
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    time = np.array(metrics["time"])
    altitude = np.array(metrics["altitude"])
    velocity = np.array(metrics["velocity"])

    # Altitude vs Time
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

    # Velocity vs Time
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

    # Save to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()

    return image_base64


def main():
    parser = argparse.ArgumentParser(description="QuASIM × SpaceX/NASA Pilot Track Demo")
    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help="Path to MECO/hot-staging profile JSON",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of evolutionary generations",
    )
    parser.add_argument(
        "--pop",
        type=int,
        default=20,
        help="Population size for evolutionary algorithm",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic execution",
    )

    args = parser.parse_args()

    # Load profile
    print(f"Loading profile: {args.profile}")
    profile = load_profile(args.profile)
    print(f"Profile: {profile['name']}")
    print(f"Description: {profile['description']}")

    # Run optimization
    print("\nRunning evolutionary optimization...")
    print(f"  Generations: {args.generations}")
    print(f"  Population: {args.pop}")
    print(f"  Seed: {args.seed}")

    best_alpha, best_metrics, best_fitness, history = evolutionary_optimization(
        profile, generations=args.generations, population_size=args.pop, seed=args.seed
    )

    print("\nOptimization complete!")
    print(f"  Best alpha: {best_alpha:.6f}")
    print(f"  Best fitness (RMSE): {best_fitness:.6f}")
    print(f"  Peak altitude: {best_metrics['peak_altitude_km']:.2f} km")
    print(f"  MECO altitude: {best_metrics['meco_altitude_km']:.2f} km")
    print(f"  MECO velocity: {best_metrics['meco_velocity_kms']:.4f} km/s")
    print(f"  MECO time: {best_metrics['meco_time_s']:.2f} s")

    # Generate visualization
    print("\nGenerating visualization...")
    viz_base64 = create_visualization(best_metrics, profile)

    # Calculate fidelity metrics
    targets = profile["targets"]
    rmse_altitude = (
        abs(best_metrics["peak_altitude_km"] - targets["peak_altitude_km"])
        / targets["peak_altitude_km"]
    )
    rmse_velocity = (
        abs(best_metrics["meco_velocity_kms"] - targets["meco_velocity_kms"])
        / targets["meco_velocity_kms"]
    )

    fidelity = 1.0 - (rmse_altitude + rmse_velocity) / 2

    # Create output report
    profile_name = Path(args.profile).stem
    output_file = f"{profile_name}_demo_report.json"

    report = {
        "profile": profile["name"],
        "description": profile["description"],
        "optimized_alpha": best_alpha,
        "fitness_rmse": best_fitness,
        "metrics": {
            "peak_altitude_km": best_metrics["peak_altitude_km"],
            "meco_altitude_km": best_metrics["meco_altitude_km"],
            "meco_velocity_kms": best_metrics["meco_velocity_kms"],
            "meco_time_s": best_metrics["meco_time_s"],
        },
        "validation_metrics": {
            "rmse_altitude_pct": rmse_altitude * 100,
            "rmse_velocity_pct": rmse_velocity * 100,
            "fidelity": fidelity,
        },
        "targets": targets,
        "optimization_history": history,
        "configuration": {
            "generations": args.generations,
            "population_size": args.pop,
            "seed": args.seed,
        },
        "visualization_png_base64": viz_base64,
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved: {output_file}")
    print(f"  RMSE altitude: {rmse_altitude * 100:.2f}%")
    print(f"  RMSE velocity: {rmse_velocity * 100:.2f}%")
    print(f"  Fidelity: {fidelity:.4f}")
    print("\nDemo complete! ✓")

    return 0


if __name__ == "__main__":
    sys.exit(main())
