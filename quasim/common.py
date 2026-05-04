#!/usr/bin/env python3
"""

QuASIM Common Utilities for Vertical Demos

Shared functions for profile-aware evolutionary optimization across
multiple industry verticals. Extracted from quasim_spacex_demo.py.
"""

import base64
import io
import json
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend


def load_profile(profile_path: str) -> dict:
    """Load vertical profile from JSON."""

    with open(profile_path) as f:
        return json.load(f)


def evolutionary_optimization(
    simulate_fn: Callable,
    evaluate_fn: Callable,
    profile: dict,
    generations: int = 50,
    population_size: int = 20,
    seed: int = 42,
) -> tuple:
    """

    Generic evolutionary algorithm to optimize alpha parameter.

    Args:
        simulate_fn: Simulation function that takes (alpha, profile, seed) and returns metrics
        evaluate_fn: Fitness evaluation function that takes (metrics, profile) and returns fitness
        profile: Configuration profile with targets, tolerances, weights, and alpha_bounds
        generations: Number of evolutionary generations
        population_size: Size of population
        seed: Random seed for deterministic execution

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
            metrics = simulate_fn(alpha, profile, seed=seed + gen)
            fitness = evaluate_fn(metrics, profile)
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
        metrics = simulate_fn(alpha, profile, seed=seed + generations)
        fitness = evaluate_fn(metrics, profile)
        final_fitness_scores.append(fitness)
        final_metrics_list.append(metrics)

    best_idx = np.argmin(final_fitness_scores)
    return (
        float(population[best_idx]),
        final_metrics_list[best_idx],
        float(final_fitness_scores[best_idx]),
        history,
    )


def create_visualization_generic(
    metrics: dict, profile: dict, x_key: str, y_keys: list, labels: list, titles: list
) -> str:
    """

    Create generic visualization and return as base64-encoded PNG.

    Args:
        metrics: Dictionary with simulation metrics
        profile: Configuration profile with targets
        x_key: Key for x-axis data in metrics
        y_keys: List of keys for y-axis data in metrics (one per subplot)
        labels: List of labels for each subplot
        titles: List of titles for each subplot

    Returns:
        Base64-encoded PNG string
    """

    n_plots = len(y_keys)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))

    if n_plots == 1:
        axes = [axes]

    x_data = np.array(metrics[x_key])

    for ax, y_key, label, title in zip(axes, y_keys, labels, titles):
        y_data = np.array(metrics[y_key])
        ax.plot(x_data, y_data, "b-", linewidth=2, label=label)

        # Add target line if available
        target_key = f"target_{y_key}"
        if target_key in profile.get("targets", {}):
            ax.axhline(
                y=profile["targets"][target_key],
                color="r",
                linestyle="--",
                label=f"Target {label}",
            )

        ax.set_xlabel(x_key.replace("_", " ").title())
        ax.set_ylabel(label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()

    # Save to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()

    return image_base64


def calculate_fidelity(metrics: dict, targets: dict, metric_keys: list) -> float:
    """

    Calculate overall fidelity as 1 - mean relative error.

    Args:
        metrics: Dictionary with actual metrics
        targets: Dictionary with target values
        metric_keys: List of keys to compare

    Returns:
        Fidelity score (0-1, higher is better)
    """

    errors = []
    for key in metric_keys:
        actual = metrics.get(key, 0)
        target = targets.get(key, 1)
        if target != 0:
            rel_error = abs(actual - target) / abs(target)
            errors.append(rel_error)

    if not errors:
        return 1.0

    return float(1.0 - np.mean(errors))


def generate_report(
    profile: dict,
    best_alpha: float,
    best_fitness: float,
    best_metrics: dict,
    history: list,
    config: dict,
    viz_base64: str,
    validation_metrics: dict,
    compliance_tags: list,
    output_file: str,
) -> None:
    """

    Generate standardized JSON report for vertical demo.

    Args:
        profile: Configuration profile
        best_alpha: Optimized alpha parameter
        best_fitness: Best fitness score
        best_metrics: Best simulation metrics
        history: Optimization history
        config: Configuration parameters (generations, pop, seed)
        viz_base64: Base64-encoded visualization
        validation_metrics: Validation metrics dictionary
        compliance_tags: List of compliance standards
        output_file: Output filename
    """

    report = {
        "profile": profile["name"],
        "description": profile["description"],
        "vertical": profile.get("vertical", "unknown"),
        "optimized_alpha": best_alpha,
        "fitness_rmse": best_fitness,
        "metrics": best_metrics,
        "validation_metrics": validation_metrics,
        "targets": profile["targets"],
        "compliance_tags": compliance_tags,
        "optimization_history": history,
        "configuration": config,
        "visualization_png_base64": viz_base64,
    }

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
