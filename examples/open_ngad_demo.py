#!/usr/bin/env python3
"""NGAD Defense Simulation Demo.

This script simulates a Next Generation Air Dominance (NGAD) defensive
engagement scenario with configurable parameters.

Features:
- Configurable number of shots and threats via CLI
- Multiple observables (engagement time, kill rates, fuel, g-loads)
- Statistical plots (fuel histogram, g-load vs fuel scatter)
- Error handling for robust execution
- JSON output for post-analysis
- Deterministic reproducibility with seed management
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


class SimulationError(Exception):
    """Custom exception for simulation failures."""

    pass


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="NGAD Defense Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=20,
        help="Number of defensive shots to simulate",
    )
    parser.add_argument(
        "--threats",
        type=int,
        default=10,
        help="Number of incoming threats to simulate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ngad_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="plots",
        help="Directory for output plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation",
    )
    return parser.parse_args()


class ThreatType:
    """Threat classification."""

    CRUISE_MISSILE = "cruise_missile"
    BALLISTIC_MISSILE = "ballistic_missile"
    FIGHTER_AIRCRAFT = "fighter_aircraft"
    DRONE = "drone"


class EngagementSimulator:
    """Simulates NGAD defensive engagements."""

    def __init__(self, seed: int = 42):
        """Initialize simulator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self.threat_types = [
            ThreatType.CRUISE_MISSILE,
            ThreatType.BALLISTIC_MISSILE,
            ThreatType.FIGHTER_AIRCRAFT,
            ThreatType.DRONE,
        ]

    def generate_threat(self) -> Dict[str, Any]:
        """Generate a random threat.

        Returns:
            Dictionary with threat properties
        """
        threat_type = self.rng.choice(self.threat_types)

        # Threat-specific parameters
        if threat_type == ThreatType.CRUISE_MISSILE:
            speed_mach = self.rng.uniform(0.7, 0.9)
            altitude_m = self.rng.uniform(50, 500)
            rcs_m2 = self.rng.uniform(0.5, 2.0)
        elif threat_type == ThreatType.BALLISTIC_MISSILE:
            speed_mach = self.rng.uniform(5.0, 10.0)
            altitude_m = self.rng.uniform(50000, 100000)
            rcs_m2 = self.rng.uniform(0.1, 1.0)
        elif threat_type == ThreatType.FIGHTER_AIRCRAFT:
            speed_mach = self.rng.uniform(1.5, 2.5)
            altitude_m = self.rng.uniform(5000, 15000)
            rcs_m2 = self.rng.uniform(1.0, 5.0)
        else:  # DRONE
            speed_mach = self.rng.uniform(0.1, 0.3)
            altitude_m = self.rng.uniform(1000, 5000)
            rcs_m2 = self.rng.uniform(0.01, 0.1)

        return {
            "type": threat_type,
            "speed_mach": speed_mach,
            "altitude_m": altitude_m,
            "rcs_m2": rcs_m2,
            "range_km": self.rng.uniform(20, 200),
        }

    def simulate_shot(
        self, threat: Dict[str, Any], fuel_remaining: float
    ) -> Tuple[bool, float, float, float]:
        """Simulate a defensive shot.

        Args:
            threat: Threat parameters
            fuel_remaining: Current fuel level (0-100%)

        Returns:
            Tuple of (kill_success, engagement_time, g_load, fuel_used)
        """
        # Check if we have enough fuel
        if fuel_remaining < 5.0:
            return False, 0.0, 0.0, 0.0

        # Engagement time depends on threat type and range
        base_time = threat["range_km"] / (threat["speed_mach"] * 340.0)  # seconds
        engagement_time = base_time * self.rng.uniform(0.8, 1.2)

        # G-load depends on threat speed and maneuverability
        base_g = 5.0
        if threat["type"] == ThreatType.BALLISTIC_MISSILE:
            base_g = 8.0
        elif threat["type"] == ThreatType.FIGHTER_AIRCRAFT:
            base_g = 7.0
        g_load = base_g + self.rng.uniform(-1.0, 3.0)

        # Fuel consumption depends on g-load and engagement time
        fuel_used = (g_load * engagement_time * 0.02) + self.rng.uniform(0.5, 2.0)

        # Kill probability depends on RCS, range, and randomness
        base_pk = 0.85
        rcs_factor = np.clip(1.0 - threat["rcs_m2"] * 0.1, 0.5, 1.0)
        range_factor = np.clip(1.0 - threat["range_km"] * 0.002, 0.5, 1.0)
        pk = base_pk * rcs_factor * range_factor

        kill_success = self.rng.random() < pk

        return kill_success, engagement_time, g_load, fuel_used


def run_simulation(num_shots: int, num_threats: int, seed: int) -> Dict[str, Any]:
    """Run the defense simulation.

    Args:
        num_shots: Number of shots to simulate
        num_threats: Number of threats to generate
        seed: Random seed

    Returns:
        Dictionary with simulation results

    Raises:
        SimulationError: If simulation fails
    """
    try:
        simulator = EngagementSimulator(seed=seed)

        # Generate threats
        threats = [simulator.generate_threat() for _ in range(num_threats)]

        # Initialize tracking
        results = {
            "shots": [],
            "threats": threats,
            "summary": {
                "total_shots": num_shots,
                "total_threats": num_threats,
                "kills": 0,
                "misses": 0,
            },
        }

        fuel_remaining = 100.0  # Start with full fuel

        # Simulate each shot
        for shot_num in range(num_shots):
            # Select threat (cycle if more shots than threats)
            threat = threats[shot_num % len(threats)]

            # Simulate the shot
            kill_success, engagement_time, g_load, fuel_used = simulator.simulate_shot(
                threat, fuel_remaining
            )

            # Update fuel
            fuel_remaining = max(0.0, fuel_remaining - fuel_used)

            # Record result
            shot_result = {
                "shot_number": shot_num + 1,
                "threat_type": threat["type"],
                "kill": bool(kill_success),
                "engagement_time_s": float(engagement_time),
                "g_load": float(g_load),
                "fuel_used": float(fuel_used),
                "fuel_remaining": float(fuel_remaining),
            }
            results["shots"].append(shot_result)

            # Update summary
            if kill_success:
                results["summary"]["kills"] += 1
            else:
                results["summary"]["misses"] += 1

        # Compute additional metrics
        compute_metrics(results)

        return results

    except Exception as e:
        raise SimulationError(f"Simulation failed: {e}") from e


def compute_metrics(results: Dict[str, Any]) -> None:
    """Compute derived metrics from simulation results.

    Args:
        results: Simulation results dictionary (modified in place)
    """
    shots = results["shots"]

    if not shots:
        return

    # Average engagement time
    engagement_times = [s["engagement_time_s"] for s in shots]
    results["summary"]["avg_engagement_time_s"] = float(np.mean(engagement_times))
    results["summary"]["std_engagement_time_s"] = float(np.std(engagement_times))

    # Threat kill rate per type
    threat_stats = {}
    for shot in shots:
        threat_type = shot["threat_type"]
        if threat_type not in threat_stats:
            threat_stats[threat_type] = {"kills": 0, "total": 0}
        threat_stats[threat_type]["total"] += 1
        if shot["kill"]:
            threat_stats[threat_type]["kills"] += 1

    # Compute kill rates
    results["summary"]["threat_kill_rates"] = {}
    for threat_type, stats in threat_stats.items():
        kill_rate = stats["kills"] / stats["total"] if stats["total"] > 0 else 0.0
        results["summary"]["threat_kill_rates"][threat_type] = float(kill_rate)

    # Overall kill rate
    results["summary"]["overall_kill_rate"] = float(results["summary"]["kills"] / len(shots))

    # Fuel statistics
    fuel_values = [s["fuel_remaining"] for s in shots]
    results["summary"]["final_fuel_remaining"] = float(fuel_values[-1])
    results["summary"]["min_fuel_remaining"] = float(min(fuel_values))

    # G-load statistics
    g_loads = [s["g_load"] for s in shots]
    results["summary"]["avg_g_load"] = float(np.mean(g_loads))
    results["summary"]["max_g_load"] = float(np.max(g_loads))


def generate_plots(results: Dict[str, Any], plots_dir: str) -> None:
    """Generate visualization plots.

    Args:
        results: Simulation results
        plots_dir: Directory to save plots

    Raises:
        SimulationError: If plot generation fails
    """
    try:
        # Create plots directory
        Path(plots_dir).mkdir(parents=True, exist_ok=True)

        shots = results["shots"]
        if not shots:
            print("Warning: No shots to plot")
            return

        # Extract data
        fuel_remaining = [s["fuel_remaining"] for s in shots]
        g_loads = [s["g_load"] for s in shots]

        # 1. Histogram of fuel remaining
        plt.figure(figsize=(10, 6))
        plt.hist(fuel_remaining, bins=20, edgecolor="black", alpha=0.7, color="blue")
        plt.xlabel("Fuel Remaining (%)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Fuel Remaining After Engagements")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{plots_dir}/fuel_histogram.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Generated: {plots_dir}/fuel_histogram.png")

        # 2. Scatter plot: g-load vs fuel remaining
        plt.figure(figsize=(10, 6))
        colors = ["green" if s["kill"] else "red" for s in shots]
        plt.scatter(g_loads, fuel_remaining, c=colors, alpha=0.6, s=100)
        plt.xlabel("G-Load")
        plt.ylabel("Fuel Remaining (%)")
        plt.title("G-Load vs Fuel Remaining (Green=Kill, Red=Miss)")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{plots_dir}/g_load_vs_fuel.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Generated: {plots_dir}/g_load_vs_fuel.png")

        # 3. Time series: fuel depletion
        plt.figure(figsize=(10, 6))
        shot_numbers = [s["shot_number"] for s in shots]
        plt.plot(shot_numbers, fuel_remaining, marker="o", linewidth=2)
        plt.xlabel("Shot Number")
        plt.ylabel("Fuel Remaining (%)")
        plt.title("Fuel Depletion Over Engagement Sequence")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{plots_dir}/fuel_depletion.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Generated: {plots_dir}/fuel_depletion.png")

        # 4. Kill rate by threat type
        kill_rates = results["summary"].get("threat_kill_rates", {})
        if kill_rates:
            plt.figure(figsize=(10, 6))
            threat_types = list(kill_rates.keys())
            rates = list(kill_rates.values())
            plt.bar(threat_types, rates, edgecolor="black", alpha=0.7, color="orange")
            plt.xlabel("Threat Type")
            plt.ylabel("Kill Rate")
            plt.title("Kill Rate by Threat Type")
            plt.xticks(rotation=45, ha="right")
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/kill_rate_by_threat.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Generated: {plots_dir}/kill_rate_by_threat.png")

    except Exception as e:
        raise SimulationError(f"Plot generation failed: {e}") from e


def save_results(results: Dict[str, Any], output_file: str) -> None:
    """Save results to JSON file.

    Args:
        results: Simulation results
        output_file: Output file path

    Raises:
        SimulationError: If save fails
    """
    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        raise SimulationError(f"Failed to save results: {e}") from e


def print_summary(results: Dict[str, Any]) -> None:
    """Print simulation summary to console.

    Args:
        results: Simulation results
    """
    summary = results["summary"]
    print("\n" + "=" * 60)
    print("NGAD DEFENSE SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Total Shots:               {summary['total_shots']}")
    print(f"Total Threats:             {summary['total_threats']}")
    print(f"Kills:                     {summary['kills']}")
    print(f"Misses:                    {summary['misses']}")
    print(f"Overall Kill Rate:         {summary.get('overall_kill_rate', 0.0):.1%}")
    print(
        f"Avg Engagement Time:       {summary.get('avg_engagement_time_s', 0.0):.2f}s "
        f"(±{summary.get('std_engagement_time_s', 0.0):.2f}s)"
    )
    print(f"Avg G-Load:                {summary.get('avg_g_load', 0.0):.2f}g")
    print(f"Max G-Load:                {summary.get('max_g_load', 0.0):.2f}g")
    print(f"Final Fuel Remaining:      {summary.get('final_fuel_remaining', 0.0):.1f}%")
    print(f"Min Fuel Remaining:        {summary.get('min_fuel_remaining', 0.0):.1f}%")

    print("\nKill Rates by Threat Type:")
    kill_rates = summary.get("threat_kill_rates", {})
    for threat_type, rate in kill_rates.items():
        print(f"  {threat_type:20s}: {rate:.1%}")
    print("=" * 60)


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Parse arguments
        args = parse_args()

        # Validate inputs
        if args.shots <= 0:
            raise ValueError("Number of shots must be positive")
        if args.threats <= 0:
            raise ValueError("Number of threats must be positive")

        print("Starting NGAD Defense Simulation...")
        print(f"  Shots:   {args.shots}")
        print(f"  Threats: {args.threats}")
        print(f"  Seed:    {args.seed}")

        # Run simulation
        results = run_simulation(args.shots, args.threats, args.seed)

        # Print summary
        print_summary(results)

        # Generate plots
        if not args.no_plots:
            print("\nGenerating plots...")
            generate_plots(results, args.plots_dir)

        # Save results
        save_results(results, args.output)

        print("\n✓ Simulation completed successfully")
        return 0

    except SimulationError as e:
        print(f"\n✗ Simulation error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"\n✗ Invalid input: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
