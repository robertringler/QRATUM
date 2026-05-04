"""CLI for aerospace demo.

Commands:
- optimize: Run trajectory optimization
- replay: Replay a saved scenario
- simulate: Run forward simulation
- capture: Run with video capture
"""

import argparse
import sys
from pathlib import Path

from quasim.common.seeding import set_global_seed
from quasim.common.serialize import save_jsonl, save_metrics
from quasim.demos.aerospace.kernels.ascent import simulate_ascent
from quasim.demos.aerospace.scenarios.hot_staging import load_scenario
from quasim.viz.run_capture import RunCapture


def cmd_optimize(args):
    """Run trajectory optimization."""

    set_global_seed(args.seed)

    print(f"Running aerospace optimization with profile={args.profile}, steps={args.steps}")

    # Load scenario
    scenario = load_scenario(args.profile)

    # Run simulation
    results = simulate_ascent(
        scenario=scenario,
        steps=args.steps,
        seed=args.seed,
    )

    # Compute metrics
    metrics = {
        "rmse_altitude": results["rmse_altitude"],
        "rmse_velocity": results["rmse_velocity"],
        "q_max": results["q_max"],
        "fuel_margin": results["fuel_margin"],
    }

    print("\nResults:")
    print(f"  RMSE Altitude: {metrics['rmse_altitude']:.3f} m")
    print(f"  RMSE Velocity: {metrics['rmse_velocity']:.3f} m/s")
    print(f"  Max Dynamic Pressure: {metrics['q_max']:.1f} Pa")
    print(f"  Fuel Margin: {metrics['fuel_margin']:.2f}%")

    # Save artifacts
    output_dir = Path("artifacts/aerospace") / f"run_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_metrics(metrics, output_dir / "metrics.json")
    save_jsonl(results["trace"], output_dir / "log.jsonl")

    print(f"\nArtifacts saved to {output_dir}")

    return 0


def cmd_replay(args):
    """Replay a saved scenario."""

    set_global_seed(args.seed)

    print(f"Replaying scenario: {args.scenario}")

    scenario = load_scenario(args.scenario)

    # Run with capture if requested
    capture = RunCapture() if args.capture else None

    simulate_ascent(
        scenario=scenario,
        steps=100,
        seed=args.seed,
        capture=capture,
    )

    if capture:
        output_dir = Path("artifacts/aerospace") / f"replay_{args.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)

        artifacts = capture.finalize(
            mp4_path=output_dir / "replay.mp4",
            gif_path=output_dir / "replay.gif",
            fps=30,
        )
        print(f"\nVideo artifacts: {list(artifacts.keys())}")

    return 0


def cmd_simulate(args):
    """Run forward simulation."""

    set_global_seed(args.seed)

    print(f"Running aerospace simulation with seed={args.seed}")

    scenario = {
        "vehicle": "generic",
        "mass_kg": 549000,
        "thrust_n": 7607000,
        "isp_s": 282,
    }

    capture = RunCapture() if args.capture else None

    results = simulate_ascent(
        scenario=scenario,
        steps=150,
        seed=args.seed,
        capture=capture,
    )

    print(f"\nFinal altitude: {results['final_altitude']:.1f} m")
    print(f"Final velocity: {results['final_velocity']:.1f} m/s")

    if capture:
        output_dir = Path("artifacts/aerospace") / f"sim_{args.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)

        capture.finalize(
            mp4_path=output_dir / "simulation.mp4",
            fps=30,
        )
        print(f"Artifacts saved to {output_dir}")

    return 0


def main():
    """Main CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="QuASIM Aerospace Demo - Hot-staging & MECO Optimization"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Optimize command
    opt_parser = subparsers.add_parser("optimize", help="Run trajectory optimization")
    opt_parser.add_argument("--steps", type=int, default=200, help="Simulation steps")
    opt_parser.add_argument("--profile", default="starship", help="Vehicle profile")
    opt_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Replay command
    replay_parser = subparsers.add_parser("replay", help="Replay saved scenario")
    replay_parser.add_argument("--scenario", default="hot_staging_v1", help="Scenario name")
    replay_parser.add_argument("--capture", action="store_true", help="Capture video")
    replay_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Run forward simulation")
    sim_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    sim_parser.add_argument("--capture", action="store_true", help="Capture video")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to command handler
    if args.command == "optimize":
        return cmd_optimize(args)
    elif args.command == "replay":
        return cmd_replay(args)
    elif args.command == "simulate":
        return cmd_simulate(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
