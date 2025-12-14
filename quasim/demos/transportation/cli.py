"""CLI for transportation demo."""

import argparse
import sys
from pathlib import Path

from quasim.common.seeding import set_global_seed
from quasim.common.serialize import save_jsonl, save_metrics
from quasim.demos.transportation.kernels.simulation import run_simulation
from quasim.viz.run_capture import RunCapture


def cmd_plan(args):
    """Run planning."""
    set_global_seed(args.seed)

    print(f"Running transportation planning with seed={args.seed}")

    scenario = {"default": True}
    results = run_simulation(scenario, steps=args.steps, seed=args.seed)

    metrics = {
        k: results[k]
        for k in ["on_time_pct", "energy_cost", "km_traveled", "charge_wait_time"]
        if k in results
    }

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    output_dir = Path("artifacts/transportation") / f"run_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_metrics(metrics, output_dir / "metrics.json")
    save_jsonl(results.get("trace", []), output_dir / "log.jsonl")

    print(f"\nArtifacts saved to {output_dir}")
    return 0


def cmd_simulate(args):
    """Run simulation."""
    set_global_seed(args.seed)

    print(f"Running transportation simulation with seed={args.seed}")

    scenario = {"default": True}
    capture = RunCapture() if args.capture else None

    run_simulation(scenario, steps=150, seed=args.seed, capture=capture)

    print("\nSimulation complete")

    if capture:
        output_dir = Path("artifacts/transportation") / f"sim_{args.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)

        capture.finalize(mp4_path=output_dir / "simulation.mp4", fps=30)
        print(f"Artifacts saved to {output_dir}")

    return 0


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="QuASIM 🚛 Transportation Demo")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    plan_parser = subparsers.add_parser("plan", help="Run planning")
    plan_parser.add_argument("--steps", type=int, default=200, help="Simulation steps")
    plan_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    sim_parser = subparsers.add_parser("simulate", help="Run simulation")
    sim_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    sim_parser.add_argument("--capture", action="store_true", help="Capture video")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "plan":
        return cmd_plan(args)
    elif args.command == "simulate":
        return cmd_simulate(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
