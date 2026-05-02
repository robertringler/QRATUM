#!/usr/bin/env python
"""Command-line interface for QCMG field simulation.

This module provides a command-line interface for running QCMG field simulations
with configurable parameters and output options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from quasim.sim.qcmg_field import (QCMGParameters,
                                   QuantacosmomorphysigeneticField)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser.

    Returns:
        Configured argument parser
    """

    parser = argparse.ArgumentParser(
        description="QCMG Field Simulation - Quantacosmorphysigenetic field evolution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of evolution iterations",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=64,
        help="Number of spatial grid points",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Time step for integration",
    )
    parser.add_argument(
        "--coupling",
        type=float,
        default=1.0,
        help="Coupling strength between fields",
    )
    parser.add_argument(
        "--dissipation",
        type=float,
        default=0.01,
        help="Energy dissipation rate",
    )
    parser.add_argument(
        "--init-mode",
        type=str,
        default="gaussian",
        choices=["gaussian", "soliton", "random"],
        help="Field initialization mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable history tracking in export (saves memory)",
    )

    return parser


def run_simulation(args: argparse.Namespace) -> int:
    """Run QCMG field simulation with given arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """

    try:
        # Create parameters
        params = QCMGParameters(
            grid_size=args.grid_size,
            dt=args.dt,
            coupling_strength=args.coupling,
            dissipation_rate=args.dissipation,
            random_seed=args.seed,
        )

        if args.verbose:
            print("=" * 60)
            print("QCMG Field Simulation")
            print("=" * 60)
            print(f"Grid size: {params.grid_size}")
            print(f"Time step: {params.dt}")
            print(f"Coupling: {params.coupling_strength}")
            print(f"Dissipation: {params.dissipation_rate}")
            print(f"Iterations: {args.iterations}")
            print(f"Init mode: {args.init_mode}")
            if params.random_seed is not None:
                print(f"Random seed: {params.random_seed}")
            print("=" * 60)

        # Initialize field
        field = QuantacosmomorphysigeneticField(params)
        field.initialize(mode=args.init_mode)

        if args.verbose:
            initial_state = field.history[-1]
            print("\nInitial state:")
            print(f"  Coherence: {initial_state.coherence:.6f}")
            print(f"  Entropy: {initial_state.entropy:.6f}")
            print(f"  Energy: {initial_state.energy:.6f}")
            print()

        # Evolution loop
        if args.verbose:
            print(f"Evolving for {args.iterations} iterations...")
            print()

            # Print header
            print(f"{'Step':>6} {'Time':>10} {'Coherence':>12} {'Entropy':>12} {'Energy':>12}")
            print("-" * 60)

            # Evolution with progress reporting
            report_interval = max(1, args.iterations // 10)
            for i in range(args.iterations):
                state = field.evolve(steps=1)

                if (i + 1) % report_interval == 0 or i == 0:
                    print(
                        f"{i + 1:6d} {state.time:10.4f} {state.coherence:12.6f} "
                        f"{state.entropy:12.6f} {state.energy:12.6f}"
                    )
        else:
            # Evolve silently
            state = field.evolve(steps=args.iterations)

        # Final state
        final_state = field.history[-1]

        if args.verbose:
            print()
            print("=" * 60)
            print("Final state:")
            print(f"  Time: {final_state.time:.6f}")
            print(f"  Coherence: {final_state.coherence:.6f}")
            print(f"  Entropy: {final_state.entropy:.6f}")
            print(f"  Energy: {final_state.energy:.6f}")
            print("=" * 60)
        else:
            # Minimal output
            print(f"Simulation complete: {args.iterations} iterations")
            print(
                f"Final: C={final_state.coherence:.4f}, "
                f"S={final_state.entropy:.4f}, E={final_state.energy:.4f}"
            )

        # Export if requested
        if args.export:
            export_path = Path(args.export)

            if args.verbose:
                print(f"\nExporting to {export_path}...")

            field.save_to_json(str(export_path), include_history=not args.no_history)

            if args.verbose:
                print(f"Export complete: {export_path}")

        return 0

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback  # pylint: disable=import-outside-toplevel

            traceback.print_exc()
        return 1


def main() -> None:
    """Main entry point for CLI."""

    parser = create_parser()
    args = parser.parse_args()
    sys.exit(run_simulation(args))


if __name__ == "__main__":
    main()
