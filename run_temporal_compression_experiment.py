#!/usr/bin/env python3
"""QRATUM Temporal Compression Experiment Runner.

This script executes the experimental validation of deterministic temporal
compression in computational systems, designed for DARPA/NSF requirements.

Usage:
    # Run with default configuration
    python run_temporal_compression_experiment.py

    # Run quick validation
    python run_temporal_compression_experiment.py --quick

    # Customize parameters
    python run_temporal_compression_experiment.py --steps 50000 --size 500

    # Enable branching exploration
    python run_temporal_compression_experiment.py --branching

    # Specify output directory
    python run_temporal_compression_experiment.py --output artifacts/my_experiment

    # Full experiment with all options
    python run_temporal_compression_experiment.py --steps 100000 --size 1000 --reps 5 --branching

Hypotheses:
    H₀ (Null): Deterministic temporal simulation provides no meaningful advantage
    H₁ (Alternative): Deterministic temporal simulation achieves measurable compression

Outputs:
    - experiment_results.json: Complete experiment data
    - verification_chain.json: Cryptographic audit trail
    - metrics_summary.json: Aggregated metrics
    - TEMPORAL_COMPRESSION_REPORT.md: Human-readable report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from temporal_compression import (
    ExperimentConfig,
    TemporalCompressionExperiment,
)


def print_banner() -> None:
    """Print experiment banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   QRATUM Temporal Compression Experimental Validation                        ║
║                                                                              ║
║   Deterministic Temporal Simulation for DARPA/NSF Review                     ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Hypotheses Under Test:                                                     ║
║   H₀: No meaningful temporal compression advantage                           ║
║   H₁: Measurable temporal compression exceeding real-time                    ║
║                                                                              ║
║   System: Rule 110 Cellular Automaton (Turing-complete)                      ║
║   Verification: SHA3-256 Merkle-chained audit trail                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="QRATUM Temporal Compression Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Default experiment
  %(prog)s --quick                  # Quick validation
  %(prog)s --steps 50000 --size 500 # Custom parameters
  %(prog)s --branching              # Enable trajectory branching

Output:
  Results are saved to artifacts/temporal_compression/ by default.
  Use --output to specify a custom directory.
""",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (reduced parameters)",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Number of simulation steps (default: 10000)",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=200,
        help="Cellular automaton size (default: 200)",
    )

    parser.add_argument(
        "--rule",
        type=int,
        default=110,
        help="Cellular automaton rule (default: 110)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Number of repetitions (default: 3)",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Steps between checkpoints (default: 1000)",
    )

    parser.add_argument(
        "--branching",
        action="store_true",
        help="Enable branching simulation exploration",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/temporal_compression",
        help="Output directory (default: artifacts/temporal_compression)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip markdown report generation",
    )

    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> int:
    """Run the temporal compression experiment.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success)
    """
    # Quick mode overrides
    if args.quick:
        args.steps = 1000
        args.size = 50
        args.reps = 2
        args.checkpoint_interval = 100

    # Create configuration
    config = ExperimentConfig(
        system_rule=args.rule,
        system_size=args.size,
        seed=args.seed,
        simulated_steps=args.steps,
        checkpoint_interval=args.checkpoint_interval,
        num_repetitions=args.reps,
        enable_branching=args.branching,
        output_dir=args.output,
    )

    verbose = not args.quiet

    if verbose:
        print_banner()
        print("Configuration:")
        print(f"  Rule: {config.system_rule}")
        print(f"  Size: {config.system_size} cells")
        print(f"  Steps: {config.simulated_steps:,}")
        print(f"  Repetitions: {config.num_repetitions}")
        print(f"  Seed: {config.seed}")
        print(f"  Branching: {'Enabled' if config.enable_branching else 'Disabled'}")
        print(f"  Output: {config.output_dir}")
        print()

    # Create and run experiment
    experiment = TemporalCompressionExperiment(config, verbose=verbose)

    try:
        result = experiment.run()
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        return 130

    # Save results
    output_path = experiment.save_results(result)

    # Generate and save report
    if not args.no_report:
        report = experiment.generate_report(result)
        report_file = output_path / "TEMPORAL_COMPRESSION_REPORT.md"
        with open(report_file, "w") as f:
            f.write(report)
        if verbose:
            print(f"[EXPERIMENT] Report saved to {report_file}")

    # Print summary
    if verbose:
        print()
        print("=" * 70)
        print("EXPERIMENT COMPLETE")
        print("=" * 70)
        print()
        print(f"Temporal Compression Ratio: {result.compression_ratio:.2f}×")
        print(f"Reproducibility: {'✅ Verified' if result.is_reproducible else '❌ Not Verified'}")
        print(
            f"Chain Integrity: {'✅ Valid' if result.verification_chain.verify_integrity() else '❌ Invalid'}"
        )
        print()
        print(f"Hypothesis Evaluation: {result.hypothesis_evaluation['conclusion']}")
        print()
        print(result.hypothesis_evaluation["interpretation"])
        print()
        print(f"Results saved to: {output_path}")
        print()

    # Return success if reproducible and compression achieved
    if result.is_reproducible and result.compression_ratio > 1.0:
        return 0
    else:
        return 1


def main() -> int:
    """Main entry point."""
    args = parse_args()
    return run_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
