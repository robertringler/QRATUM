#!/usr/bin/env python3
"""Run Full QRATUM ASI on AION.

This script executes the complete QRATUM ASI (Artificial Superintelligence)
stack on the AION runtime, leveraging:
- Cross-language/hardware fusion
- Proof-preserving execution
- Adaptive runtime scheduling

Usage:
    python run_qratum_asi_on_aion.py [--output-dir OUTPUT_DIR] [--config CONFIG_FILE]

Version: 1.0.0
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def main() -> int:
    """Main entry point for QRATUM ASI on AION execution."""
    parser = argparse.ArgumentParser(
        description="Execute QRATUM ASI on AION Runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default configuration
    python run_qratum_asi_on_aion.py

    # Run with custom output directory
    python run_qratum_asi_on_aion.py --output-dir ./output

    # Run with custom configuration file
    python run_qratum_asi_on_aion.py --config my_config.json

    # Run with verbose output
    python run_qratum_asi_on_aion.py --verbose
        """,
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory for artifacts (reports, traces, SIR files)",
    )

    parser.add_argument(
        "--config", "-c", type=str, default=None, help="Path to custom ASI configuration JSON file"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--json-output", action="store_true", help="Output results as JSON instead of ASCII report"
    )

    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run benchmarks only without full ASI execution",
    )

    args = parser.parse_args()

    # Print header
    print("=" * 70)
    print("  QRATUM ASI on AION Runtime")
    print("  Full Deployment, Execution, and Optimization")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().isoformat()}")
    print("")

    # Import AION modules
    try:
        from aion.executor import (
            create_default_asi_config,
            generate_ascii_report,
            run_full_qratum_asi_on_aion,
        )
    except ImportError as e:
        print(f"Error: Could not import AION modules: {e}")
        print("Please ensure AION is properly installed.")
        return 1

    # Load configuration
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {args.config}")
            return 1

        with open(config_path) as f:
            config = json.load(f)
        print(f"Loaded configuration from: {args.config}")
    else:
        config = create_default_asi_config()
        print("Using default QRATUM ASI configuration")

    if args.verbose:
        print("\nConfiguration:")
        print(json.dumps(config, indent=2))

    # Set up output directory
    output_dir = args.output_dir
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path}")

    print("\n" + "-" * 70)
    print("Starting QRATUM ASI Execution on AION...")
    print("-" * 70 + "\n")

    # Run benchmarks only if requested
    if args.benchmark_only:
        return run_benchmarks_only(args)

    # Execute full QRATUM ASI on AION
    try:
        result = run_full_qratum_asi_on_aion(
            config=config,
            output_dir=output_dir,
        )
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback

        if args.verbose:
            traceback.print_exc()
        return 1

    # Output results
    if args.json_output:
        report = result.generate_report()
        print(json.dumps(report, indent=2, default=str))
    else:
        print(generate_ascii_report(result))

    # Save to output directory if specified
    if output_dir and result.success:
        print(f"\nArtifacts saved to: {output_dir}")
        print("  - execution_report.json")
        print("  - hypergraph_traces.json")
        print("  - aion_sir.json")

    # Print summary
    print("\n" + "=" * 70)
    if result.success:
        print("  QRATUM ASI Execution COMPLETED SUCCESSFULLY")
        print("=" * 70)

        # Print key metrics
        if result.metrics:
            print("\nKey Performance Metrics:")
            print(f"  - Throughput: {result.metrics.throughput_ops_per_sec:.2e} ops/s")
            print(f"  - Latency: {result.metrics.latency_ms:.2f} ms")
            print(f"  - Tasks Completed: {result.metrics.tasks_completed}")

        if result.optimization_result:
            print(f"  - Optimization Speedup: {result.optimization_result.estimated_speedup:.2f}x")

        if result.verification_result:
            verified = result.verification_result.all_verified
            print(f"  - All Proofs Verified: {'Yes' if verified else 'Partial'}")

        print(f"\nTotal Execution Time: {result.total_execution_time:.3f}s")
        return 0
    else:
        print("  QRATUM ASI Execution FAILED")
        print("=" * 70)

        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  - {error}")

        return 1


def run_benchmarks_only(args: argparse.Namespace) -> int:
    """Run benchmarks only without full ASI execution."""
    print("Running AION Benchmarks...")
    print("-" * 70)

    try:
        from aion.benchmarks import generate_visualization, run_all_benchmarks

        suite = run_all_benchmarks()
        print(generate_visualization(suite))

        # Save results if output directory specified
        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            summary = suite.summary()
            with open(output_path / "benchmark_results.json", "w") as f:
                json.dump(summary, f, indent=2)

            print(f"\nBenchmark results saved to: {output_path / 'benchmark_results.json'}")

        return 0

    except Exception as e:
        print(f"Error running benchmarks: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
