#!/usr/bin/env python3
"""Execute Ansys Tier-0 Embedment Package for BM_001 Benchmark.

This script executes the BM_001 (Large-Strain Rubber Block Compression) benchmark
with deterministic reproducibility, comparing Ansys baseline against QuASIM solver.

Usage:
    python3 run_bm001_tier0.py --output results/bm001_tier0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent / "evaluation" / "ansys"))
sys.path.insert(0, str(Path(__file__).parent / "sdk" / "ansys"))

from performance_runner import (
    AnsysBaselineExecutor,
    BenchmarkDefinition,
    PerformanceComparer,
    QuasimExecutor,
    ReportGenerator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bm001_execution.log"),
    ],
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for BM_001 Tier-0 execution."""

    parser = argparse.ArgumentParser(
        description="Execute Ansys Tier-0 Embedment Package for BM_001",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/bm001_tier0"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Number of runs per solver (default: 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic execution (default: 42)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Compute device for QuASIM (default: gpu)",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=60,
        help="Cooldown period between runs in seconds (default: 60)",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path("benchmarks/ansys/benchmark_definitions.yaml"),
        help="Path to benchmark definitions YAML",
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Ansys Tier-0 Embedment Package - BM_001 Execution")
    logger.info("=" * 80)
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Runs per solver: {args.runs}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Cooldown: {args.cooldown}s")
    logger.info("")

    # Load benchmark definition
    try:
        benchmark = BenchmarkDefinition.load_from_yaml(args.yaml, "BM_001")
        logger.info(f"Loaded benchmark: {benchmark.name}")
        logger.info(f"Description: {benchmark.description}")
        logger.info("")
    except Exception as e:
        logger.error(f"Failed to load benchmark definition: {e}")
        return 1

    # Step 1: Execute Ansys baseline solves
    logger.info("-" * 80)
    logger.info("STEP 1: Executing Ansys Baseline Solves")
    logger.info("-" * 80)

    ansys_results = []
    ansys_executor = AnsysBaselineExecutor(benchmark, args.output / "ansys" / "BM_001")

    for run in range(1, args.runs + 1):
        logger.info(f"\n[Ansys Run {run}/{args.runs}]")
        start_time = time.time()

        try:
            result = ansys_executor.execute(run)
            ansys_results.append(result)

            elapsed = time.time() - start_time
            logger.info(f"  ✓ Completed in {elapsed:.2f}s")
            logger.info(f"    Solve time: {result.solve_time:.2f}s")
            logger.info(f"    Iterations: {result.iterations}")
            logger.info(f"    Memory: {result.memory_usage:.2f} GB")
            logger.info(f"    Hash: {result.state_hash[:16]}...")

            # Save individual result
            result_file = args.output / "ansys" / "BM_001" / f"run_{run}.json"
            result_file.parent.mkdir(parents=True, exist_ok=True)
            with open(result_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2)

            # Cooldown between runs
            if run < args.runs:
                logger.info(f"  Cooldown: {args.cooldown}s")
                time.sleep(min(args.cooldown, 5))  # Cap at 5s for CI

        except Exception as e:
            logger.error(f"  ✗ Ansys run {run} failed: {e}", exc_info=True)
            return 1

    logger.info(f"\n✓ Ansys baseline: {len(ansys_results)} runs completed")

    # Step 2: Execute QuASIM solves
    logger.info("\n" + "-" * 80)
    logger.info("STEP 2: Executing QuASIM Solves")
    logger.info("-" * 80)

    quasim_results = []
    quasim_executor = QuasimExecutor(
        benchmark, args.output / "quasim" / "BM_001", device=args.device, random_seed=args.seed
    )

    for run in range(1, args.runs + 1):
        logger.info(f"\n[QuASIM Run {run}/{args.runs}]")
        start_time = time.time()

        try:
            result = quasim_executor.execute(run)
            quasim_results.append(result)

            elapsed = time.time() - start_time
            logger.info(f"  ✓ Completed in {elapsed:.2f}s")
            logger.info(f"    Solve time: {result.solve_time:.2f}s")
            logger.info(f"    Iterations: {result.iterations}")
            logger.info(f"    Memory: {result.memory_usage:.2f} GB")
            logger.info(f"    Hash: {result.state_hash[:16]}...")

            # Save individual result
            result_file = args.output / "quasim" / "BM_001" / f"run_{run}.json"
            result_file.parent.mkdir(parents=True, exist_ok=True)
            with open(result_file, "w") as f:
                json.dump(result.to_dict(), f, indent=2)

            # Cooldown between runs
            if run < args.runs:
                logger.info(f"  Cooldown: {args.cooldown}s")
                time.sleep(min(args.cooldown, 5))  # Cap at 5s for CI

        except Exception as e:
            logger.error(f"  ✗ QuASIM run {run} failed: {e}", exc_info=True)
            return 1

    logger.info(f"\n✓ QuASIM: {len(quasim_results)} runs completed")

    # Step 3: Compare results and compute statistics
    logger.info("\n" + "-" * 80)
    logger.info("STEP 3: Statistical Analysis and Comparison")
    logger.info("-" * 80)

    try:
        # Load acceptance criteria
        with open(args.yaml) as f:
            import yaml

            yaml_data = yaml.safe_load(f)
        acceptance_criteria = yaml_data.get("acceptance_criteria", {})

        comparer = PerformanceComparer(benchmark, acceptance_criteria)
        comparison = comparer.compare(ansys_results, quasim_results)

        # Print summary
        logger.info(f"\nBenchmark: {comparison.benchmark_id}")
        logger.info(f"Status: {'✓ PASS' if comparison.passed else '✗ FAIL'}")
        if not comparison.passed:
            logger.info(f"Failure reason: {comparison.failure_reason}")

        logger.info("\nAccuracy Metrics:")
        for metric, value in comparison.accuracy_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        logger.info("\nPerformance Metrics:")
        perf = comparison.performance_metrics
        logger.info(f"  Ansys median time: {perf['ansys_median_time']:.2f}s")
        logger.info(f"  QuASIM median time: {perf['quasim_median_time']:.2f}s")
        logger.info(f"  Speedup: {perf['speedup']:.2f}x")
        logger.info(f"  Memory overhead: {perf['memory_overhead']:.2f}x")

        logger.info("\nStatistical Analysis:")
        stats = comparison.statistical_analysis
        logger.info(
            f"  Speedup 95% CI: [{stats['speedup_ci_lower']:.2f}, {stats['speedup_ci_upper']:.2f}]"
        )
        logger.info(f"  Ansys outliers: {stats['ansys_outliers']}")
        logger.info(f"  QuASIM outliers: {stats['quasim_outliers']}")
        logger.info(
            f"  Statistical significance: {stats['significance']} (p={stats['p_value']:.3f})"
        )

    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        return 1

    # Step 4: Generate reports
    logger.info("\n" + "-" * 80)
    logger.info("STEP 4: Generating Reports")
    logger.info("-" * 80)

    try:
        report_dir = args.output / "reports"
        report_gen = ReportGenerator([comparison], report_dir)
        report_gen.generate_all()

        logger.info(f"\n✓ Reports generated in: {report_dir}")
        logger.info(f"  - CSV: {report_dir / 'results.csv'}")
        logger.info(f"  - JSON: {report_dir / 'results.json'}")
        logger.info(f"  - HTML: {report_dir / 'report.html'}")
        logger.info(f"  - PDF: {report_dir / 'report.pdf'}")

    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        return 1

    # Step 5: Verify reproducibility
    logger.info("\n" + "-" * 80)
    logger.info("STEP 5: Reproducibility Verification")
    logger.info("-" * 80)

    # Check that all QuASIM runs with same seed produce same hash
    quasim_hashes = [r.state_hash for r in quasim_results]
    unique_hashes = set(quasim_hashes)

    if len(unique_hashes) == 1:
        logger.info("✓ Deterministic reproducibility verified!")
        logger.info(f"  All {len(quasim_results)} QuASIM runs produced identical hash:")
        logger.info(f"  {quasim_hashes[0]}")
    else:
        logger.warning("✗ Non-deterministic behavior detected!")
        logger.warning(
            f"  Found {len(unique_hashes)} unique hashes across {len(quasim_results)} runs"
        )
        for i, hash_val in enumerate(quasim_hashes):
            logger.warning(f"  Run {i + 1}: {hash_val}")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("EXECUTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Benchmark: BM_001 - {benchmark.name}")
    logger.info(f"Ansys runs: {len(ansys_results)}")
    logger.info(f"QuASIM runs: {len(quasim_results)}")
    logger.info(f"Result: {'✓ PASS' if comparison.passed else '✗ FAIL'}")
    logger.info(f"Speedup: {perf['speedup']:.2f}x")
    logger.info(
        f"Displacement error: {comparison.accuracy_metrics.get('displacement_error', 0):.2%}"
    )
    logger.info(f"Stress error: {comparison.accuracy_metrics.get('stress_error', 0):.2%}")
    logger.info(f"Reproducibility: {'✓ Verified' if len(unique_hashes) == 1 else '✗ Failed'}")
    logger.info("=" * 80)

    if comparison.passed and len(unique_hashes) == 1:
        logger.info("\n✓✓✓ BM_001 TIER-0 EXECUTION SUCCESSFUL ✓✓✓")
        return 0
    else:
        logger.error("\n✗✗✗ BM_001 TIER-0 EXECUTION FAILED ✗✗✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
