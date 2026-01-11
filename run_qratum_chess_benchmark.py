#!/usr/bin/env python3
"""QRATUM-Chess Benchmark Execution Script (Stage IV).

This script automatically runs the full QRATUM-Chess benchmarking suite
against the AsymmetricAdaptiveSearch (AAS) engine.

Usage:
    python run_qratum_chess_benchmark.py
    python run_qratum_chess_benchmark.py --output-dir /benchmarks/auto_run
    python run_qratum_chess_benchmark.py --quick  # Quick mode with reduced iterations

Components:
    - Performance benchmarks (nodes/sec, MCTS rollouts, NN latency, hash hit rate)
    - Strategic torture suite (zugzwang, fortress, king hunts, tablebase positions)
    - Elo certification (logistic regression Elo calculation)
    - Resilience tests (hash corruption, NaN injection, thread kill recovery)
    - Telemetry capture (heatmaps, entropy curves, evaluation distributions)
    - Stage III certification verification

Outputs:
    - JSON data files with all metrics
    - CSV summary of benchmark results
    - HTML report with visualizations
    - Telemetry dashboard data
    - Certification status report
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Ensure the repository root is in the path
repo_root = Path(__file__).parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from qratum_chess.benchmarks.runner import BenchmarkRunner, BenchmarkConfig
from qratum_chess.search.aas import AsymmetricAdaptiveSearch


def print_banner() -> None:
    """Print the QRATUM-Chess benchmark banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🤖 QRATUM-Chess "Bob" - Stage IV Benchmarking Suite                        ║
║                                                                              ║
║   Asymmetric Adaptive Search (AAS) Engine Benchmark                          ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Components:                                                                ║
║   • Performance Metrics (NPS, MCTS, NN latency, hash hit rate)              ║
║   • Strategic Torture Suite (zugzwang, fortress, trapped pieces)            ║
║   • Elo Certification (vs Stockfish-level baseline)                         ║
║   • Resilience Testing (failure injection & recovery)                        ║
║   • Telemetry & Snapshot Capture                                            ║
║   • Stage III Certification Verification                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run QRATUM-Chess Stage IV Benchmarking Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="benchmarks/auto_run",
        help="Base output directory for results (default: benchmarks/auto_run)",
    )
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run quick benchmark with reduced iterations",
    )
    
    parser.add_argument(
        "--no-performance",
        action="store_true",
        help="Skip performance benchmarks",
    )
    
    parser.add_argument(
        "--no-torture",
        action="store_true",
        help="Skip strategic torture suite",
    )
    
    parser.add_argument(
        "--no-elo",
        action="store_true",
        help="Skip Elo certification",
    )
    
    parser.add_argument(
        "--no-resilience",
        action="store_true",
        help="Skip resilience tests",
    )
    
    parser.add_argument(
        "--torture-depth",
        type=int,
        default=15,
        help="Search depth for torture suite (default: 15)",
    )
    
    parser.add_argument(
        "--resilience-iterations",
        type=int,
        default=10,
        help="Number of resilience test iterations (default: 10)",
    )
    
    parser.add_argument(
        "--certify",
        action="store_true",
        help="Run Stage III certification verification",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser.parse_args()


def create_benchmark_config(args: argparse.Namespace) -> BenchmarkConfig:
    """Create benchmark configuration from arguments.
    
    Args:
        args: Parsed command line arguments.
        
    Returns:
        Benchmark configuration.
    """
    torture_depth = args.torture_depth
    resilience_iterations = args.resilience_iterations
    
    # Quick mode reduces iterations
    if args.quick:
        torture_depth = min(8, torture_depth)
        resilience_iterations = min(3, resilience_iterations)
    
    return BenchmarkConfig(
        run_performance=not args.no_performance,
        run_torture=not args.no_torture,
        run_elo=not args.no_elo,
        run_resilience=not args.no_resilience,
        run_telemetry=True,
        torture_depth=torture_depth,
        resilience_iterations=resilience_iterations,
        output_telemetry=True,
    )


def run_benchmark(args: argparse.Namespace) -> int:
    """Run the complete benchmark suite.
    
    Args:
        args: Parsed command line arguments.
        
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    print_banner()
    
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    print(f"Benchmark started at: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Session timestamp: {timestamp}")
    print()
    
    # Create configuration
    config = create_benchmark_config(args)
    
    print("Configuration:")
    print(f"  Performance benchmarks: {'ON' if config.run_performance else 'OFF'}")
    print(f"  Strategic torture suite: {'ON' if config.run_torture else 'OFF'}")
    print(f"  Elo certification: {'ON' if config.run_elo else 'OFF'}")
    print(f"  Resilience tests: {'ON' if config.run_resilience else 'OFF'}")
    print(f"  Telemetry capture: {'ON' if config.run_telemetry else 'OFF'}")
    print(f"  Torture depth: {config.torture_depth}")
    print(f"  Resilience iterations: {config.resilience_iterations}")
    print()
    
    # Initialize engine
    print("Initializing AsymmetricAdaptiveSearch engine (Bob)...")
    engine = AsymmetricAdaptiveSearch()
    print("  Engine initialized successfully")
    print()
    
    # Create benchmark runner
    print("Initializing benchmark runner...")
    runner = BenchmarkRunner(config)
    print("  Benchmark runner initialized")
    print()
    
    # Run benchmarks
    print("=" * 60)
    print("EXECUTING BENCHMARK SUITE")
    print("=" * 60)
    print()
    
    try:
        summary = runner.run(engine)
    except Exception as e:
        print(f"\n❌ Benchmark execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    # Run Stage III certification if requested
    if args.certify:
        print("\nRunning Stage III certification verification...")
        summary.certification = runner.certify_against_stage_iii(summary)
    
    # Print summary
    print()
    runner.print_summary(summary)
    
    # Save results
    print("\nSaving results...")
    output_path = runner.save_results(args.output_dir, summary)
    
    # Print final status
    print()
    print("=" * 60)
    if summary.overall_passed:
        print("✅ BENCHMARK SUITE PASSED")
    else:
        print("❌ BENCHMARK SUITE FAILED")
    
    if summary.certification:
        if summary.certification.overall_certified:
            print("🎖️  Stage III CERTIFICATION: PASSED")
        else:
            print("⚠️  Stage III CERTIFICATION: NOT PASSED")
    
    print(f"\nResults saved to: {output_path}")
    print("=" * 60)
    
    return 0 if summary.overall_passed else 1


def main() -> int:
    """Main entry point."""
    args = parse_args()
    return run_benchmark(args)


if __name__ == "__main__":
    sys.exit(main())
