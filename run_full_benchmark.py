#!/usr/bin/env python3
"""CLI wrapper for QRATUM-Chess automated benchmarking and motif extraction.

This script provides a convenient command-line interface to run the complete
Stage IV benchmark suite with automatic motif extraction and report generation.

Usage:
    python run_full_benchmark.py
    python run_full_benchmark.py --quick
    python run_full_benchmark.py --certify --extract-motifs
    python run_full_benchmark.py --output-dir /custom/path --gpu
    python run_full_benchmark.py --help

Features:
    - Complete benchmark suite execution
    - Stage III certification verification
    - Automated motif extraction and classification
    - Comprehensive report generation (JSON, CSV, HTML, PGN)
    - GPU acceleration support
    - Quick mode for faster iteration
    - Progress tracking with ETA estimation
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from qratum_chess.benchmarks.auto_benchmark import (
    AutoBenchmark,
    AutoBenchmarkConfig,
)
from qratum_chess.benchmarks.runner import BenchmarkConfig


def print_banner() -> None:
    """Print the QRATUM-Chess benchmark banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🤖 QRATUM-Chess "Bob" - Automated Benchmarking & Motif Extraction          ║
║                                                                              ║
║   Stage IV Benchmarking Suite + Novel Pattern Discovery                     ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Components:                                                                ║
║   • Performance Metrics (NPS, MCTS, NN latency, hash hit rate)              ║
║   • Strategic Torture Suite (zugzwang, fortress, trapped pieces)            ║
║   • Elo Certification (vs Stockfish-level baseline)                         ║
║   • Resilience Testing (failure injection & recovery)                        ║
║   • Stage III Certification Verification                                     ║
║   • Novel Motif Extraction & Classification                                  ║
║   • Comprehensive Report Generation                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="QRATUM-Chess Automated Benchmarking Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run full benchmark suite
  %(prog)s --quick                            # Quick mode (reduced iterations)
  %(prog)s --certify --extract-motifs         # With certification and motifs
  %(prog)s --output-dir ./my_results --gpu    # Custom output with GPU
  %(prog)s --cpu-only --no-performance        # CPU only, skip performance tests
        """
    )
    
    # Mode options
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run in quick mode with reduced iterations (faster, less comprehensive)",
    )
    
    parser.add_argument(
        "--certify",
        action="store_true",
        help="Run Stage III certification verification (≥75%% Stockfish, ≥70%% Lc0)",
    )
    
    parser.add_argument(
        "--extract-motifs",
        action="store_true",
        default=True,
        help="Enable automated motif extraction and classification (default: enabled)",
    )
    
    parser.add_argument(
        "--no-extract-motifs",
        dest="extract_motifs",
        action="store_false",
        help="Disable motif extraction",
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="benchmarks/auto_run",
        help="Base output directory for results (default: benchmarks/auto_run)",
    )
    
    # Hardware options
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU acceleration (if available)",
    )
    
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Disable GPU and run on CPU only",
    )
    
    # Benchmark component toggles
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
        "--no-telemetry",
        action="store_true",
        help="Disable telemetry capture",
    )
    
    # Benchmark parameters
    parser.add_argument(
        "--torture-depth",
        type=int,
        default=15,
        help="Search depth for torture suite (default: 15, quick mode: 8)",
    )
    
    parser.add_argument(
        "--resilience-iterations",
        type=int,
        default=10,
        help="Number of resilience test iterations (default: 10, quick mode: 3)",
    )
    
    # Advanced options
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpoint/resume capability",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with detailed logging",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="QRATUM-Chess Benchmark Suite v1.0.0",
    )
    
    return parser.parse_args()


def create_auto_config(args: argparse.Namespace) -> AutoBenchmarkConfig:
    """Create automation configuration from arguments.
    
    Args:
        args: Parsed command line arguments.
        
    Returns:
        AutoBenchmarkConfig instance.
    """
    # Create benchmark configuration
    benchmark_config = BenchmarkConfig(
        run_performance=not args.no_performance,
        run_torture=not args.no_torture,
        run_elo=not args.no_elo,
        run_resilience=not args.no_resilience,
        run_telemetry=not args.no_telemetry,
        torture_depth=args.torture_depth,
        resilience_iterations=args.resilience_iterations,
        output_telemetry=not args.no_telemetry,
    )
    
    # Create automation configuration
    config = AutoBenchmarkConfig(
        quick_mode=args.quick,
        certify=args.certify,
        extract_motifs=args.extract_motifs,
        output_dir=args.output_dir,
        gpu_enabled=args.gpu if args.gpu else None,
        cpu_only=args.cpu_only,
        checkpoint_enabled=not args.no_checkpoint,
        benchmark_config=benchmark_config,
    )
    
    return config


def print_config_summary(config: AutoBenchmarkConfig) -> None:
    """Print configuration summary.
    
    Args:
        config: Automation configuration.
    """
    print("\nConfiguration:")
    print(f"  Mode: {'Quick' if config.quick_mode else 'Full'}")
    print(f"  Certification: {'ON' if config.certify else 'OFF'}")
    print(f"  Motif Extraction: {'ON' if config.extract_motifs else 'OFF'}")
    print(f"  Output Directory: {config.output_dir}")
    print(f"  Hardware: ", end="")
    if config.cpu_only:
        print("CPU Only")
    elif config.gpu_enabled:
        print("GPU (forced)")
    else:
        print("Auto-detect")
    
    print("\nBenchmark Components:")
    bc = config.benchmark_config
    print(f"  Performance: {'ON' if bc.run_performance else 'OFF'}")
    print(f"  Torture Suite: {'ON' if bc.run_torture else 'OFF'} (depth: {bc.torture_depth})")
    print(f"  Elo Certification: {'ON' if bc.run_elo else 'OFF'}")
    print(f"  Resilience Tests: {'ON' if bc.run_resilience else 'OFF'} (iterations: {bc.resilience_iterations})")
    print(f"  Telemetry: {'ON' if bc.run_telemetry else 'OFF'}")
    print()


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code (0 for success, 1 for failure).
    """
    # Parse arguments
    args = parse_args()
    
    # Configure logging level
    if args.verbose:
        import logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Print banner
    print_banner()
    
    # Show start time
    start_time = datetime.now()
    print(f"Benchmark started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Session timestamp: {start_time.strftime('%Y%m%d_%H%M%S')}")
    
    # Create configuration
    config = create_auto_config(args)
    print_config_summary(config)
    
    # Initialize automation system
    print("Initializing automation system...")
    auto = AutoBenchmark(config)
    
    # Run full suite
    try:
        results = auto.run_full_suite()
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        print(f"\n\n❌ Benchmark failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    # Check results
    if not results.get("success", False):
        print("\n\n❌ BENCHMARK SUITE FAILED")
        error = results.get("error", "Unknown error")
        print(f"Error: {error}")
        return 1
    
    # Print final summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUITE COMPLETED")
    print("=" * 80)
    
    summary = results.get("summary", {})
    motifs = results.get("motifs", [])
    output_path = results.get("output_path", "N/A")
    elapsed_time = results.get("elapsed_time", 0)
    
    print(f"\nResults:")
    print(f"  Overall Status: {'PASSED' if summary.get('overall_passed', False) else 'FAILED'}")
    print(f"  Total Time: {elapsed_time:.2f} seconds")
    print(f"  Motifs Discovered: {len(motifs)}")
    print(f"  Output Directory: {output_path}")
    
    if config.certify:
        cert = summary.get("certification", {})
        if cert:
            print(f"\nCertification:")
            print(f"  Certified: {'✓ YES' if cert.get('overall_certified', False) else '✗ NO'}")
            print(f"  Stockfish Winrate: {cert.get('stockfish_winrate', 0)*100:.1f}%")
            print(f"  Lc0 Winrate: {cert.get('lc0_winrate', 0)*100:.1f}%")
            print(f"  Motif Emergence: {'✓' if cert.get('motif_emergence', False) else '✗'}")
    
    if motifs:
        print(f"\nTop Motifs:")
        for i, motif in enumerate(motifs[:5], 1):
            print(f"  {i}. {motif.get('motif_id', 'N/A')} - "
                  f"{motif.get('motif_type', 'N/A')} "
                  f"(novelty: {motif.get('novelty_score', 0):.3f})")
    
    print("\n" + "=" * 80)
    print(f"✅ All results saved to: {output_path}")
    print("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
