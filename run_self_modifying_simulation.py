#!/usr/bin/env python3
"""QRATUM-Chess Self-Modifying Engine Simulation Script.

This script runs the QRATUM-Chess engine in full self-modifying mode for
large-scale simulations to capture tactical, strategic, and conceptual
evolution, track motif emergence, and benchmark against baseline.

Usage:
    python run_self_modifying_simulation.py
    python run_self_modifying_simulation.py --games 1000
    python run_self_modifying_simulation.py --output-dir /benchmarks/self_modifying_full_run
    python run_self_modifying_simulation.py --checkpoint-interval 100

Components:
    - Self-modifying engine with meta-dynamics
    - Tri-modal cortex (tactical, strategic, conceptual)
    - Motif discovery and recording
    - ELO progression tracking
    - Checkpoint save/load for recovery
    - Comprehensive telemetry and reporting
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Ensure the repository root is in the path
repo_root = Path(__file__).parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from qratum_chess.benchmarks.runner import BenchmarkConfig, BenchmarkRunner
from qratum_chess.self_modifying import SelfModifyingEngine


# Custom JSON serializer for numpy types (module level to avoid repeated imports)
def json_serializer(obj):
    """Handle numpy and other non-serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "__dict__"):
        return str(obj)
    return str(obj)


def print_banner() -> None:
    """Print the QRATUM-Chess self-modifying simulation banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🤖 QRATUM-Chess "Bob" - Self-Modifying Engine Simulation                   ║
║                                                                              ║
║   Full Meta-Dynamics Evolution Benchmark                                     ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Features:                                                                  ║
║   • Tri-Modal Cortex (Tactical, Strategic, Conceptual)                      ║
║   • Ontology Evolution with Meta-Dynamics                                   ║
║   • Motif Discovery and Recording                                           ║
║   • ELO Progression Tracking                                                ║
║   • Periodic Checkpointing for Recovery                                     ║
║   • Comprehensive Telemetry and Reporting                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run QRATUM-Chess Self-Modifying Engine Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--games",
        "-g",
        type=int,
        default=100,
        help="Number of games to simulate (default: 100, max recommended: 100000)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="benchmarks/self_modifying_full_run",
        help="Base output directory for results",
    )

    parser.add_argument(
        "--checkpoint-dir",
        "-c",
        type=str,
        default="checkpoints",
        help="Directory for checkpoint files",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N games (default: 1000)",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint file",
    )

    # Engine parameters
    parser.add_argument(
        "--tactical-weight",
        type=float,
        default=0.4,
        help="Tactical cortex weight (default: 0.4)",
    )

    parser.add_argument(
        "--strategic-weight",
        type=float,
        default=0.4,
        help="Strategic cortex weight (default: 0.4)",
    )

    parser.add_argument(
        "--conceptual-weight",
        type=float,
        default=0.2,
        help="Conceptual cortex weight (default: 0.2)",
    )

    parser.add_argument(
        "--novelty-pressure",
        type=float,
        default=0.5,
        help="Initial novelty pressure (default: 0.5)",
    )

    parser.add_argument(
        "--memory-decay",
        type=float,
        default=0.01,
        help="Memory kernel decay factor (default: 0.01)",
    )

    parser.add_argument(
        "--recursive-depth-limit",
        type=int,
        default=10,
        help="Maximum recursion depth for self-modification (default: 10)",
    )

    parser.add_argument(
        "--no-ontology-evolution",
        action="store_true",
        help="Disable ontology evolution",
    )

    # Benchmark options
    parser.add_argument(
        "--record-motifs",
        action="store_true",
        default=True,
        help="Enable motif discovery and recording (default: True)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output (only progress and results)",
    )

    return parser.parse_args()


def create_engine(args: argparse.Namespace) -> SelfModifyingEngine:
    """Create and configure the self-modifying engine.

    Args:
        args: Parsed command line arguments.

    Returns:
        Configured SelfModifyingEngine instance.
    """
    engine = SelfModifyingEngine(
        tactical_weight=args.tactical_weight,
        strategic_weight=args.strategic_weight,
        conceptual_weight=args.conceptual_weight,
        novelty_pressure=args.novelty_pressure,
        memory_decay=args.memory_decay,
        ontology_evolution=not args.no_ontology_evolution,
        recursive_depth_limit=args.recursive_depth_limit,
    )

    return engine


def create_benchmark_config(args: argparse.Namespace) -> BenchmarkConfig:
    """Create benchmark configuration from arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        Benchmark configuration.
    """
    return BenchmarkConfig(
        run_performance=True,
        run_torture=True,
        run_elo=True,
        run_resilience=True,
        run_telemetry=True,
        record_motifs=args.record_motifs,
        output_dir=args.output_dir,
    )


def run_simulation(args: argparse.Namespace) -> int:
    """Run the complete self-modifying simulation.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    if not args.quiet:
        print_banner()

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    if not args.quiet:
        print(f"Simulation started at: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Session timestamp: {timestamp}")
        print(f"Target games: {args.games}")
        print()

    # Create engine
    if not args.quiet:
        print("Initializing SelfModifyingEngine...")

    engine = create_engine(args)

    # Resume from checkpoint if specified
    if args.resume:
        if not args.quiet:
            print(f"Resuming from checkpoint: {args.resume}")
        try:
            engine.load_checkpoint(args.resume)
            if not args.quiet:
                print(f"  Resumed at game {engine.games_played}")
        except Exception as e:
            print(f"  Warning: Failed to load checkpoint: {e}")
            print("  Starting fresh simulation")

    if not args.quiet:
        print("  Engine configuration:")
        print(f"    Tactical weight: {engine.engine_config.tactical_weight:.2f}")
        print(f"    Strategic weight: {engine.engine_config.strategic_weight:.2f}")
        print(f"    Conceptual weight: {engine.engine_config.conceptual_weight:.2f}")
        print(f"    Novelty pressure: {engine.engine_config.novelty_pressure:.2f}")
        print(f"    Memory decay: {engine.engine_config.memory_decay:.4f}")
        print(
            f"    Ontology evolution: {'ON' if engine.engine_config.ontology_evolution else 'OFF'}"
        )
        print(f"    Recursive depth limit: {engine.engine_config.recursive_depth_limit}")
        print()

    # Create benchmark config and runner
    config = create_benchmark_config(args)
    runner = BenchmarkRunner(config)

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run simulation
    if not args.quiet:
        print("=" * 60)
        print("EXECUTING SELF-MODIFYING SIMULATION")
        print("=" * 60)
        print()

    start_game = engine.games_played
    target_games = args.games
    simulation_start = time.perf_counter()

    try:
        for i in range(start_game + 1, start_game + target_games + 1):
            # Run single game
            game_summary = runner.run_single_game(engine)
            runner.log_game_summary(game_summary)

            # Progress output
            if not args.quiet and (i % 10 == 0 or i == start_game + 1):
                elapsed = time.perf_counter() - simulation_start
                games_done = i - start_game
                games_per_sec = games_done / max(1, elapsed)
                eta_seconds = (target_games - games_done) / max(0.001, games_per_sec)

                current_elo = engine.elo_history[-1] if engine.elo_history else 2500.0
                motif_count = len(engine.discovered_motifs)

                print(
                    f"Game {i}/{start_game + target_games} | "
                    f"ELO: {current_elo:.0f} | "
                    f"Motifs: {motif_count} | "
                    f"Speed: {games_per_sec:.1f} games/s | "
                    f"ETA: {eta_seconds/60:.1f}m"
                )

            # Verbose output
            if args.verbose:
                print(
                    f"  Result: {game_summary['result']} | "
                    f"Moves: {game_summary['move_count']} | "
                    f"Avg eval: {game_summary['avg_evaluation']:.3f} | "
                    f"Avg novelty: {game_summary['avg_novelty_pressure']:.3f}"
                )

            # Periodic checkpoint
            if i % args.checkpoint_interval == 0:
                checkpoint_path = checkpoint_dir / f"self_modifying_{i}.ckpt"
                engine.save_checkpoint(str(checkpoint_path))
                if not args.quiet:
                    print(f"  📁 Checkpoint saved: {checkpoint_path}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
        print("Saving final checkpoint...")
        final_checkpoint = checkpoint_dir / f"self_modifying_interrupted_{engine.games_played}.ckpt"
        engine.save_checkpoint(str(final_checkpoint))
        print(f"  Checkpoint saved: {final_checkpoint}")

    except Exception as e:
        print(f"\n\n❌ Simulation error: {e}")
        print("Saving error checkpoint...")
        error_checkpoint = checkpoint_dir / f"self_modifying_error_{engine.games_played}.ckpt"
        engine.save_checkpoint(str(error_checkpoint))
        print(f"  Checkpoint saved: {error_checkpoint}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    simulation_end = time.perf_counter()
    total_time = simulation_end - simulation_start

    # Compile summary
    if not args.quiet:
        print()
        print("=" * 60)
        print("COMPILING RESULTS")
        print("=" * 60)
        print()

    summary = runner.compile_full_summary()
    engine_summary = engine.get_engine_summary()

    # Print summary
    if not args.quiet:
        print("SIMULATION SUMMARY")
        print("-" * 60)
        print(f"Total games: {summary['total_games']}")
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"Games per second: {summary['performance']['games_per_second']:.2f}")
        print()
        print("RESULTS:")
        print(f"  Wins: {summary['results']['wins']} ({summary['results']['win_rate']*100:.1f}%)")
        print(
            f"  Draws: {summary['results']['draws']} ({summary['results']['draw_rate']*100:.1f}%)"
        )
        print(
            f"  Losses: {summary['results']['losses']} ({summary['results']['loss_rate']*100:.1f}%)"
        )
        print()
        print("ELO PROGRESSION:")
        print(f"  Start: {summary['elo']['start']:.0f}")
        print(f"  End: {summary['elo']['end']:.0f}")
        print(f"  Delta: {summary['elo']['delta']:+.0f}")
        print()
        print("MOTIFS:")
        print(f"  Total discovered: {summary['motifs']['total_discovered']}")
        print()
        print("META-DYNAMICS:")
        print(f"  Ontology version: {engine_summary['meta_dynamics']['ontology_version']}")
        print(f"  Rule changes: {engine_summary['meta_dynamics']['rule_changes_count']}")
        print(f"  Parameter changes: {engine_summary['meta_dynamics']['parameter_changes_count']}")
        print()
        print("CURRENT ENGINE STATE:")
        print(
            f"  Tactical weight: {engine_summary['current_state']['rules']['tactical_weight']:.4f}"
        )
        print(
            f"  Strategic weight: {engine_summary['current_state']['rules']['strategic_weight']:.4f}"
        )
        print(f"  Novelty weight: {engine_summary['current_state']['rules']['novelty_weight']:.4f}")
        print(f"  Search depth: {engine_summary['current_state']['parameters']['search_depth']}")
        print(
            f"  Novelty pressure: {engine_summary['current_state']['parameters']['novelty_pressure']:.4f}"
        )

    # Save results
    if not args.quiet:
        print()
        print("Saving results...")

    # Save final checkpoint
    final_checkpoint = checkpoint_dir / f"self_modifying_final_{engine.games_played}.ckpt"
    engine.save_checkpoint(str(final_checkpoint))

    # Save JSON results
    results_file = output_dir / f"simulation_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "simulation_summary": summary,
                "engine_summary": engine_summary,
                "timestamp": timestamp,
                "args": vars(args),
            },
            f,
            indent=2,
            default=json_serializer,
        )

    # Save CSV summary
    csv_file = output_dir / f"simulation_metrics_{timestamp}.csv"
    with open(csv_file, "w") as f:
        f.write("metric,value\n")
        f.write(f"total_games,{summary['total_games']}\n")
        f.write(f"wins,{summary['results']['wins']}\n")
        f.write(f"losses,{summary['results']['losses']}\n")
        f.write(f"draws,{summary['results']['draws']}\n")
        f.write(f"win_rate,{summary['results']['win_rate']:.4f}\n")
        f.write(f"elo_start,{summary['elo']['start']:.0f}\n")
        f.write(f"elo_end,{summary['elo']['end']:.0f}\n")
        f.write(f"elo_delta,{summary['elo']['delta']:.0f}\n")
        f.write(f"motifs_discovered,{summary['motifs']['total_discovered']}\n")
        f.write(f"total_time_seconds,{total_time:.2f}\n")
        f.write(f"games_per_second,{summary['performance']['games_per_second']:.4f}\n")

    # Generate HTML report
    html_file = output_dir / f"simulation_report_{timestamp}.html"
    _generate_html_report(html_file, summary, engine_summary, timestamp)

    if not args.quiet:
        print()
        print("=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        print()
        print(f"Results saved to: {output_dir}")
        print(f"  • JSON: {results_file.name}")
        print(f"  • CSV: {csv_file.name}")
        print(f"  • HTML: {html_file.name}")
        print(f"  • Checkpoint: {final_checkpoint}")
        print()

    return 0


def _generate_html_report(
    filepath: Path,
    summary: dict,
    engine_summary: dict,
    timestamp: str,
) -> None:
    """Generate HTML report for simulation results."""
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>QRATUM-Chess Self-Modifying Simulation Report - {timestamp}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; 
               background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 100%); color: #fff; min-height: 100vh; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #00f5ff; text-align: center; margin-bottom: 5px; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}
        .section {{ background: rgba(15,15,25,0.9); border: 1px solid #333; 
                   border-radius: 8px; padding: 20px; margin: 20px 0; }}
        h2 {{ color: #7b2cbf; border-bottom: 1px solid #333; padding-bottom: 10px; margin-top: 0; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .metric {{ background: #1a1a2e; padding: 15px; border-radius: 4px; }}
        .metric-label {{ color: #888; font-size: 0.9em; margin-bottom: 5px; }}
        .metric-value {{ color: #00f5ff; font-size: 1.3em; font-weight: bold; }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff6b6b; }}
        .footer {{ text-align: center; color: #666; margin-top: 30px; padding: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 QRATUM-Chess Self-Modifying Simulation Report</h1>
        <p class="subtitle">Generated: {timestamp}</p>
        
        <div class="section">
            <h2>📊 Simulation Overview</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Total Games</div>
                    <div class="metric-value">{summary['total_games']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">{summary['results']['win_rate']*100:.1f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Games/Second</div>
                    <div class="metric-value">{summary['performance']['games_per_second']:.2f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Time</div>
                    <div class="metric-value">{summary['performance']['total_time_seconds']/60:.1f}m</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🏆 ELO Progression</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Starting ELO</div>
                    <div class="metric-value">{summary['elo']['start']:.0f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Final ELO</div>
                    <div class="metric-value">{summary['elo']['end']:.0f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">ELO Delta</div>
                    <div class="metric-value {'positive' if summary['elo']['delta'] > 0 else 'negative'}">{summary['elo']['delta']:+.0f}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🧩 Motif Discovery</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Total Motifs Discovered</div>
                    <div class="metric-value">{summary['motifs']['total_discovered']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Avg per Game</div>
                    <div class="metric-value">{summary['motifs']['total_discovered'] / max(1, summary['total_games']):.2f}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>⚙️ Meta-Dynamics Evolution</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Ontology Version</div>
                    <div class="metric-value">{engine_summary['meta_dynamics']['ontology_version']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Rule Changes</div>
                    <div class="metric-value">{engine_summary['meta_dynamics']['rule_changes_count']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Parameter Changes</div>
                    <div class="metric-value">{engine_summary['meta_dynamics']['parameter_changes_count']}</div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🧠 Final Cortex State</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Tactical Weight</div>
                    <div class="metric-value">{engine_summary['current_state']['rules']['tactical_weight']:.4f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Strategic Weight</div>
                    <div class="metric-value">{engine_summary['current_state']['rules']['strategic_weight']:.4f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Conceptual Weight</div>
                    <div class="metric-value">{engine_summary['current_state']['rules']['novelty_weight']:.4f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Novelty Pressure</div>
                    <div class="metric-value">{engine_summary['current_state']['parameters']['novelty_pressure']:.4f}</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by QRATUM-Chess Self-Modifying Simulation Runner</p>
            <p>Report timestamp: {timestamp}</p>
        </div>
    </div>
</body>
</html>
"""

    with open(filepath, "w") as f:
        f.write(html_content)


def main() -> int:
    """Main entry point."""
    args = parse_args()
    return run_simulation(args)


if __name__ == "__main__":
    sys.exit(main())
