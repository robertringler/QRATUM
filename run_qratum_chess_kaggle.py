#!/usr/bin/env python3
"""QRATUM-Chess Kaggle Competition Runner.

Fully automated pipeline for Kaggle chess competition domination:
- Downloads real competition data
- Runs actual QRATUM engine (no mocks)
- Generates compliant submissions
- Submits automatically
- Tracks ranking and performance

Usage:
    python run_qratum_chess_kaggle.py --competition chess-positions --submit
    python run_qratum_chess_kaggle.py --input data/test.csv --output submission.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add repository to path
repo_root = Path(__file__).parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from qratum_chess.core.position import Move, Position
from qratum_chess.kaggle.client import KaggleClient
from qratum_chess.kaggle.config import KaggleConfig
from qratum_chess.kaggle.leaderboard import LeaderboardTracker
from qratum_chess.kaggle.submission import (
    SubmissionFormatter,
    SubmissionValidator,
    create_submission_metadata,
)
from qratum_chess.search.aas import AsymmetricAdaptiveSearch


def print_banner() -> None:
    """Print pipeline banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🏆 QRATUM-Chess Kaggle Competition Pipeline                                ║
║                                                                              ║
║   Objective: Sustained Leaderboard Supremacy                                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="QRATUM-Chess Kaggle Competition Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--competition", "-c", type=str, help="Kaggle competition ID")

    parser.add_argument(
        "--input", "-i", type=str, help="Input CSV file with positions (if not downloading)"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="submissions/latest.csv",
        help="Output submission CSV file",
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="AsymmetricAdaptiveSearch",
        choices=["AsymmetricAdaptiveSearch"],
        help="Engine to use",
    )

    parser.add_argument("--depth", type=int, default=15, help="Search depth")

    parser.add_argument(
        "--enable-trimodal", action="store_true", help="Enable tri-modal cortex fusion"
    )

    parser.add_argument(
        "--enable-novelty-pressure", action="store_true", help="Enable novelty pressure"
    )

    parser.add_argument(
        "--novelty-pressure", type=float, default=0.5, help="Novelty pressure value (0.0-1.0)"
    )

    parser.add_argument(
        "--disable-randomness", action="store_true", help="Disable randomness for reproducibility"
    )

    parser.add_argument("--submit", action="store_true", help="Submit to Kaggle after generation")

    parser.add_argument(
        "--optimize", action="store_true", help="Enable hyperparameter optimization"
    )

    parser.add_argument("--track", action="store_true", help="Track leaderboard ranking")

    parser.add_argument(
        "--no-mock", action="store_true", help="Enforce no mock data (real competition only)"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/kaggle/current",
        help="Data directory for competition datasets",
    )

    parser.add_argument(
        "--feedback-file",
        type=str,
        default="benchmarks/kaggle_feedback.json",
        help="Feedback file for optimization",
    )

    return parser.parse_args()


def load_positions_from_csv(csv_file: Path) -> list[tuple[str, Position]]:
    """Load chess positions from CSV file.

    Args:
        csv_file: Path to CSV file

    Returns:
        List of (id, Position) tuples
    """
    import csv

    positions = []

    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pos_id = row.get("id", row.get("Id", ""))
            fen = row.get("fen", row.get("FEN", row.get("position", "")))

            if not fen:
                print(f"Warning: No FEN for position {pos_id}")
                continue

            try:
                position = Position.from_fen(fen)
                positions.append((pos_id, position))
            except Exception as e:
                print(f"Warning: Failed to parse position {pos_id}: {e}")
                continue

    return positions


def run_engine_on_positions(
    engine, positions: list[tuple[str, Position]], config: dict[str, Any]
) -> tuple[list[Move], list[float], dict[str, Any]]:
    """Run QRATUM engine on positions.

    Args:
        engine: QRATUM engine instance
        positions: List of (id, Position) tuples
        config: Engine configuration

    Returns:
        Tuple of (moves, evaluations, stats)
    """
    depth = config.get("search_depth", 15)
    time_limit = config.get("time_limit_ms")

    moves = []
    evaluations = []
    stats = {"total_nodes": 0, "total_time_ms": 0.0, "positions_processed": 0}

    print(f"\nRunning QRATUM engine on {len(positions)} positions...")
    print(f"  Engine: {engine.__class__.__name__}")
    print(f"  Depth: {depth}")
    print(f"  Novelty pressure: {config.get('novelty_pressure', 0.0)}")
    print()

    for idx, (pos_id, position) in enumerate(positions):
        print(f"  Position {idx+1}/{len(positions)}: {pos_id}...", end="\r")

        try:
            # Search for best move
            if time_limit:
                best_move, eval_score, search_stats = engine.search(
                    position, depth=depth, time_limit_ms=time_limit
                )
            else:
                best_move, eval_score, search_stats = engine.search(position, depth=depth)

            moves.append(best_move)
            evaluations.append(eval_score)

            if search_stats:
                stats["total_nodes"] += getattr(search_stats, "nodes_searched", 0)
                stats["total_time_ms"] += getattr(search_stats, "time_ms", 0.0)

            stats["positions_processed"] += 1

        except Exception as e:
            print(f"\n  Warning: Engine error on {pos_id}: {e}")
            # Use null move as fallback
            moves.append(Move(0, 0))
            evaluations.append(0.0)

    print(f"\n✓ Engine completed {stats['positions_processed']} positions")
    print(f"  Total nodes: {stats['total_nodes']:,}")
    print(f"  Total time: {stats['total_time_ms']/1000:.1f}s")
    if stats["total_time_ms"] > 0:
        nps = stats["total_nodes"] / (stats["total_time_ms"] / 1000)
        print(f"  Nodes/sec: {nps:,.0f}")

    return moves, evaluations, stats


def load_feedback(feedback_file: Path) -> dict[str, Any]:
    """Load optimization feedback from previous runs.

    Args:
        feedback_file: Path to feedback JSON

    Returns:
        Feedback dictionary
    """
    if not feedback_file.exists():
        return {"submissions": [], "best_rank": None, "best_score": None, "parameters": {}}

    with open(feedback_file) as f:
        return json.load(f)


def save_feedback(feedback_file: Path, feedback: dict[str, Any]) -> None:
    """Save optimization feedback.

    Args:
        feedback_file: Path to feedback JSON
        feedback: Feedback dictionary
    """
    feedback_file.parent.mkdir(parents=True, exist_ok=True)

    with open(feedback_file, "w") as f:
        json.dump(feedback, f, indent=2)


def optimize_parameters(
    feedback: dict[str, Any], current_rank: int | None, current_score: float | None
) -> dict[str, Any]:
    """Optimize parameters based on feedback.

    Args:
        feedback: Historical feedback
        current_rank: Current rank
        current_score: Current score

    Returns:
        Updated parameters
    """
    params = feedback.get(
        "parameters",
        {
            "novelty_pressure": 0.5,
            "search_depth": 15,
            "cortex_weights": {"tactical": 0.33, "strategic": 0.33, "conceptual": 0.34},
        },
    )

    best_rank = feedback.get("best_rank")

    if current_rank is None or best_rank is None:
        return params

    # Simple gradient-based optimization
    delta_rank = best_rank - current_rank  # Positive = improved

    # Adjust novelty pressure
    novelty_pressure = params.get("novelty_pressure", 0.5)
    if delta_rank > 0:
        # Improved - increase novelty pressure slightly
        novelty_pressure = min(1.0, novelty_pressure * 1.05)
    elif delta_rank < 0:
        # Worsened - decrease novelty pressure
        novelty_pressure = max(0.0, novelty_pressure * 0.95)

    params["novelty_pressure"] = novelty_pressure

    # Adjust search depth
    search_depth = params.get("search_depth", 15)
    if delta_rank > 0 and search_depth < 20:
        search_depth += 1
    elif delta_rank < 0 and search_depth > 10:
        search_depth -= 1

    params["search_depth"] = search_depth

    print("\n📊 Parameter Optimization:")
    print(f"  Novelty pressure: {params['novelty_pressure']:.3f}")
    print(f"  Search depth: {params['search_depth']}")

    return params


def main() -> int:
    """Main execution function."""
    args = parse_args()

    print_banner()

    # Load configuration
    try:
        config = KaggleConfig.from_file()
        print(f"✓ Kaggle credentials loaded for user: {config.username}")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return 1

    # Initialize Kaggle client
    try:
        client = KaggleClient(config)
        print("✓ Kaggle API client initialized")
    except Exception as e:
        print(f"✗ Error initializing client: {e}")
        return 1

    # Load feedback for optimization
    feedback_file = Path(args.feedback_file)
    feedback = load_feedback(feedback_file) if args.optimize else {}

    # Get optimized parameters
    if args.optimize and feedback:
        params = optimize_parameters(feedback, None, None)
    else:
        params = {
            "novelty_pressure": args.novelty_pressure if args.enable_novelty_pressure else 0.0,
            "search_depth": args.depth,
            "cortex_weights": {"tactical": 0.33, "strategic": 0.33, "conceptual": 0.34},
        }

    # Load or download positions
    if args.input:
        # Use provided input file
        input_file = Path(args.input)
        if not input_file.exists():
            print(f"✗ Error: Input file not found: {input_file}")
            return 1

        positions = load_positions_from_csv(input_file)
        print(f"✓ Loaded {len(positions)} positions from {input_file}")

    elif args.competition:
        # Download competition data
        print(f"\nDownloading competition: {args.competition}")

        data_dir = Path(args.data_dir)

        try:
            dataset_info = client.download_competition_data(args.competition, data_dir)
            print(f"✓ Downloaded dataset: {dataset_info.checksum[:12]}...")

            # Find test file
            test_files = [
                f for f in dataset_info.files if "test" in f.lower() and f.endswith(".csv")
            ]
            if not test_files:
                print("✗ Error: No test CSV found in dataset")
                return 1

            test_file = data_dir / test_files[0]
            positions = load_positions_from_csv(test_file)
            print(f"✓ Loaded {len(positions)} positions from {test_file.name}")

        except Exception as e:
            print(f"✗ Error downloading competition data: {e}")
            return 1
    else:
        print("✗ Error: Must specify either --competition or --input")
        return 1

    if not positions:
        print("✗ Error: No positions to process")
        return 1

    # Initialize engine
    print(f"\nInitializing {args.engine} engine...")
    engine = AsymmetricAdaptiveSearch()
    print("✓ Engine initialized")

    # Run engine
    moves, evaluations, engine_stats = run_engine_on_positions(engine, positions, params)

    # Format submission
    print("\nFormatting submission...")
    formatter = SubmissionFormatter()

    predictions = formatter.convert_engine_output(positions, moves, evaluations)

    metadata = create_submission_metadata(engine, params)

    output_file = Path(args.output)
    submission_file = formatter.format_submission(predictions, output_file, metadata=metadata)

    # Validate submission
    print("\nValidating submission...")
    validator = SubmissionValidator()

    # Create position dictionary for validation
    position_dict = {pid: pos for pid, pos in positions}

    is_valid = validator.validate_submission(submission_file, positions=position_dict)

    if not is_valid:
        print("✗ Validation failed!")
        for error in validator.get_errors():
            print(f"  {error}")
        return 1

    # Submit if requested
    if args.submit:
        if not args.competition:
            print("✗ Error: --competition required for submission")
            return 1

        print(f"\nSubmitting to competition: {args.competition}")

        try:
            result = client.submit_competition(
                args.competition,
                submission_file,
                message=f"QRATUM {metadata.engine_version} | Hash: {metadata.qratum_hash[:8]}",
            )
            print("✓ Submission successful!")

            # Track leaderboard if requested
            if args.track:
                tracker = LeaderboardTracker(args.competition, config.username)

                # Wait for score
                entry = tracker.wait_for_score(max_wait=300, poll_interval=15)

                if entry:
                    tracker.print_status()

                    # Save history
                    history_file = Path("benchmarks") / f"kaggle_history_{args.competition}.json"
                    tracker.save_history(history_file)

                    # Update feedback for optimization
                    if args.optimize:
                        feedback["submissions"].append(
                            {
                                "timestamp": time.time(),
                                "rank": entry.rank,
                                "score": entry.score,
                                "parameters": params,
                            }
                        )

                        if feedback.get("best_rank") is None or entry.rank < feedback["best_rank"]:
                            feedback["best_rank"] = entry.rank
                            feedback["best_score"] = entry.score
                            feedback["best_parameters"] = params

                        feedback["parameters"] = params
                        save_feedback(feedback_file, feedback)
                        print(f"✓ Feedback saved to {feedback_file}")

        except Exception as e:
            print(f"✗ Submission error: {e}")
            return 1

    print("\n" + "=" * 60)
    print("✓ Pipeline completed successfully!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
