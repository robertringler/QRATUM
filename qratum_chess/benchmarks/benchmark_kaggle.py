#!/usr/bin/env python3
"""QRATUM Chess Kaggle Benchmark Script.

Runs QRATUM AsymmetricAdaptiveSearch engine against Kaggle benchmark positions
and optionally submits results to the Kaggle leaderboard.

Usage:
    python qratum_chess/benchmarks/benchmark_kaggle.py
    python qratum_chess/benchmarks/benchmark_kaggle.py --submit
    python qratum_chess/benchmarks/benchmark_kaggle.py --submit --message "QRATUM v1.0"
    python qratum_chess/benchmarks/benchmark_kaggle.py --input kaggle_data.json
"""Kaggle Chess Benchmark Runner for QRATUM-Chess.

This module runs the QRATUM AsymmetricAdaptiveSearch engine against
Kaggle benchmark positions and analyzes the results.

Usage:
    benchmark = KaggleBenchmarkRunner()
    results = benchmark.run_benchmark(engine, leaderboard)
    benchmark.save_results(results, "benchmarks/kaggle_results/")
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Repository path setup for standalone execution
# Note: For development, consider using: pip install -e .
# This allows proper imports without sys.path manipulation
if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from qratum_chess.benchmarks.kaggle_config import KaggleConfig, load_config
from qratum_chess.benchmarks.kaggle_integration import (
    KaggleIntegration,
    KaggleBenchmarkPosition,
)
from qratum_chess.benchmarks.kaggle_submission import KaggleSubmission
from qratum_chess.search.aas import AsymmetricAdaptiveSearch, AASConfig
from qratum_chess.core.position import Position


def print_banner() -> None:
    """Print the Kaggle benchmark banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🤖 QRATUM Chess - Kaggle Leaderboard Benchmark                             ║
║                                                                              ║
║   AsymmetricAdaptiveSearch vs Kaggle Chess Positions                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run QRATUM Chess engine against Kaggle benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path to Kaggle leaderboard JSON file (if not downloading)",
    )
    
    parser.add_argument(
        "--download", "-d",
        action="store_true",
        help="Download fresh data from Kaggle API",
    )
    
    parser.add_argument(
        "--competition",
        type=str,
        default="chess-engine-leaderboard",
        help="Kaggle competition ID (default: chess-engine-leaderboard)",
    )
    
    parser.add_argument(
        "--submit", "-s",
        action="store_true",
        help="Submit results to Kaggle leaderboard",
    )
    
    parser.add_argument(
        "--message", "-m",
        type=str,
        default="QRATUM Chess Engine - AsymmetricAdaptiveSearch",
        help="Submission message/description",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate submission format without actually submitting",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="kaggle_benchmark_results.json",
        help="Output file for results (default: kaggle_benchmark_results.json)",
    )
    
    parser.add_argument(
        "--depth",
        type=int,
        default=15,
        help="Search depth for engine (default: 15)",
    )
    
    parser.add_argument(
        "--time-limit",
        type=float,
        default=5000.0,
        help="Time limit per position in milliseconds (default: 5000)",
    )
    
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use sample positions instead of real Kaggle data",
    )
    
    parser.add_argument(
        "--credentials",
        type=str,
        help="Path to kaggle.json credentials file",
    )
    
    parser.add_argument(
        "--use-env",
        action="store_true",
        help="Use environment variables for Kaggle credentials",
    )
    
    return parser.parse_args()


def run_benchmark_on_position(
    engine: AsymmetricAdaptiveSearch,
    position: KaggleBenchmarkPosition,
    depth: int,
    time_limit_ms: float
) -> dict[str, Any]:
    """Run benchmark on a single position.
    
    Args:
        engine: Chess engine to use.
        position: Benchmark position.
        depth: Search depth.
        time_limit_ms: Time limit in milliseconds.
        
    Returns:
        Benchmark result dictionary.
    """
    if position.position is None:
        return {
            "position_id": position.position_id,
            "error": "Invalid position",
            "best_move": "",
            "evaluation": 0.0,
            "nodes_searched": 0,
            "time_ms": 0.0,
        }
    
    try:
        start_time = time.perf_counter()
        
        # Search for best move
        best_move, evaluation, stats = engine.search(
            position.position,
            depth=depth,
            time_limit_ms=time_limit_ms
        )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        result = {
            "position_id": position.position_id,
            "fen": position.fen,
            "description": position.description,
            "best_move": best_move.to_uci() if best_move else "",
            "evaluation": float(evaluation) if evaluation is not None else 0.0,
            "nodes_searched": stats.nodes_searched if stats else 0,
            "time_ms": elapsed_ms,
            "depth_reached": stats.depth_reached if stats else 0,
        }
        
        # Compare with expected move if available
        if position.expected_move and best_move:
            result["matches_expected"] = (best_move.to_uci() == position.expected_move)
        
        return result
    
    except Exception as e:
        return {
            "position_id": position.position_id,
            "error": str(e),
            "best_move": "",
            "evaluation": 0.0,
            "nodes_searched": 0,
            "time_ms": 0.0,
        }


def run_benchmark_suite(
    positions: list[KaggleBenchmarkPosition],
    depth: int = 15,
    time_limit_ms: float = 5000.0
) -> list[dict[str, Any]]:
    """Run benchmark suite on all positions.
    
    Args:
        positions: List of benchmark positions.
        depth: Search depth.
        time_limit_ms: Time limit per position.
        
    Returns:
        List of benchmark results.
    """
    # Initialize engine
    config = AASConfig(
        opening_width=2.0,
        middlegame_focus=1.5,
        endgame_precision=2.0,
    )
    engine = AsymmetricAdaptiveSearch(config=config)
    
    print(f"\nRunning benchmark on {len(positions)} positions...")
    print(f"Search depth: {depth}")
    print(f"Time limit: {time_limit_ms:.0f} ms per position")
    print("")
    
    results = []
    
    for i, position in enumerate(positions, 1):
        print(f"[{i}/{len(positions)}] Position {position.position_id}: ", end="", flush=True)
        
        result = run_benchmark_on_position(
            engine, position, depth, time_limit_ms
        )
        
        results.append(result)
        
        # Print result
        if "error" in result and result["error"]:
            print(f"ERROR - {result['error']}")
        else:
            move = result.get("best_move", "none")
            eval_score = result.get("evaluation", 0.0)
            nodes = result.get("nodes_searched", 0)
            time_taken = result.get("time_ms", 0.0)
            
            print(f"{move} (eval: {eval_score:+.2f}, nodes: {nodes}, time: {time_taken:.0f}ms)")
    
    return results


def print_benchmark_summary(results: list[dict[str, Any]]) -> None:
    """Print benchmark summary statistics.
    
    Args:
        results: List of benchmark results.
    """
    total = len(results)
    errors = sum(1 for r in results if "error" in r and r["error"])
    successful = total - errors
    
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    print(f"Total positions: {total}")
    print(f"Successful: {successful}")
    print(f"Errors: {errors}")
    
    if successful > 0:
        avg_time = sum(r.get("time_ms", 0) for r in results) / successful
        avg_nodes = sum(r.get("nodes_searched", 0) for r in results) / successful
        avg_depth = sum(r.get("depth_reached", 0) for r in results) / successful
        
        print(f"\nAverage time per position: {avg_time:.2f} ms")
        print(f"Average nodes searched: {avg_nodes:.0f}")
        print(f"Average depth reached: {avg_depth:.1f}")
        
        # Calculate matches if expected moves available
        with_expected = [r for r in results if "matches_expected" in r]
        if with_expected:
            matches = sum(1 for r in with_expected if r["matches_expected"])
            match_rate = matches / len(with_expected) * 100
            print(f"\nExpected move match rate: {match_rate:.1f}% ({matches}/{len(with_expected)})")
    
    print("=" * 70 + "\n")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    print_banner()
    
    # Load Kaggle configuration
    try:
        if args.submit or args.download:
            config = load_config(
                credentials_path=args.credentials,
                use_env=args.use_env
            )
            print("✓ Kaggle credentials loaded")
        else:
            config = None
            print("✓ Running in offline mode (no submission)")
    except Exception as e:
        print(f"✗ Failed to load Kaggle credentials: {e}")
        if args.submit or args.download:
            print("\nTo submit results, you need Kaggle API credentials:")
            print("1. Go to https://www.kaggle.com/settings")
            print("2. Click 'Create New API Token'")
            print("3. Save kaggle.json to ~/.kaggle/kaggle.json")
            return 1
        config = None
    
    # Load or download benchmark positions
    integration = KaggleIntegration(config)
    
    try:
        if args.use_sample:
            print("\n✓ Using sample benchmark positions")
            positions = integration.create_sample_positions()
        elif args.download and config:
            print(f"\n⬇ Downloading data from Kaggle competition: {args.competition}")
            leaderboard_data = integration.download_leaderboard_data(
                competition_id=args.competition,
                save_path="kaggle_leaderboard.json"
            )
            positions = leaderboard_data.positions
            
            if not positions:
                print("⚠ No positions found in leaderboard data, trying to download test data...")
                positions = integration.download_benchmark_positions(
                    competition_id=args.competition,
                    save_path="kaggle_test_positions.json"
                )
            
            if not positions:
                print("⚠ No positions found, using sample positions")
                positions = integration.create_sample_positions()
        elif args.input:
            print(f"\n✓ Loading benchmark data from: {args.input}")
            leaderboard_data = integration.load_leaderboard_from_file(args.input)
            positions = leaderboard_data.positions
        else:
            print("\n✓ Using sample benchmark positions (use --input or --download for real data)")
            positions = integration.create_sample_positions()
        
        print(f"✓ Loaded {len(positions)} benchmark positions")
    
    except Exception as e:
        print(f"✗ Failed to load benchmark positions: {e}")
        return 1
    
    # Run benchmarks
    results = run_benchmark_suite(
        positions,
        depth=args.depth,
        time_limit_ms=args.time_limit
    )
    
    # Print summary
    print_benchmark_summary(results)
    
    # Save results to file
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "engine": "AsymmetricAdaptiveSearch",
            "depth": args.depth,
            "time_limit_ms": args.time_limit,
            "total_positions": len(positions),
        },
        "results": results,
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Results saved to: {args.output}")
    
    # Submit to Kaggle if requested
    if args.submit and config:
        print("\n" + "=" * 70)
        print("Submitting to Kaggle Leaderboard")
        print("=" * 70)
        
        submission = KaggleSubmission(config)
        
        # Submit results
        result = submission.submit_to_kaggle(
            results,
            message=args.message,
            dry_run=args.dry_run
        )
        
        # Display result
        submission.display_submission_summary(result)
        
        # If successful and not dry run, wait for score
        if result.success and not args.dry_run and result.submission_id:
            print("⏳ Waiting for submission to be scored...")
            status = submission.get_submission_status(result.submission_id, timeout=300)
            submission.display_submission_summary(status)
        
        if not result.success:
            return 1
    
    elif args.submit:
        print("\n⚠ --submit requires Kaggle credentials (use --credentials or --use-env)")
    
    print("\n✓ Benchmark complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import time

from qratum_chess.benchmarks.kaggle_integration import (
    KaggleLeaderboard,
    KaggleBenchmarkPosition,
    KaggleLeaderboardLoader,
)
from qratum_chess.core.position import Position, Move


@dataclass
class KaggleBenchmarkResult:
    """Result from running QRATUM engine on a single position.
    
    Attributes:
        test_id: ID of the test position.
        fen: FEN string of the position.
        best_move: Best move found by QRATUM engine (UCI format).
        evaluation: Evaluation score from QRATUM engine.
        depth_reached: Search depth reached.
        nodes_searched: Number of nodes searched.
        time_ms: Time taken in milliseconds.
        expected_move: Expected best move from Kaggle data (if available).
        expected_eval: Expected evaluation from Kaggle data (if available).
        move_matches: Whether the move matches the expected move.
        eval_diff: Difference between QRATUM eval and expected eval.
        category: Position category.
        difficulty: Position difficulty.
    """
    test_id: str
    fen: str
    best_move: str
    evaluation: float
    depth_reached: int
    nodes_searched: int
    time_ms: float
    expected_move: str | None = None
    expected_eval: float | None = None
    move_matches: bool = False
    eval_diff: float | None = None
    category: str = "general"
    difficulty: str = "medium"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "test_id": self.test_id,
            "fen": self.fen,
            "best_move": self.best_move,
            "evaluation": self.evaluation,
            "depth_reached": self.depth_reached,
            "nodes_searched": self.nodes_searched,
            "time_ms": self.time_ms,
            "expected_move": self.expected_move,
            "expected_eval": self.expected_eval,
            "move_matches": self.move_matches,
            "eval_diff": self.eval_diff,
            "category": self.category,
            "difficulty": self.difficulty,
        }


@dataclass
class KaggleBenchmarkSummary:
    """Summary of Kaggle benchmark results.
    
    Attributes:
        total_positions: Total number of positions tested.
        move_accuracy: Percentage of moves matching expected moves.
        avg_eval_diff: Average evaluation difference.
        avg_depth: Average search depth.
        avg_nodes: Average nodes searched.
        avg_time_ms: Average time per position.
        total_time_ms: Total time for all positions.
        results_by_category: Results grouped by category.
        results_by_difficulty: Results grouped by difficulty.
        timestamp: Timestamp of the benchmark run.
    """
    total_positions: int = 0
    move_accuracy: float = 0.0
    avg_eval_diff: float = 0.0
    avg_depth: float = 0.0
    avg_nodes: float = 0.0
    avg_time_ms: float = 0.0
    total_time_ms: float = 0.0
    results_by_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    results_by_difficulty: dict[str, dict[str, Any]] = field(default_factory=dict)
    timestamp: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "total_positions": self.total_positions,
            "move_accuracy": self.move_accuracy,
            "avg_eval_diff": self.avg_eval_diff,
            "avg_depth": self.avg_depth,
            "avg_nodes": self.avg_nodes,
            "avg_time_ms": self.avg_time_ms,
            "total_time_ms": self.total_time_ms,
            "results_by_category": self.results_by_category,
            "results_by_difficulty": self.results_by_difficulty,
            "timestamp": self.timestamp,
        }


class KaggleBenchmarkRunner:
    """Runs QRATUM engine against Kaggle benchmark positions.
    
    Orchestrates benchmark execution, result collection, and analysis.
    """
    
    def __init__(self):
        """Initialize the Kaggle benchmark runner."""
        self.loader = KaggleLeaderboardLoader()
        self.results: list[KaggleBenchmarkResult] = []
    
    def load_leaderboard(self, filepath: str | Path) -> KaggleLeaderboard:
        """Load Kaggle leaderboard data.
        
        Args:
            filepath: Path to the leaderboard JSON file.
            
        Returns:
            Loaded leaderboard data.
        """
        return self.loader.load_from_file(filepath)
    
    def run_benchmark(
        self,
        engine,
        leaderboard: KaggleLeaderboard,
        depth: int = 15,
        time_limit_ms: float | None = None,
        max_positions: int | None = None
    ) -> list[KaggleBenchmarkResult]:
        """Run benchmark on all positions.
        
        Args:
            engine: QRATUM engine instance (e.g., AsymmetricAdaptiveSearch).
            leaderboard: Kaggle leaderboard data.
            depth: Search depth for the engine.
            time_limit_ms: Optional time limit per position.
            max_positions: Optional limit on number of positions to test.
            
        Returns:
            List of benchmark results.
        """
        positions = leaderboard.test_positions
        
        if max_positions:
            positions = positions[:max_positions]
        
        self.results = []
        
        print(f"\n{'='*60}")
        print(f"Running QRATUM-Chess Kaggle Benchmark")
        print(f"{'='*60}")
        print(f"Positions to test: {len(positions)}")
        print(f"Search depth: {depth}")
        if time_limit_ms:
            print(f"Time limit: {time_limit_ms}ms per position")
        print(f"{'='*60}\n")
        
        for idx, benchmark_pos in enumerate(positions):
            print(f"Testing position {idx+1}/{len(positions)}: {benchmark_pos.test_id}")
            
            result = self._run_single_position(
                engine,
                benchmark_pos,
                depth,
                time_limit_ms
            )
            
            self.results.append(result)
            
            # Print immediate result
            status = "✓" if result.move_matches else "✗"
            print(f"  {status} Best move: {result.best_move} "
                  f"(eval: {result.evaluation:.2f}, "
                  f"nodes: {result.nodes_searched}, "
                  f"time: {result.time_ms:.0f}ms)")
        
        return self.results
    
    def _run_single_position(
        self,
        engine,
        benchmark_pos: KaggleBenchmarkPosition,
        depth: int,
        time_limit_ms: float | None
    ) -> KaggleBenchmarkResult:
        """Run engine on a single position.
        
        Args:
            engine: QRATUM engine instance.
            benchmark_pos: Benchmark position to test.
            depth: Search depth.
            time_limit_ms: Optional time limit.
            
        Returns:
            Benchmark result for this position.
        """
        position = benchmark_pos.position
        
        start_time = time.perf_counter()
        
        try:
            # Run engine search
            if time_limit_ms:
                best_move, eval_score, stats = engine.search(
                    position,
                    depth=depth,
                    time_limit_ms=time_limit_ms
                )
            else:
                best_move, eval_score, stats = engine.search(
                    position,
                    depth=depth
                )
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            # Extract stats
            best_move_uci = best_move.to_uci() if best_move else "none"
            depth_reached = stats.depth_reached if stats else depth
            nodes_searched = stats.nodes_searched if stats else 0
            
        except Exception as e:
            # Handle engine errors gracefully
            print(f"  Warning: Engine error on {benchmark_pos.test_id}: {e}")
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            best_move_uci = "error"
            eval_score = 0.0
            depth_reached = 0
            nodes_searched = 0
        
        # Compare with expected results
        move_matches = False
        eval_diff = None
        
        if benchmark_pos.expected_move:
            move_matches = (best_move_uci.lower() == benchmark_pos.expected_move.lower())
        
        if benchmark_pos.expected_eval is not None:
            eval_diff = abs(eval_score - benchmark_pos.expected_eval)
        
        return KaggleBenchmarkResult(
            test_id=benchmark_pos.test_id,
            fen=benchmark_pos.fen,
            best_move=best_move_uci,
            evaluation=eval_score,
            depth_reached=depth_reached,
            nodes_searched=nodes_searched,
            time_ms=elapsed_ms,
            expected_move=benchmark_pos.expected_move,
            expected_eval=benchmark_pos.expected_eval,
            move_matches=move_matches,
            eval_diff=eval_diff,
            category=benchmark_pos.category,
            difficulty=benchmark_pos.difficulty,
        )
    
    def generate_summary(
        self,
        results: list[KaggleBenchmarkResult] | None = None
    ) -> KaggleBenchmarkSummary:
        """Generate summary statistics from results.
        
        Args:
            results: List of results (uses self.results if None).
            
        Returns:
            Summary of benchmark results.
        """
        if results is None:
            results = self.results
        
        if not results:
            return KaggleBenchmarkSummary(timestamp=datetime.now().isoformat())
        
        total = len(results)
        
        # Calculate move accuracy (only for positions with expected moves)
        positions_with_expected = [r for r in results if r.expected_move]
        if positions_with_expected:
            matching_moves = sum(1 for r in positions_with_expected if r.move_matches)
            move_accuracy = matching_moves / len(positions_with_expected)
        else:
            move_accuracy = 0.0
        
        # Calculate average evaluation difference
        eval_diffs = [r.eval_diff for r in results if r.eval_diff is not None]
        avg_eval_diff = sum(eval_diffs) / len(eval_diffs) if eval_diffs else 0.0
        
        # Calculate averages
        avg_depth = sum(r.depth_reached for r in results) / total
        avg_nodes = sum(r.nodes_searched for r in results) / total
        avg_time_ms = sum(r.time_ms for r in results) / total
        total_time_ms = sum(r.time_ms for r in results)
        
        # Group by category
        results_by_category = self._group_by_field(results, "category")
        
        # Group by difficulty
        results_by_difficulty = self._group_by_field(results, "difficulty")
        
        return KaggleBenchmarkSummary(
            total_positions=total,
            move_accuracy=move_accuracy,
            avg_eval_diff=avg_eval_diff,
            avg_depth=avg_depth,
            avg_nodes=avg_nodes,
            avg_time_ms=avg_time_ms,
            total_time_ms=total_time_ms,
            results_by_category=results_by_category,
            results_by_difficulty=results_by_difficulty,
            timestamp=datetime.now().isoformat(),
        )
    
    def _group_by_field(
        self,
        results: list[KaggleBenchmarkResult],
        field: str
    ) -> dict[str, dict[str, Any]]:
        """Group results by a specific field.
        
        Args:
            results: List of results.
            field: Field name to group by.
            
        Returns:
            Dictionary of grouped statistics.
        """
        groups: dict[str, list[KaggleBenchmarkResult]] = {}
        
        for result in results:
            key = getattr(result, field)
            if key not in groups:
                groups[key] = []
            groups[key].append(result)
        
        grouped_stats = {}
        for key, group_results in groups.items():
            total = len(group_results)
            positions_with_expected = [r for r in group_results if r.expected_move]
            
            if positions_with_expected:
                matching = sum(1 for r in positions_with_expected if r.move_matches)
                accuracy = matching / len(positions_with_expected)
            else:
                accuracy = 0.0
            
            grouped_stats[key] = {
                "count": total,
                "move_accuracy": accuracy,
                "avg_time_ms": sum(r.time_ms for r in group_results) / total,
                "avg_nodes": sum(r.nodes_searched for r in group_results) / total,
            }
        
        return grouped_stats
    
    def save_results(
        self,
        output_dir: str | Path,
        results: list[KaggleBenchmarkResult] | None = None
    ) -> Path:
        """Save benchmark results to a directory.
        
        Args:
            output_dir: Output directory path.
            results: List of results (uses self.results if None).
            
        Returns:
            Path to the output directory.
        """
        if results is None:
            results = self.results
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate summary
        summary = self.generate_summary(results)
        
        # Save detailed results
        results_file = output_path / f"kaggle_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(
                {
                    "results": [r.to_dict() for r in results],
                    "summary": summary.to_dict()
                },
                f,
                indent=2
            )
        
        # Save summary only
        summary_file = output_path / f"kaggle_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        print(f"  - Detailed results: {results_file.name}")
        print(f"  - Summary: {summary_file.name}")
        
        return output_path
    
    def print_summary(
        self,
        summary: KaggleBenchmarkSummary | None = None
    ) -> None:
        """Print benchmark summary to console.
        
        Args:
            summary: Summary to print (generates from self.results if None).
        """
        if summary is None:
            summary = self.generate_summary()
        
        print(f"\n{'='*60}")
        print("QRATUM-Chess Kaggle Benchmark Summary")
        print(f"{'='*60}")
        print(f"Total positions tested: {summary.total_positions}")
        print(f"Move accuracy: {summary.move_accuracy*100:.1f}%")
        
        if summary.avg_eval_diff > 0:
            print(f"Avg eval difference: {summary.avg_eval_diff:.2f}")
        
        print(f"Avg search depth: {summary.avg_depth:.1f}")
        print(f"Avg nodes searched: {summary.avg_nodes:.0f}")
        print(f"Avg time per position: {summary.avg_time_ms:.0f}ms")
        print(f"Total time: {summary.total_time_ms/1000:.1f}s")
        
        # Print category breakdown
        if summary.results_by_category:
            print(f"\nResults by Category:")
            for category, stats in summary.results_by_category.items():
                print(f"  {category}: {stats['count']} positions, "
                      f"{stats['move_accuracy']*100:.1f}% accuracy")
        
        # Print difficulty breakdown
        if summary.results_by_difficulty:
            print(f"\nResults by Difficulty:")
            for difficulty, stats in summary.results_by_difficulty.items():
                print(f"  {difficulty}: {stats['count']} positions, "
                      f"{stats['move_accuracy']*100:.1f}% accuracy")
        
        print(f"{'='*60}\n")
    
    def compare_with_leaderboard(
        self,
        leaderboard: KaggleLeaderboard,
        summary: KaggleBenchmarkSummary | None = None
    ) -> dict[str, Any]:
        """Compare QRATUM results with top leaderboard submissions.
        
        Args:
            leaderboard: Kaggle leaderboard data.
            summary: Summary to compare (generates from self.results if None).
            
        Returns:
            Comparison analysis.
        """
        if summary is None:
            summary = self.generate_summary()
        
        top_submissions = self.loader.get_top_submissions(n=5, leaderboard=leaderboard)
        
        comparison = {
            "qratum_score": summary.move_accuracy,
            "qratum_avg_time_ms": summary.avg_time_ms,
            "top_submissions": [
                {
                    "rank": sub.rank,
                    "team_name": sub.team_name,
                    "score": sub.score,
                }
                for sub in top_submissions
            ],
            "estimated_rank": self._estimate_rank(
                summary.move_accuracy,
                [sub.score for sub in leaderboard.submissions]
            ),
        }
        
        return comparison
    
    def _estimate_rank(self, qratum_score: float, all_scores: list[float]) -> int:
        """Estimate QRATUM's rank based on its score.
        
        Args:
            qratum_score: QRATUM's accuracy score.
            all_scores: All scores from the leaderboard.
            
        Returns:
            Estimated rank.
        """
        if not all_scores:
            return 1
        
        sorted_scores = sorted(all_scores, reverse=True)
        rank = 1
        
        for score in sorted_scores:
            if qratum_score >= score:
                break
            rank += 1
        
        return rank
