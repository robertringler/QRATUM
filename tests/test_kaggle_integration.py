#!/usr/bin/env python3
"""Simple test for Kaggle Chess integration.

Validates basic functionality of the Kaggle integration modules.
"""

import sys
from pathlib import Path

# Add repository to path
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from qratum_chess.benchmarks.benchmark_kaggle import (
    KaggleBenchmarkResult,
    KaggleBenchmarkRunner,
)
from qratum_chess.benchmarks.kaggle_integration import (
    KaggleLeaderboardLoader,
)
from qratum_chess.core.position import Position


def test_kaggle_loader():
    """Test KaggleLeaderboardLoader basic functionality."""
    print("Testing KaggleLeaderboardLoader...")

    # Create mock data
    mock_data = {
        "benchmarkName": "chess",
        "version": "1",
        "submissions": [
            {"teamName": "Team A", "score": 0.95, "rank": 1, "submissionDate": "2024-01-01"},
            {"teamName": "Team B", "score": 0.90, "rank": 2, "submissionDate": "2024-01-02"},
        ],
        "testData": [
            {
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "id": "test_001",
                "bestMove": "e2e4",
                "evaluation": 0.25,
                "category": "opening",
                "difficulty": "easy",
            }
        ],
    }

    loader = KaggleLeaderboardLoader()
    leaderboard = loader.load_from_dict(mock_data)

    assert leaderboard.benchmark_name == "chess"
    assert leaderboard.version == "1"
    assert len(leaderboard.submissions) == 2
    assert len(leaderboard.test_positions) == 1

    # Test top submissions
    top_submissions = loader.get_top_submissions(n=1, leaderboard=leaderboard)
    assert len(top_submissions) == 1
    assert top_submissions[0].team_name == "Team A"
    assert top_submissions[0].rank == 1

    print("✓ KaggleLeaderboardLoader tests passed")


def test_standard_positions():
    """Test generation of standard benchmark positions."""
    print("Testing standard position generation...")

    loader = KaggleLeaderboardLoader()

    # Load with empty data (should generate standard positions)
    leaderboard = loader.load_from_dict({})

    assert len(leaderboard.test_positions) > 0
    print(f"  Generated {len(leaderboard.test_positions)} standard positions")

    # Validate positions
    for pos in leaderboard.test_positions:
        assert isinstance(pos.position, Position)
        assert pos.fen
        assert pos.test_id

    print("✓ Standard position generation tests passed")


def test_benchmark_runner():
    """Test KaggleBenchmarkRunner basic functionality."""
    print("Testing KaggleBenchmarkRunner...")

    # Create a simple mock engine
    class MockEngine:
        def search(self, position, depth=10, time_limit_ms=None):
            from dataclasses import dataclass

            legal_moves = position.generate_legal_moves()
            best_move = legal_moves[0] if legal_moves else None
            eval_score = 0.0

            # Create a proper SearchStats instance
            @dataclass
            class Stats:
                depth_reached: int = depth
                nodes_searched: int = 1000
                time_ms: float = 100.0

            stats = Stats(depth_reached=depth, nodes_searched=1000, time_ms=100.0)
            return best_move, eval_score, stats

    # Create test leaderboard
    loader = KaggleLeaderboardLoader()
    leaderboard = loader.load_from_dict({})

    # Run benchmark with mock engine
    runner = KaggleBenchmarkRunner()
    engine = MockEngine()

    # Test with just 3 positions
    leaderboard.test_positions = leaderboard.test_positions[:3]

    results = runner.run_benchmark(engine, leaderboard, depth=5, max_positions=3)

    assert len(results) == 3
    assert all(isinstance(r, KaggleBenchmarkResult) for r in results)

    # Test summary generation
    summary = runner.generate_summary(results)
    assert summary.total_positions == 3
    assert summary.avg_depth > 0
    assert summary.avg_nodes > 0
    assert summary.avg_time_ms > 0

    print("✓ KaggleBenchmarkRunner tests passed")


def test_result_serialization():
    """Test that results can be serialized to JSON."""
    print("Testing result serialization...")

    import json

    result = KaggleBenchmarkResult(
        test_id="test_001",
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        best_move="e2e4",
        evaluation=0.25,
        depth_reached=10,
        nodes_searched=1000,
        time_ms=100.0,
        category="opening",
        difficulty="easy",
    )

    # Convert to dict and serialize
    result_dict = result.to_dict()
    json_str = json.dumps(result_dict, indent=2)

    # Verify it can be deserialized
    loaded = json.loads(json_str)
    assert loaded["test_id"] == "test_001"
    assert loaded["best_move"] == "e2e4"

    print("✓ Result serialization tests passed")


def test_gauntlet_integration():
    """Test Kaggle integration with AdversarialGauntlet."""
    print("Testing gauntlet integration...")

    from qratum_chess.benchmarks.gauntlet import AdversarialGauntlet, AdversaryType

    # Verify Kaggle adversary type exists
    assert hasattr(AdversaryType, "KAGGLE")
    assert AdversaryType.KAGGLE.value == "kaggle"

    # Verify gauntlet has Kaggle method
    gauntlet = AdversarialGauntlet()
    assert hasattr(gauntlet, "run_kaggle_benchmark_adversary")

    print("✓ Gauntlet integration tests passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Kaggle Chess Integration Tests")
    print("=" * 60)
    print()

    try:
        test_kaggle_loader()
        print()

        test_standard_positions()
        print()

        test_benchmark_runner()
        print()

        test_result_serialization()
        print()

        test_gauntlet_integration()
        print()

        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
