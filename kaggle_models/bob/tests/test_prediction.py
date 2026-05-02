"""Test suite for BOB Chess Engine prediction functionality."""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from predict import batch_predict, predict


def test_predict_starting_position():
    """Test prediction on starting position."""
    input_data = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "time_limit_ms": 1000,
        "depth": 10,
    }

    result = predict(input_data)

    # Validate result structure
    assert "move" in result
    assert "evaluation" in result
    assert "depth" in result
    assert "nodes" in result
    assert "time_ms" in result
    assert "pv" in result
    assert "engine" in result
    assert "elo" in result

    # Validate move is legal
    board = chess.Board(input_data["fen"])
    move = chess.Move.from_uci(result["move"])
    assert move in board.legal_moves, f"Move {result['move']} is not legal"

    # Validate metadata
    assert result["engine"] == "BOB"
    assert result["elo"] == 1508

    print("✓ Starting position test passed")
    print(f"  Best move: {result['move']}")
    print(f"  Evaluation: {result['evaluation']:.2f}")
    print(f"  Depth: {result['depth']}")
    print(f"  Nodes: {result['nodes']}")
    print(f"  Time: {result['time_ms']:.2f}ms")


def test_predict_tactical_position():
    """Test prediction on tactical position."""
    # Position with fork opportunity
    input_data = {
        "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "time_limit_ms": 1000,
        "depth": 10,
    }

    result = predict(input_data)

    # Validate move is legal
    board = chess.Board(input_data["fen"])
    move = chess.Move.from_uci(result["move"])
    assert move in board.legal_moves, f"Move {result['move']} is not legal"

    print("✓ Tactical position test passed")
    print(f"  Best move: {result['move']}")
    print(f"  Evaluation: {result['evaluation']:.2f}")


def test_predict_endgame_position():
    """Test prediction on endgame position."""
    # King and pawn endgame
    input_data = {"fen": "8/5k2/3p4/8/3P4/5K2/8/8 w - - 0 1", "time_limit_ms": 1000, "depth": 15}

    result = predict(input_data)

    # Validate move is legal
    board = chess.Board(input_data["fen"])
    move = chess.Move.from_uci(result["move"])
    assert move in board.legal_moves, f"Move {result['move']} is not legal"

    print("✓ Endgame position test passed")
    print(f"  Best move: {result['move']}")
    print(f"  Evaluation: {result['evaluation']:.2f}")


def test_batch_predict():
    """Test batch prediction."""
    positions = [
        {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "time_limit_ms": 500,
            "depth": 8,
        },
        {
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "time_limit_ms": 500,
            "depth": 8,
        },
        {
            "fen": "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
            "time_limit_ms": 500,
            "depth": 8,
        },
    ]

    results = batch_predict(positions)

    assert len(results) == len(positions)

    for i, result in enumerate(results):
        board = chess.Board(positions[i]["fen"])
        move = chess.Move.from_uci(result["move"])
        assert move in board.legal_moves, f"Move {result['move']} is not legal in position {i}"

    print(f"✓ Batch predict test passed ({len(results)} positions)")


def test_time_limit():
    """Test that engine respects time limits."""
    input_data = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "time_limit_ms": 500,
        "depth": 20,
    }

    result = predict(input_data)

    # Allow 50% overhead for processing
    assert (
        result["time_ms"] < input_data["time_limit_ms"] * 1.5
    ), f"Engine took {result['time_ms']:.2f}ms, limit was {input_data['time_limit_ms']}ms"

    print("✓ Time limit test passed")
    print(f"  Time used: {result['time_ms']:.2f}ms / {input_data['time_limit_ms']}ms")


def test_checkmate_position():
    """Test prediction in checkmate position."""
    # Position: Black is checkmated
    input_data = {
        "fen": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "time_limit_ms": 100,
        "depth": 5,
    }

    result = predict(input_data)

    # Should still return a result
    assert "move" in result
    assert "evaluation" in result

    print("✓ Checkmate position test passed")
    print(f"  Result: {result['move']}")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 70)
    print("BOB Chess Engine - Test Suite")
    print("=" * 70 + "\n")

    tests = [
        test_predict_starting_position,
        test_predict_tactical_position,
        test_predict_endgame_position,
        test_batch_predict,
        test_time_limit,
        test_checkmate_position,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
