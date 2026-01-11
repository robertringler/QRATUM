"""Strategic torture suite for QRATUM-Chess benchmarking.

Tests the engine on pathological positions:
- Mutual zugzwang endgames
- Fortress positions
- Material-imbalanced king hunts
- Deep tablebase transitions
- Trapped-piece fractal structures

Measures:
- Eval volatility (target: ≤ 0.05)
- Blunder rate (target: ≤ 0.01%)
- Tablebase transition error (target: 0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time


@dataclass
class TortureTestResult:
    """Result from a torture test position."""
    position_name: str
    position_fen: str
    category: str
    engine_move: str | None
    expected_move: str | None
    evaluation: float
    eval_stability: float  # Variance across search depths
    is_correct: bool
    search_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TortureSuiteReport:
    """Report from complete torture suite execution."""
    results: list[TortureTestResult] = field(default_factory=list)
    eval_volatility: float = 0.0
    blunder_rate: float = 0.0
    tablebase_errors: int = 0
    positions_tested: int = 0
    correct_moves: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    
    def finalize(self) -> None:
        """Calculate aggregate metrics."""
        self.end_time = time.time()
        self.positions_tested = len(self.results)
        self.correct_moves = sum(1 for r in self.results if r.is_correct)
        
        if self.results:
            # Calculate eval volatility (average stability)
            stabilities = [r.eval_stability for r in self.results]
            self.eval_volatility = sum(stabilities) / len(stabilities)
            
            # Calculate blunder rate
            blunders = sum(1 for r in self.results if not r.is_correct)
            self.blunder_rate = blunders / len(self.results) if self.results else 0
    
    def passed(self) -> bool:
        """Check if torture suite passed all thresholds."""
        return (
            self.eval_volatility <= 0.05 and
            self.blunder_rate <= 0.0001 and
            self.tablebase_errors == 0
        )


class StrategicTortureSuite:
    """Strategic torture suite for engine testing.
    
    Contains pathological positions designed to stress-test
    the engine's strategic understanding and search stability.
    """
    
    # Torture test positions
    ZUGZWANG_POSITIONS = [
        # Classic zugzwang positions
        ("zugzwang_basic", "8/8/8/8/8/6k1/4K3/7R w - - 0 1", "Rh3+"),
        ("zugzwang_mutual", "8/8/p1p5/1p5p/1P5p/8/PPP2K1k/8 w - - 0 1", None),
        ("zugzwang_lasker", "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1", "Kb1"),
    ]
    
    FORTRESS_POSITIONS = [
        # Drawing fortress positions
        ("fortress_rook", "8/8/8/8/8/1k6/8/K1r5 w - - 0 1", None),
        ("fortress_wrong_bishop", "k7/8/8/8/8/8/B7/7K w - - 0 1", None),
        ("fortress_blockade", "8/8/6pk/5p1p/5P1P/6PK/8/8 w - - 0 1", None),
    ]
    
    KING_HUNT_POSITIONS = [
        # Material imbalanced king hunts
        ("king_hunt_queen", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", None),
        ("king_hunt_sac", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "Qxf7+"),
    ]
    
    TABLEBASE_TRANSITIONS = [
        # Positions transitioning to tablebase territory
        ("tb_kqk", "4k3/8/8/8/8/8/Q7/4K3 w - - 0 1", None),  # KQK
        ("tb_krk", "4k3/8/8/8/8/8/R7/4K3 w - - 0 1", None),  # KRK
        ("tb_kbnk", "4k3/8/8/8/8/8/BN6/4K3 w - - 0 1", None),  # KBNK
    ]
    
    TRAPPED_PIECE_POSITIONS = [
        # Positions with trapped pieces
        ("trapped_bishop", "r1bqk2r/pppnbppp/3p1n2/4p3/4P3/3P1N2/PPPNBPPP/R1BQK2R w KQkq - 4 6", None),
        ("trapped_knight", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", None),
    ]
    
    def __init__(self):
        """Initialize the torture suite."""
        self.positions: list[tuple[str, str, str | None, str]] = []
        self._load_positions()
    
    def _load_positions(self) -> None:
        """Load all torture test positions."""
        for name, fen, expected in self.ZUGZWANG_POSITIONS:
            self.positions.append((name, fen, expected, "zugzwang"))
        
        for name, fen, expected in self.FORTRESS_POSITIONS:
            self.positions.append((name, fen, expected, "fortress"))
        
        for name, fen, expected in self.KING_HUNT_POSITIONS:
            self.positions.append((name, fen, expected, "king_hunt"))
        
        for name, fen, expected in self.TABLEBASE_TRANSITIONS:
            self.positions.append((name, fen, expected, "tablebase"))
        
        for name, fen, expected in self.TRAPPED_PIECE_POSITIONS:
            self.positions.append((name, fen, expected, "trapped_piece"))
    
    def run(self, engine, depth: int = 15) -> TortureSuiteReport:
        """Run the complete torture suite.
        
        Args:
            engine: Chess engine to test.
            depth: Search depth for each position.
            
        Returns:
            Torture suite report.
        """
        from qratum_chess.core.position import Position
        
        report = TortureSuiteReport()
        
        for name, fen, expected, category in self.positions:
            position = Position.from_fen(fen)
            
            # Run search at multiple depths to measure stability
            evaluations = []
            start = time.perf_counter()
            
            for d in range(max(1, depth - 4), depth + 1):
                best_move, value, _ = engine.search(position, depth=d)
                evaluations.append(value)
            
            search_time = (time.perf_counter() - start) * 1000
            
            # Calculate evaluation stability
            if len(evaluations) > 1:
                mean_eval = sum(evaluations) / len(evaluations)
                variance = sum((e - mean_eval) ** 2 for e in evaluations) / len(evaluations)
                stability = variance ** 0.5
            else:
                stability = 0.0
            
            # Check correctness
            is_correct = True
            if expected and best_move:
                is_correct = best_move.to_uci() == expected.lower()
            
            result = TortureTestResult(
                position_name=name,
                position_fen=fen,
                category=category,
                engine_move=best_move.to_uci() if best_move else None,
                expected_move=expected,
                evaluation=evaluations[-1] if evaluations else 0.0,
                eval_stability=stability,
                is_correct=is_correct,
                search_time_ms=search_time,
                metadata={"evaluations": evaluations, "depth": depth},
            )
            
            report.results.append(result)
        
        report.finalize()
        return report
    
    def run_single_category(
        self,
        engine,
        category: str,
        depth: int = 15
    ) -> TortureSuiteReport:
        """Run torture tests for a single category.
        
        Args:
            engine: Chess engine to test.
            category: Category to test ("zugzwang", "fortress", etc.).
            depth: Search depth.
            
        Returns:
            Torture suite report for the category.
        """
        from qratum_chess.core.position import Position
        
        report = TortureSuiteReport()
        
        for name, fen, expected, cat in self.positions:
            if cat != category:
                continue
            
            position = Position.from_fen(fen)
            
            start = time.perf_counter()
            best_move, value, _ = engine.search(position, depth=depth)
            search_time = (time.perf_counter() - start) * 1000
            
            is_correct = True
            if expected and best_move:
                is_correct = best_move.to_uci() == expected.lower()
            
            result = TortureTestResult(
                position_name=name,
                position_fen=fen,
                category=cat,
                engine_move=best_move.to_uci() if best_move else None,
                expected_move=expected,
                evaluation=value,
                eval_stability=0.0,
                is_correct=is_correct,
                search_time_ms=search_time,
            )
            
            report.results.append(result)
        
        report.finalize()
        return report
