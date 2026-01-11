"""Adversarial gauntlet for QRATUM-Chess benchmarking.

Tests engine against ensemble adversaries:
- Stockfish-style alpha-beta clones
- NNUE mutation engines
- Lc0-like neural agents
- Human GM move distributions

Target performance:
- ≥ 75% winrate vs Stockfish-NNUE baseline
- ≥ 70% winrate vs Lc0-class nets
- ≥ 85% alignment with GM move corpus
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import random
import time


class AdversaryType(Enum):
    """Types of adversary engines."""
    ALPHABETA = "alphabeta"
    NNUE = "nnue"
    NEURAL = "neural"
    HUMAN_GM = "human_gm"
    KAGGLE = "kaggle"  # Kaggle leaderboard benchmark positions
    KAGGLE = "kaggle"


@dataclass
class GameResult:
    """Result of a single game."""
    white_engine: str
    black_engine: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    moves: list[str]
    opening: str
    termination: str  # "checkmate", "stalemate", "draw", "timeout", etc.
    ply_count: int
    time_white_ms: float
    time_black_ms: float
    
    @property
    def winner(self) -> str | None:
        """Get winner name or None for draw."""
        if self.result == "1-0":
            return self.white_engine
        elif self.result == "0-1":
            return self.black_engine
        return None


@dataclass
class GauntletReport:
    """Report from adversarial gauntlet."""
    engine_name: str
    games: list[GameResult] = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    draws: int = 0
    win_rate: float = 0.0
    vs_alphabeta_winrate: float = 0.0
    vs_nnue_winrate: float = 0.0
    vs_neural_winrate: float = 0.0
    vs_human_alignment: float = 0.0
    
    def finalize(self) -> None:
        """Calculate aggregate statistics."""
        self.wins = sum(1 for g in self.games if g.winner == self.engine_name)
        self.losses = sum(
            1 for g in self.games
            if g.winner and g.winner != self.engine_name
        )
        self.draws = sum(1 for g in self.games if g.winner is None)
        
        total = len(self.games)
        if total > 0:
            # Win rate includes draws as 0.5
            self.win_rate = (self.wins + 0.5 * self.draws) / total
    
    def passed(self) -> bool:
        """Check if gauntlet passed all thresholds."""
        return (
            self.vs_alphabeta_winrate >= 0.75 and
            self.vs_neural_winrate >= 0.70 and
            self.vs_human_alignment >= 0.85
        )


class AdversarialGauntlet:
    """Adversarial gauntlet for engine testing.
    
    Runs the engine against a diverse set of adversaries
    to ensure dominance across engine paradigms.
    """
    
    # Standard openings for testing
    OPENINGS = [
        ("e2e4", "Italian Game"),
        ("d2d4", "Queen's Pawn"),
        ("c2c4", "English Opening"),
        ("g1f3", "Reti Opening"),
        ("b2b3", "Larsen's Opening"),
    ]
    
    def __init__(self, engine_name: str = "QRATUM-Chess"):
        """Initialize the gauntlet.
        
        Args:
            engine_name: Name of the engine being tested.
        """
        self.engine_name = engine_name
        self.adversaries: dict[AdversaryType, Any] = {}
    
    def register_adversary(
        self,
        adversary_type: AdversaryType,
        adversary: Any
    ) -> None:
        """Register an adversary engine.
        
        Args:
            adversary_type: Type of adversary.
            adversary: Adversary engine instance.
        """
        self.adversaries[adversary_type] = adversary
    
    def run_match(
        self,
        engine,
        adversary,
        adversary_name: str,
        num_games: int = 100,
        time_control_ms: float = 60000
    ) -> list[GameResult]:
        """Run a match between engine and adversary.
        
        Args:
            engine: Engine being tested.
            adversary: Adversary engine.
            adversary_name: Name of adversary.
            num_games: Number of games to play.
            time_control_ms: Time per side in milliseconds.
            
        Returns:
            List of game results.
        """
        from qratum_chess.core.position import Position
        
        results = []
        
        for game_num in range(num_games):
            # Alternate colors
            engine_white = (game_num % 2 == 0)
            
            # Select opening
            opening_move, opening_name = random.choice(self.OPENINGS)
            
            # Play game
            position = Position.starting()
            moves = []
            time_white = 0.0
            time_black = 0.0
            
            # Make opening move
            from qratum_chess.core.position import Move
            position = position.make_move(Move.from_uci(opening_move))
            moves.append(opening_move)
            
            # Play until game over
            max_ply = 500
            for ply in range(max_ply):
                if position.is_checkmate() or position.is_stalemate() or position.is_draw():
                    break
                
                is_white = (position.side_to_move.value == 0)
                current_engine = engine if (engine_white == is_white) else adversary
                
                start = time.perf_counter()
                try:
                    best_move, _, _ = current_engine.search(
                        position,
                        time_limit_ms=min(time_control_ms / 40, 1000)
                    )
                except Exception:
                    best_move = None
                elapsed = (time.perf_counter() - start) * 1000
                
                if is_white:
                    time_white += elapsed
                else:
                    time_black += elapsed
                
                if not best_move:
                    # Engine failed, opponent wins
                    break
                
                moves.append(best_move.to_uci())
                position = position.make_move(best_move)
            
            # Determine result
            if position.is_checkmate():
                if position.side_to_move.value == 0:  # White to move, Black won
                    result = "0-1"
                else:
                    result = "1-0"
                termination = "checkmate"
            elif position.is_stalemate():
                result = "1/2-1/2"
                termination = "stalemate"
            elif position.is_draw():
                result = "1/2-1/2"
                termination = "draw"
            else:
                result = "1/2-1/2"
                termination = "adjudication"
            
            game_result = GameResult(
                white_engine=self.engine_name if engine_white else adversary_name,
                black_engine=adversary_name if engine_white else self.engine_name,
                result=result,
                moves=moves,
                opening=opening_name,
                termination=termination,
                ply_count=len(moves),
                time_white_ms=time_white,
                time_black_ms=time_black,
            )
            results.append(game_result)
        
        return results
    
    def run_full_gauntlet(
        self,
        engine,
        games_per_adversary: int = 100
    ) -> GauntletReport:
        """Run full adversarial gauntlet.
        
        Args:
            engine: Engine being tested.
            games_per_adversary: Number of games per adversary type.
            
        Returns:
            Complete gauntlet report.
        """
        report = GauntletReport(engine_name=self.engine_name)
        
        # Run against each registered adversary type
        for adv_type, adversary in self.adversaries.items():
            results = self.run_match(
                engine,
                adversary,
                f"{adv_type.value}_adversary",
                num_games=games_per_adversary
            )
            report.games.extend(results)
            
            # Calculate type-specific win rate
            type_wins = sum(1 for g in results if g.winner == self.engine_name)
            type_draws = sum(1 for g in results if g.winner is None)
            type_rate = (type_wins + 0.5 * type_draws) / len(results) if results else 0
            
            if adv_type == AdversaryType.ALPHABETA:
                report.vs_alphabeta_winrate = type_rate
            elif adv_type == AdversaryType.NNUE:
                report.vs_nnue_winrate = type_rate
            elif adv_type == AdversaryType.NEURAL:
                report.vs_neural_winrate = type_rate
        
        report.finalize()
        return report
    
    def run_human_alignment_test(
        self,
        engine,
        gm_games: list[tuple[str, list[str]]],  # List of (fen, moves)
        sample_positions: int = 1000
    ) -> float:
        """Test alignment with human GM moves.
        
        Args:
            engine: Engine to test.
            gm_games: List of GM games as (initial_fen, move_list).
            sample_positions: Number of positions to sample.
            
        Returns:
            Alignment rate (fraction of matching moves).
        """
        from qratum_chess.core.position import Position
        
        matches = 0
        total = 0
        
        for fen, moves in gm_games[:sample_positions]:
            position = Position.from_fen(fen)
            
            for gm_move in moves[:20]:  # First 20 moves
                try:
                    engine_move, _, _ = engine.search(position, depth=10)
                    if engine_move and engine_move.to_uci() == gm_move.lower():
                        matches += 1
                except Exception:
                    pass
                
                total += 1
                
                # Make GM move
                from qratum_chess.core.position import Move
                position = position.make_move(Move.from_uci(gm_move))
        
        return matches / total if total > 0 else 0.0
    
    def run_kaggle_benchmark(
        self,
        engine,
        kaggle_positions: list,  # List of KaggleBenchmarkPosition
        depth: int = 15,
        time_limit_ms: float = 5000.0
    ) -> tuple[float, list[dict]]:
        """Run engine against Kaggle benchmark positions.
        
        Args:
            engine: Engine to test.
            kaggle_positions: List of Kaggle benchmark positions.
            depth: Search depth.
            time_limit_ms: Time limit per position.
            
        Returns:
            Tuple of (match_rate, detailed_results).
        """
        from qratum_chess.core.position import Position
        import time
        
        matches = 0
        total = 0
        results = []
        
        for pos_data in kaggle_positions:
            position = pos_data.position
            expected_move = pos_data.expected_move
            
            if position is None:
                continue
            
            try:
                start = time.perf_counter()
                engine_move, eval_score, stats = engine.search(
                    position,
                    depth=depth,
                    time_limit_ms=time_limit_ms
                )
                elapsed = (time.perf_counter() - start) * 1000
                
                result = {
                    "position_id": pos_data.position_id,
                    "engine_move": engine_move.to_uci() if engine_move else "",
                    "expected_move": expected_move,
                    "evaluation": eval_score,
                    "time_ms": elapsed,
                    "nodes": stats.nodes_searched if stats else 0,
                }
                
                if expected_move and engine_move:
                    if engine_move.to_uci() == expected_move:
                        matches += 1
                        result["match"] = True
                    else:
                        result["match"] = False
                    total += 1
                
                results.append(result)
            
            except Exception as e:
                results.append({
                    "position_id": pos_data.position_id,
                    "error": str(e),
                })
        
        match_rate = matches / total if total > 0 else 0.0
        return match_rate, results
    def run_kaggle_benchmark_adversary(
        self,
        engine,
        leaderboard_file: str,
        depth: int = 15,
        max_positions: int | None = None
    ) -> float:
        """Run Kaggle benchmark as an adversary test.
        
        Tests engine against Kaggle benchmark positions and returns
        accuracy score as win rate approximation.
        
        Args:
            engine: Engine to test.
            leaderboard_file: Path to Kaggle leaderboard JSON file.
            depth: Search depth for engine.
            max_positions: Maximum positions to test.
            
        Returns:
            Accuracy score (move accuracy as win rate proxy).
        """
        try:
            from qratum_chess.benchmarks.kaggle_integration import KaggleLeaderboardLoader
            from qratum_chess.benchmarks.benchmark_kaggle import KaggleBenchmarkRunner
            
            # Load Kaggle data
            loader = KaggleLeaderboardLoader()
            leaderboard = loader.load_from_file(leaderboard_file)
            
            # Run benchmark
            runner = KaggleBenchmarkRunner()
            results = runner.run_benchmark(
                engine,
                leaderboard,
                depth=depth,
                max_positions=max_positions
            )
            
            # Generate summary
            summary = runner.generate_summary(results)
            
            # Return move accuracy as proxy for win rate
            return summary.move_accuracy
            
        except Exception as e:
            print(f"Warning: Kaggle benchmark adversary test failed: {e}")
            return 0.0
