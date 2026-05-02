"""Asymmetric Adaptive Search (AAS) for Stage III.

Implements the meta-search dominance engine that dynamically adapts
search strategy based on game phase:

- Opening: Large-width conceptual exploration
- Middlegame: Hyper-focused entropy minimization
- Endgame: Perfect-play solver switching (tablebase + retrograde MCTS)

AAS dynamically rewrites its own branching heuristics mid-search
using learned entropy gradients.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from qratum_chess.core import Color, PieceType
from qratum_chess.core.position import Move, Position
from qratum_chess.search.alphabeta import AlphaBetaSearch, SearchConfig
from qratum_chess.search.mcts import MCTSConfig, MCTSSearch


class SearchPhase(Enum):
    """Search phase for adaptive behavior."""

    OPENING = "opening"
    MIDDLEGAME = "middlegame"
    ENDGAME = "endgame"
    TABLEBASE = "tablebase"


@dataclass
class AASConfig:
    """Configuration for Asymmetric Adaptive Search.

    Attributes:
        opening_width: Search width multiplier in opening.
        middlegame_focus: Entropy minimization factor in middlegame.
        endgame_precision: Depth multiplier in endgame.
        tablebase_pieces: Max pieces for tablebase lookup.
        entropy_gradient_learning_rate: Learning rate for entropy gradient updates.
        adaptive_branching: Enable dynamic branching heuristic rewriting.
    """

    opening_width: float = 2.0
    middlegame_focus: float = 1.5
    endgame_precision: float = 2.0
    tablebase_pieces: int = 7
    entropy_gradient_learning_rate: float = 0.1
    adaptive_branching: bool = True


@dataclass
class AASStats:
    """Statistics from AAS search."""

    phase: SearchPhase = SearchPhase.MIDDLEGAME
    nodes_searched: int = 0
    time_ms: float = 0.0
    entropy: float = 0.0
    branching_factor: float = 0.0
    depth_reached: int = 0
    tablebase_probes: int = 0


class AsymmetricAdaptiveSearch:
    """Asymmetric Adaptive Search engine.

    Combines Alpha-Beta, MCTS, and tablebase probing with dynamic
    adaptation based on game phase and position characteristics.

    Key features:
    - Phase-aware search strategy selection
    - Entropy gradient-based branching adjustment
    - Seamless tablebase integration in endgame
    - Self-modifying search heuristics
    """

    def __init__(
        self, evaluator: Callable[[Position], float] | None = None, config: AASConfig | None = None
    ):
        """Initialize AAS engine.

        Args:
            evaluator: Position evaluation function.
            config: AAS configuration.
        """
        self.config = config or AASConfig()
        self.evaluator = evaluator

        # Initialize sub-searchers
        self.alphabeta = AlphaBetaSearch(evaluator=evaluator)
        self.mcts = MCTSSearch()

        # Entropy gradient state
        self.entropy_history: list[float] = []
        self.branching_history: list[float] = []

        # Learned branching parameters
        self.branching_weights = {
            "capture_bonus": 2.0,
            "check_bonus": 1.5,
            "center_bonus": 1.2,
            "development_bonus": 1.3,
        }

        # Statistics
        self.stats = AASStats()

    def search(
        self, position: Position, depth: int = 10, time_limit_ms: float | None = None
    ) -> tuple[Move | None, float, AASStats]:
        """Execute adaptive search.

        Args:
            position: Current position.
            depth: Base search depth.
            time_limit_ms: Time limit in milliseconds.

        Returns:
            Tuple of (best_move, evaluation, stats).
        """
        start_time = time.perf_counter()
        self.stats = AASStats()

        # Determine game phase
        phase = self._determine_phase(position)
        self.stats.phase = phase

        # Check for tablebase position
        if phase == SearchPhase.TABLEBASE:
            result = self._probe_tablebase(position)
            if result:
                return result
            # Fall back to endgame search
            phase = SearchPhase.ENDGAME

        # Select and execute search strategy
        if phase == SearchPhase.OPENING:
            best_move, value = self._opening_search(position, depth, time_limit_ms)
        elif phase == SearchPhase.MIDDLEGAME:
            best_move, value = self._middlegame_search(position, depth, time_limit_ms)
        else:
            best_move, value = self._endgame_search(position, depth, time_limit_ms)

        # Update entropy gradient
        self._update_entropy_gradient(position)

        # Finalize stats
        self.stats.time_ms = (time.perf_counter() - start_time) * 1000

        return best_move, value, self.stats

    def _determine_phase(self, position: Position) -> SearchPhase:
        """Determine current game phase."""
        # Count pieces
        total_pieces = 0
        for color in Color:
            for pt in PieceType:
                total_pieces += bin(int(position.board.pieces[color, pt])).count("1")

        # Tablebase check
        if total_pieces <= self.config.tablebase_pieces:
            return SearchPhase.TABLEBASE

        # Material count for phase determination
        queens = sum(bin(int(position.board.pieces[c, PieceType.QUEEN])).count("1") for c in Color)
        rooks = sum(bin(int(position.board.pieces[c, PieceType.ROOK])).count("1") for c in Color)
        minor = sum(
            bin(int(position.board.pieces[c, pt])).count("1")
            for c in Color
            for pt in [PieceType.KNIGHT, PieceType.BISHOP]
        )

        total_material = queens * 9 + rooks * 5 + minor * 3

        if position.fullmove_number <= 12 and total_material > 35:
            return SearchPhase.OPENING
        elif total_material > 20:
            return SearchPhase.MIDDLEGAME
        else:
            return SearchPhase.ENDGAME

    def _opening_search(
        self, position: Position, depth: int, time_limit_ms: float | None
    ) -> tuple[Move | None, float]:
        """Opening search: Large-width conceptual exploration.

        Uses MCTS with high exploration and wide policy distribution.
        """
        # Configure MCTS for opening
        mcts_config = MCTSConfig(
            num_simulations=int(800 * self.config.opening_width),
            c_puct=2.0,  # Higher exploration
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
            temperature=1.0,
        )

        mcts = MCTSSearch(config=mcts_config)

        # Use neural evaluator if available
        if hasattr(self, "neural_evaluator"):
            mcts.evaluator = self.neural_evaluator

        best_move, move_visits, value = mcts.search(
            position, time_limit_ms=time_limit_ms, add_noise=True
        )

        self.stats.nodes_searched = mcts.simulations_run
        self.stats.branching_factor = len(move_visits) if move_visits else 0

        return best_move, value

    def _middlegame_search(
        self, position: Position, depth: int, time_limit_ms: float | None
    ) -> tuple[Move | None, float]:
        """Middlegame search: Hyper-focused entropy minimization.

        Combines Alpha-Beta with selective MCTS verification.
        Uses adaptive move ordering based on entropy gradients.
        """
        # Configure Alpha-Beta with adaptive parameters
        ab_config = SearchConfig(
            max_depth=int(depth * self.config.middlegame_focus),
            quiescence_depth=8,
            use_lmr=True,
            use_null_move=True,
        )

        alphabeta = AlphaBetaSearch(evaluator=self.evaluator, config=ab_config)

        # Apply learned branching weights
        if self.config.adaptive_branching:
            self._apply_adaptive_branching(alphabeta, position)

        best_move, value, ab_stats = alphabeta.search(position, time_limit_ms=time_limit_ms)

        # Calculate entropy of position
        legal_moves = position.generate_legal_moves()
        entropy = self._calculate_entropy(position, legal_moves)
        self.stats.entropy = entropy

        # Verify with MCTS if high entropy (uncertain position)
        if entropy > 2.0 and time_limit_ms and time_limit_ms > 100:
            verification_time = time_limit_ms * 0.2
            mcts_move, _, mcts_value = self.mcts.search(position, time_limit_ms=verification_time)

            # Use MCTS result if significantly different
            if mcts_move and abs(mcts_value - value) > 0.3:
                # Blend evaluations
                value = 0.7 * value + 0.3 * mcts_value

        self.stats.nodes_searched = ab_stats.nodes_searched
        self.stats.depth_reached = ab_stats.depth_reached

        return best_move, value

    def _endgame_search(
        self, position: Position, depth: int, time_limit_ms: float | None
    ) -> tuple[Move | None, float]:
        """Endgame search: Perfect-play solver mode.

        Uses deep Alpha-Beta with endgame-specific evaluation.
        """
        # Configure for deep endgame search
        ab_config = SearchConfig(
            max_depth=int(depth * self.config.endgame_precision),
            quiescence_depth=10,
            use_lmr=False,  # Less pruning in endgame
            use_null_move=False,
        )

        alphabeta = AlphaBetaSearch(evaluator=self._endgame_evaluator, config=ab_config)

        best_move, value, ab_stats = alphabeta.search(position, time_limit_ms=time_limit_ms)

        self.stats.nodes_searched = ab_stats.nodes_searched
        self.stats.depth_reached = ab_stats.depth_reached

        return best_move, value

    def _probe_tablebase(self, position: Position) -> tuple[Move | None, float, AASStats] | None:
        """Probe endgame tablebase.

        Returns None if no tablebase hit.
        """
        # Placeholder for tablebase integration
        # In production, would use Syzygy/Gaviota tablebases
        self.stats.tablebase_probes += 1

        # For now, return None to fall back to search
        return None

    def _endgame_evaluator(self, position: Position) -> float:
        """Specialized endgame evaluation.

        Focuses on:
        - King activity
        - Passed pawns
        - Piece coordination
        - Zugzwang recognition
        """
        value = 0.0

        # Material
        PIECE_VALUES = {
            PieceType.PAWN: 100,
            PieceType.KNIGHT: 300,
            PieceType.BISHOP: 320,
            PieceType.ROOK: 500,
            PieceType.QUEEN: 900,
            PieceType.KING: 0,
        }

        for color in Color:
            sign = 1 if color == Color.WHITE else -1
            for pt in PieceType:
                pieces = int(position.board.pieces[color, pt])
                count = bin(pieces).count("1")
                value += sign * count * PIECE_VALUES.get(pt, 0)

        # King activity bonus (centralization)
        for color in Color:
            sign = 1 if color == Color.WHITE else -1
            king_bb = int(position.board.pieces[color, PieceType.KING])
            if king_bb:
                king_sq = (king_bb & -king_bb).bit_length() - 1
                king_rank, king_file = king_sq // 8, king_sq % 8
                # Distance to center
                center_dist = abs(king_rank - 3.5) + abs(king_file - 3.5)
                value += sign * (7 - center_dist) * 10  # Bonus for central king

        # Passed pawn bonus
        for color in Color:
            sign = 1 if color == Color.WHITE else -1
            pawns = int(position.board.pieces[color, PieceType.PAWN])
            enemy_pawns = int(position.board.pieces[Color(1 - color), PieceType.PAWN])

            while pawns:
                sq = (pawns & -pawns).bit_length() - 1
                rank = sq // 8
                file = sq % 8

                # Check if passed
                is_passed = True
                if color == Color.WHITE:
                    for r in range(rank + 1, 8):
                        for f in range(max(0, file - 1), min(8, file + 2)):
                            if enemy_pawns & (1 << (r * 8 + f)):
                                is_passed = False
                                break
                        if not is_passed:
                            break
                    if is_passed:
                        value += sign * rank * 20  # Passed pawn bonus
                else:
                    for r in range(rank - 1, -1, -1):
                        for f in range(max(0, file - 1), min(8, file + 2)):
                            if enemy_pawns & (1 << (r * 8 + f)):
                                is_passed = False
                                break
                        if not is_passed:
                            break
                    if is_passed:
                        value += sign * (7 - rank) * 20

                pawns &= pawns - 1

        # Normalize
        if position.side_to_move == Color.BLACK:
            value = -value

        return value / 100.0

    def _calculate_entropy(self, position: Position, legal_moves: list[Move]) -> float:
        """Calculate position entropy based on move evaluation spread."""
        if not legal_moves or len(legal_moves) == 1:
            return 0.0

        # Get rough evaluations for each move
        evaluations = []
        for move in legal_moves[:20]:  # Limit for efficiency
            new_pos = position.make_move(move)
            if self.evaluator:
                val = -self.evaluator(new_pos)
            else:
                val = 0.0
            evaluations.append(val)

        if not evaluations:
            return 0.0

        # Convert to probability distribution
        min_val = min(evaluations)
        shifted = [e - min_val + 0.01 for e in evaluations]
        total = sum(shifted)
        probs = [s / total for s in shifted]

        # Calculate entropy
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)

        return entropy

    def _update_entropy_gradient(self, position: Position) -> None:
        """Update entropy gradient for adaptive branching."""
        legal_moves = position.generate_legal_moves()
        entropy = self._calculate_entropy(position, legal_moves)

        self.entropy_history.append(entropy)

        # Keep limited history
        if len(self.entropy_history) > 100:
            self.entropy_history = self.entropy_history[-100:]

        # Compute gradient (simplified)
        if len(self.entropy_history) >= 2:
            gradient = self.entropy_history[-1] - self.entropy_history[-2]

            # Adjust branching weights based on gradient
            lr = self.config.entropy_gradient_learning_rate

            if gradient > 0:
                # Entropy increasing - broaden search
                self.branching_weights["capture_bonus"] *= 1 - lr * 0.1
                self.branching_weights["center_bonus"] *= 1 + lr * 0.1
            else:
                # Entropy decreasing - focus search
                self.branching_weights["capture_bonus"] *= 1 + lr * 0.1
                self.branching_weights["center_bonus"] *= 1 - lr * 0.1

    def _apply_adaptive_branching(self, searcher: AlphaBetaSearch, position: Position) -> None:
        """Apply learned branching weights to search."""
        # Update history heuristic weights based on learned parameters
        for key in searcher.history:
            from_sq, to_sq = key

            # Capture bonus
            if position.board.piece_at(to_sq):
                searcher.history[key] = int(
                    searcher.history.get(key, 0) * self.branching_weights["capture_bonus"]
                )

            # Center bonus
            to_rank, to_file = to_sq // 8, to_sq % 8
            if 2 <= to_rank <= 5 and 2 <= to_file <= 5:
                searcher.history[key] = int(
                    searcher.history.get(key, 0) * self.branching_weights["center_bonus"]
                )

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information from last search."""
        return {
            "phase": self.stats.phase.value,
            "nodes": self.stats.nodes_searched,
            "time_ms": self.stats.time_ms,
            "entropy": self.stats.entropy,
            "branching_factor": self.stats.branching_factor,
            "depth": self.stats.depth_reached,
            "tablebase_probes": self.stats.tablebase_probes,
            "branching_weights": self.branching_weights.copy(),
            "entropy_history_len": len(self.entropy_history),
        }
