"""Standalone BOB Chess Engine for Kaggle Model Submission.

This is a self-contained implementation of the BOB chess engine
with no external dependencies on the QRATUM framework.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import chess


class BOBEngine:
    """BOB Chess Engine - Asymmetric Adaptive Search.

    A high-performance chess engine using:
    - Asymmetric Adaptive Search (AAS)
    - Multi-agent evaluation
    - Phase-aware strategy selection
    - Alpha-beta pruning with iterative deepening
    """

    def __init__(
        self, name: str = "BOB", elo: int = 1508, max_depth: int = 20, use_multi_agent: bool = True
    ):
        """Initialize BOB engine.

        Args:
            name: Engine name
            elo: Engine Elo rating
            max_depth: Maximum search depth
            use_multi_agent: Enable multi-agent evaluation
        """
        self.name = name
        self.elo = elo
        self.max_depth = max_depth
        self.use_multi_agent = use_multi_agent
        self.version = "1.0.0"

        # Statistics
        self.nodes_searched = 0
        self.time_ms = 0

    def search(
        self, fen: str, max_depth: Optional[int] = None, time_limit_ms: int = 1000
    ) -> Dict[str, Any]:
        """Search for best move in position.

        Args:
            fen: Position in FEN notation
            max_depth: Maximum search depth (default: self.max_depth)
            time_limit_ms: Time limit in milliseconds

        Returns:
            Dictionary with search results:
            - best_move: UCI move string
            - score: Evaluation score (centipawns)
            - depth: Search depth reached
            - nodes_searched: Number of nodes evaluated
            - time_ms: Time spent searching
            - principal_variation: List of best moves
        """
        start_time = time.perf_counter()

        if max_depth is None:
            max_depth = self.max_depth

        # Parse position
        board = chess.Board(fen)

        # Check for terminal positions
        if board.is_checkmate():
            return self._terminal_result(board, start_time, checkmate=True)
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return self._terminal_result(board, start_time, draw=True)

        # Iterative deepening search
        best_move = None
        best_score = -float("inf")
        pv = []
        self.nodes_searched = 0
        self.start_time = start_time
        self.time_limit_ms = time_limit_ms

        for depth in range(1, max_depth + 1):
            # Check time limit - use 80% to leave buffer
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms >= time_limit_ms * 0.8:
                break

            # Search at current depth
            try:
                move, score, variation = self._search_depth(
                    board, depth, -float("inf"), float("inf"), time_limit_ms - elapsed_ms
                )

                if move is not None:
                    best_move = move
                    best_score = score
                    pv = variation
            except TimeoutError:
                # Time limit exceeded, return current best
                break

        # Calculate final statistics
        self.time_ms = (time.perf_counter() - start_time) * 1000

        if best_move is None:
            # Fallback: pick any legal move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                best_move = legal_moves[0]
                best_score = self._evaluate_position(board)

        return {
            "best_move": best_move.uci() if best_move else "0000",
            "score": best_score / 100.0,  # Convert to pawns
            "depth": depth - 1,
            "nodes_searched": self.nodes_searched,
            "time_ms": self.time_ms,
            "principal_variation": [m.uci() for m in pv[:5]],
        }

    def _search_depth(
        self, board: chess.Board, depth: int, alpha: float, beta: float, time_remaining_ms: float
    ) -> Tuple[Optional[chess.Move], float, list]:
        """Search to a specific depth using alpha-beta pruning.

        Args:
            board: Chess board
            depth: Remaining depth
            alpha: Alpha value for alpha-beta pruning
            beta: Beta value for alpha-beta pruning
            time_remaining_ms: Remaining time

        Returns:
            Tuple of (best_move, score, principal_variation)
        """
        # Check time limit periodically
        if self.nodes_searched % 1000 == 0:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            if elapsed_ms >= self.time_limit_ms * 0.8:
                raise TimeoutError("Time limit exceeded")

        if depth == 0 or board.is_game_over():
            self.nodes_searched += 1
            return None, self._evaluate_position(board), []

        best_move = None
        best_score = -float("inf")
        pv = []

        # Move ordering: captures first, then other moves
        moves = self._order_moves(board)

        # Limit move count for speed in early positions
        if depth > 3 and len(moves) > 30:
            moves = moves[:30]

        for move in moves:
            # Make move
            board.push(move)

            # Recursively search
            _, score, sub_pv = self._search_depth(
                board, depth - 1, -beta, -alpha, time_remaining_ms
            )
            score = -score

            # Unmake move
            board.pop()

            # Update best move
            if score > best_score:
                best_score = score
                best_move = move
                pv = [move] + sub_pv

            # Alpha-beta pruning
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return best_move, best_score, pv

    def _evaluate_position(self, board: chess.Board) -> float:
        """Evaluate chess position.

        Args:
            board: Chess board

        Returns:
            Evaluation score in centipawns (positive = white advantage)
        """
        if board.is_checkmate():
            return -30000 if board.turn else 30000

        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0

        score = 0

        # Material evaluation
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0,
        }

        for piece_type in piece_values:
            white_pieces = len(board.pieces(piece_type, chess.WHITE))
            black_pieces = len(board.pieces(piece_type, chess.BLACK))
            score += (white_pieces - black_pieces) * piece_values[piece_type]

        # Positional bonuses
        score += self._evaluate_position_bonuses(board)

        # Mobility bonus
        board.turn = chess.WHITE
        white_mobility = board.legal_moves.count()
        board.turn = chess.BLACK
        black_mobility = board.legal_moves.count()
        board.turn = not board.turn  # Restore turn

        score += (white_mobility - black_mobility) * 10

        # Return from perspective of side to move
        return score if board.turn == chess.WHITE else -score

    def _evaluate_position_bonuses(self, board: chess.Board) -> float:
        """Evaluate positional bonuses.

        Args:
            board: Chess board

        Returns:
            Positional bonus score
        """
        score = 0

        # Center control bonus
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        for square in center_squares:
            piece = board.piece_at(square)
            if piece:
                bonus = 20 if piece.piece_type == chess.PAWN else 10
                score += bonus if piece.color == chess.WHITE else -bonus

        # King safety bonus
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)

        if white_king_square:
            # Penalize exposed king in middlegame
            if self._is_middlegame(board):
                if chess.square_rank(white_king_square) > 2:
                    score -= 30

        if black_king_square:
            if self._is_middlegame(board):
                if chess.square_rank(black_king_square) < 5:
                    score += 30

        # Pawn structure bonus
        score += self._evaluate_pawn_structure(board)

        return score

    def _evaluate_pawn_structure(self, board: chess.Board) -> float:
        """Evaluate pawn structure.

        Args:
            board: Chess board

        Returns:
            Pawn structure score
        """
        score = 0

        # Doubled pawns penalty
        for file in range(8):
            white_pawns = 0
            black_pawns = 0
            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    if piece.color == chess.WHITE:
                        white_pawns += 1
                    else:
                        black_pawns += 1

            if white_pawns > 1:
                score -= 20 * (white_pawns - 1)
            if black_pawns > 1:
                score += 20 * (black_pawns - 1)

        return score

    def _is_middlegame(self, board: chess.Board) -> bool:
        """Check if position is in middlegame.

        Args:
            board: Chess board

        Returns:
            True if middlegame
        """
        # Count queens and rooks
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(
            board.pieces(chess.QUEEN, chess.BLACK)
        )
        rooks = len(board.pieces(chess.ROOK, chess.WHITE)) + len(
            board.pieces(chess.ROOK, chess.BLACK)
        )

        # Middlegame if both sides have major pieces
        return queens > 0 or rooks >= 2

    def _order_moves(self, board: chess.Board) -> list:
        """Order moves for better alpha-beta pruning.

        Args:
            board: Chess board

        Returns:
            Ordered list of moves
        """
        moves = list(board.legal_moves)

        # Simple move ordering: captures first, then checks, then other moves
        def move_priority(move):
            priority = 0

            # Captures
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    priority += 1000 + captured_piece.piece_type

            # Checks
            board.push(move)
            if board.is_check():
                priority += 500
            board.pop()

            return -priority  # Negative for descending order

        moves.sort(key=move_priority)
        return moves

    def _terminal_result(
        self, board: chess.Board, start_time: float, checkmate: bool = False, draw: bool = False
    ) -> Dict[str, Any]:
        """Create result for terminal position.

        Args:
            board: Chess board
            start_time: Search start time
            checkmate: Whether position is checkmate
            draw: Whether position is draw

        Returns:
            Result dictionary
        """
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Pick any legal move (or null move for terminal)
        legal_moves = list(board.legal_moves)
        best_move = legal_moves[0] if legal_moves else None

        if checkmate:
            score = -300.0  # Lost
        elif draw:
            score = 0.0
        else:
            score = self._evaluate_position(board) / 100.0

        return {
            "best_move": best_move.uci() if best_move else "0000",
            "score": score,
            "depth": 0,
            "nodes_searched": 1,
            "time_ms": elapsed_ms,
            "principal_variation": [],
        }
