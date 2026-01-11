"""Alpha-Beta search with iterative deepening and optimizations.

Implements:
- Alpha-beta pruning
- Iterative deepening
- Aspiration windows
- Quiescence search
- Late move reduction (LMR)
- Killer move heuristic
- History heuristic
- MVV/LVA move ordering
- Transposition table
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any
import time

from qratum_chess.core.position import Position, Move
from qratum_chess.core import Color, PieceType


@dataclass
class TranspositionEntry:
    """Entry in the transposition table."""
    hash_key: int
    depth: int
    value: float
    flag: str  # "exact", "lower", "upper"
    best_move: Move | None


@dataclass
class SearchStats:
    """Statistics from search execution."""
    nodes_searched: int = 0
    quiescence_nodes: int = 0
    beta_cutoffs: int = 0
    tt_hits: int = 0
    depth_reached: int = 0
    time_ms: float = 0.0
    nps: float = 0.0  # Nodes per second


@dataclass
class SearchConfig:
    """Configuration for alpha-beta search.
    
    Attributes:
        max_depth: Maximum search depth.
        quiescence_depth: Maximum quiescence search depth.
        aspiration_window: Initial aspiration window size.
        use_lmr: Enable late move reduction.
        use_null_move: Enable null move pruning.
        tt_size: Transposition table size (entries).
    """
    max_depth: int = 10
    quiescence_depth: int = 6
    aspiration_window: float = 0.5
    use_lmr: bool = True
    use_null_move: bool = True
    tt_size: int = 1_000_000


class AlphaBetaSearch:
    """Alpha-beta search engine with advanced optimizations.
    
    Features:
    - Iterative deepening with aspiration windows
    - Quiescence search for tactical stability
    - Transposition table for position caching
    - Killer moves and history heuristic for move ordering
    - Late move reduction for search efficiency
    """
    
    # Material values for MVV/LVA ordering
    PIECE_VALUES = {
        PieceType.PAWN: 100,
        PieceType.KNIGHT: 320,
        PieceType.BISHOP: 330,
        PieceType.ROOK: 500,
        PieceType.QUEEN: 900,
        PieceType.KING: 20000,
    }
    
    def __init__(
        self,
        evaluator: Callable[[Position], float] | None = None,
        config: SearchConfig | None = None
    ):
        """Initialize the search engine.
        
        Args:
            evaluator: Position evaluation function.
            config: Search configuration.
        """
        self.config = config or SearchConfig()
        self.evaluator = evaluator or self._default_evaluator
        
        # Transposition table
        self.tt: dict[int, TranspositionEntry] = {}
        
        # Killer moves (2 per ply)
        self.killer_moves: list[list[Move | None]] = [
            [None, None] for _ in range(100)
        ]
        
        # History heuristic
        self.history: dict[tuple[int, int], int] = {}
        
        # Search state
        self.stats = SearchStats()
        self.stop_search = False
        self.start_time = 0.0
        self.time_limit = float('inf')
    
    def search(
        self,
        position: Position,
        depth: int | None = None,
        time_limit_ms: float | None = None
    ) -> tuple[Move | None, float, SearchStats]:
        """Search for the best move.
        
        Args:
            position: Current position.
            depth: Search depth (uses config default if None).
            time_limit_ms: Time limit in milliseconds.
            
        Returns:
            Tuple of (best_move, evaluation, search_stats).
        """
        self.stats = SearchStats()
        self.stop_search = False
        self.start_time = time.perf_counter()
        self.time_limit = time_limit_ms / 1000.0 if time_limit_ms else float('inf')
        
        max_depth = depth or self.config.max_depth
        
        legal_moves = position.generate_legal_moves()
        if not legal_moves:
            return None, 0.0, self.stats
        
        if len(legal_moves) == 1:
            return legal_moves[0], 0.0, self.stats
        
        best_move = legal_moves[0]
        best_value = float('-inf')
        
        # Iterative deepening
        alpha = float('-inf')
        beta = float('inf')
        
        for d in range(1, max_depth + 1):
            if self.stop_search:
                break
            
            # Aspiration windows after depth 3
            if d > 3 and self.config.aspiration_window > 0:
                window = self.config.aspiration_window
                alpha = best_value - window
                beta = best_value + window
            else:
                alpha = float('-inf')
                beta = float('inf')
            
            value = self._alphabeta_root(position, d, alpha, beta, legal_moves)
            
            if self.stop_search:
                break
            
            # Re-search if outside aspiration window
            if value <= alpha or value >= beta:
                alpha = float('-inf')
                beta = float('inf')
                value = self._alphabeta_root(position, d, alpha, beta, legal_moves)
            
            if self.stop_search:
                break
            
            # Update best move from TT
            tt_entry = self.tt.get(position.hash())
            if tt_entry and tt_entry.best_move:
                best_move = tt_entry.best_move
                best_value = value
            
            self.stats.depth_reached = d
        
        # Calculate final stats
        elapsed = time.perf_counter() - self.start_time
        self.stats.time_ms = elapsed * 1000
        total_nodes = self.stats.nodes_searched + self.stats.quiescence_nodes
        self.stats.nps = total_nodes / elapsed if elapsed > 0 else 0
        
        return best_move, best_value, self.stats
    
    def _alphabeta_root(
        self,
        position: Position,
        depth: int,
        alpha: float,
        beta: float,
        legal_moves: list[Move]
    ) -> float:
        """Alpha-beta search at root node."""
        best_value = float('-inf')
        best_move = None
        
        # Order moves
        ordered_moves = self._order_moves(position, legal_moves, 0)
        
        for move in ordered_moves:
            if self._check_time():
                self.stop_search = True
                break
            
            new_pos = position.make_move(move)
            value = -self._alphabeta(new_pos, depth - 1, -beta, -alpha, 1)
            
            if value > best_value:
                best_value = value
                best_move = move
            
            if value > alpha:
                alpha = value
            
            if alpha >= beta:
                break
        
        # Store in TT
        self._store_tt(position, depth, best_value, "exact", best_move)
        
        return best_value
    
    def _alphabeta(
        self,
        position: Position,
        depth: int,
        alpha: float,
        beta: float,
        ply: int
    ) -> float:
        """Alpha-beta search with pruning."""
        self.stats.nodes_searched += 1
        
        # Check for timeout
        if self._check_time():
            self.stop_search = True
            return 0.0
        
        # Check for draw
        if position.is_draw():
            return 0.0
        
        # Probe transposition table
        tt_entry = self.tt.get(position.hash())
        if tt_entry and tt_entry.depth >= depth:
            self.stats.tt_hits += 1
            if tt_entry.flag == "exact":
                return tt_entry.value
            elif tt_entry.flag == "lower" and tt_entry.value > alpha:
                alpha = tt_entry.value
            elif tt_entry.flag == "upper" and tt_entry.value < beta:
                beta = tt_entry.value
            if alpha >= beta:
                return tt_entry.value
        
        # Quiescence search at depth 0
        if depth <= 0:
            return self._quiescence(position, alpha, beta, 0)
        
        # Check extension
        in_check = position.is_in_check()
        if in_check:
            depth += 1
        
        legal_moves = position.generate_legal_moves()
        
        # Checkmate or stalemate
        if not legal_moves:
            if in_check:
                return float('-inf') + ply  # Checkmate
            return 0.0  # Stalemate
        
        # Order moves
        ordered_moves = self._order_moves(position, legal_moves, ply, tt_entry)
        
        best_value = float('-inf')
        best_move = None
        flag = "upper"
        
        for i, move in enumerate(ordered_moves):
            new_pos = position.make_move(move)
            
            # Late move reduction
            if (
                self.config.use_lmr
                and depth >= 3
                and i >= 4
                and not in_check
                and not position.board.piece_at(move.to_sq)  # Not a capture
                and not move.promotion
            ):
                # Reduced depth search
                value = -self._alphabeta(new_pos, depth - 2, -alpha - 1, -alpha, ply + 1)
                
                # Re-search if promising
                if value > alpha:
                    value = -self._alphabeta(new_pos, depth - 1, -beta, -alpha, ply + 1)
            else:
                value = -self._alphabeta(new_pos, depth - 1, -beta, -alpha, ply + 1)
            
            if self.stop_search:
                return 0.0
            
            if value > best_value:
                best_value = value
                best_move = move
            
            if value > alpha:
                alpha = value
                flag = "exact"
            
            if alpha >= beta:
                self.stats.beta_cutoffs += 1
                
                # Update killer moves for the move that caused the beta cutoff
                # Only store quiet (non-capture) moves as killers
                if not position.board.piece_at(move.to_sq):  # Quiet move
                    self.killer_moves[ply][1] = self.killer_moves[ply][0]
                    self.killer_moves[ply][0] = move
                    
                    # Update history heuristic - reward moves that cause cutoffs
                    key = (move.from_sq, move.to_sq)
                    self.history[key] = self.history.get(key, 0) + depth * depth
                
                flag = "lower"
                break
        
        # Store in TT
        self._store_tt(position, depth, best_value, flag, best_move)
        
        return best_value
    
    def _quiescence(
        self,
        position: Position,
        alpha: float,
        beta: float,
        q_depth: int
    ) -> float:
        """Quiescence search for tactical stability."""
        self.stats.quiescence_nodes += 1
        
        # Stand pat
        stand_pat = self.evaluator(position)
        
        if stand_pat >= beta:
            return beta
        
        if alpha < stand_pat:
            alpha = stand_pat
        
        if q_depth >= self.config.quiescence_depth:
            return stand_pat
        
        # Generate captures only
        legal_moves = position.generate_legal_moves()
        captures = [
            m for m in legal_moves
            if position.board.piece_at(m.to_sq) or m.is_en_passant
        ]
        
        # Order captures by MVV/LVA
        captures.sort(
            key=lambda m: self._mvv_lva_score(position, m),
            reverse=True
        )
        
        for move in captures:
            new_pos = position.make_move(move)
            value = -self._quiescence(new_pos, -beta, -alpha, q_depth + 1)
            
            if value >= beta:
                return beta
            
            if value > alpha:
                alpha = value
        
        return alpha
    
    def _order_moves(
        self,
        position: Position,
        moves: list[Move],
        ply: int,
        tt_entry: TranspositionEntry | None = None
    ) -> list[Move]:
        """Order moves for better pruning efficiency."""
        scored_moves: list[tuple[float, Move]] = []
        
        for move in moves:
            score = 0.0
            
            # TT move first
            if tt_entry and tt_entry.best_move == move:
                score = 10000.0
            # Captures ordered by MVV/LVA
            elif position.board.piece_at(move.to_sq):
                score = 5000.0 + self._mvv_lva_score(position, move)
            # Promotions
            elif move.promotion:
                score = 4000.0 + (self.PIECE_VALUES.get(move.promotion, 0) if move.promotion else 0)
            # Killer moves
            elif move in self.killer_moves[ply]:
                idx = self.killer_moves[ply].index(move)
                score = 3000.0 - idx * 100
            # History heuristic
            else:
                key = (move.from_sq, move.to_sq)
                score = self.history.get(key, 0) * 0.001
            
            scored_moves.append((score, move))
        
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored_moves]
    
    def _mvv_lva_score(self, position: Position, move: Move) -> float:
        """Most Valuable Victim / Least Valuable Attacker score."""
        victim = position.board.piece_at(move.to_sq)
        if not victim:
            return 0.0
        
        _, victim_type = victim
        attacker = position.board.piece_at(move.from_sq)
        if not attacker:
            return 0.0
        
        _, attacker_type = attacker
        
        victim_value = self.PIECE_VALUES.get(victim_type, 0)
        attacker_value = self.PIECE_VALUES.get(attacker_type, 0)
        
        return victim_value - attacker_value * 0.01
    
    def _store_tt(
        self,
        position: Position,
        depth: int,
        value: float,
        flag: str,
        best_move: Move | None
    ) -> None:
        """Store position in transposition table."""
        hash_key = position.hash()
        
        # Replace if deeper or same depth
        existing = self.tt.get(hash_key)
        if existing is None or existing.depth <= depth:
            self.tt[hash_key] = TranspositionEntry(
                hash_key=hash_key,
                depth=depth,
                value=value,
                flag=flag,
                best_move=best_move,
            )
            
            # Limit TT size
            if len(self.tt) > self.config.tt_size:
                # Simple eviction: remove oldest entries
                keys = list(self.tt.keys())
                for k in keys[:len(keys) // 4]:
                    del self.tt[k]
    
    def _check_time(self) -> bool:
        """Check if time limit exceeded."""
        if self.time_limit == float('inf'):
            return False
        return (time.perf_counter() - self.start_time) >= self.time_limit
    
    def _default_evaluator(self, position: Position) -> float:
        """Default position evaluator using material count."""
        value = 0.0
        
        for color in Color:
            sign = 1 if color == Color.WHITE else -1
            for pt in PieceType:
                pieces = int(position.board.pieces[color, pt])
                count = bin(pieces).count('1')
                value += sign * count * self.PIECE_VALUES.get(pt, 0)
        
        # Adjust for side to move
        if position.side_to_move == Color.BLACK:
            value = -value
        
        return value / 100.0  # Normalize to ~[-10, 10] range
    
    def clear_tables(self) -> None:
        """Clear transposition table and history."""
        self.tt.clear()
        self.history.clear()
        self.killer_moves = [[None, None] for _ in range(100)]
