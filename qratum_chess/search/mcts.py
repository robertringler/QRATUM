"""Monte Carlo Tree Search (MCTS) implementation.

Implements MCTS with neural network guidance for move selection.
Follows the AlphaZero-style UCB formula with policy prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time
from typing import Callable, Any

from qratum_chess.core.position import Position, Move
from qratum_chess.core import Color


@dataclass
class MCTSNode:
    """Node in the MCTS tree.
    
    Attributes:
        position: Chess position at this node.
        parent: Parent node.
        move: Move that led to this node.
        children: Child nodes.
        visits: Number of visits.
        value_sum: Sum of values from simulations.
        prior: Prior probability from policy network.
        is_terminal: Whether this is a terminal node.
    """
    position: Position
    parent: 'MCTSNode | None' = None
    move: Move | None = None
    children: dict[Move, 'MCTSNode'] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    is_terminal: bool = False
    terminal_value: float = 0.0
    
    @property
    def q_value(self) -> float:
        """Average value of this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits
    
    @property
    def is_expanded(self) -> bool:
        """Whether this node has been expanded."""
        return len(self.children) > 0 or self.is_terminal


@dataclass
class MCTSConfig:
    """Configuration for MCTS.
    
    Attributes:
        num_simulations: Number of simulations per search.
        c_puct: Exploration constant for UCB formula.
        dirichlet_alpha: Alpha parameter for Dirichlet noise.
        dirichlet_epsilon: Fraction of prior that is noise.
        temperature: Temperature for final move selection.
        temperature_threshold: Ply after which temperature drops to 0.
    """
    num_simulations: int = 800
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30


class MCTSSearch:
    """Monte Carlo Tree Search engine.
    
    Implements MCTS with neural network policy and value guidance.
    Uses UCB formula: Q + c_puct * P * sqrt(N_parent) / (1 + N)
    """
    
    def __init__(
        self,
        evaluator: Callable[[Position, list[Move]], tuple[dict[Move, float], float]] | None = None,
        config: MCTSConfig | None = None
    ):
        """Initialize MCTS search.
        
        Args:
            evaluator: Function that returns (policy_dict, value) for a position.
            config: MCTS configuration.
        """
        self.config = config or MCTSConfig()
        self.evaluator = evaluator or self._default_evaluator
        
        # Statistics
        self.simulations_run = 0
        self.time_ms = 0.0
    
    def search(
        self,
        position: Position,
        num_simulations: int | None = None,
        time_limit_ms: float | None = None,
        add_noise: bool = True
    ) -> tuple[Move | None, dict[Move, float], float]:
        """Run MCTS search from a position.
        
        Args:
            position: Root position.
            num_simulations: Number of simulations (overrides config).
            time_limit_ms: Time limit in milliseconds.
            add_noise: Whether to add Dirichlet noise at root.
            
        Returns:
            Tuple of (best_move, move_visits, root_value).
        """
        start_time = time.perf_counter()
        
        num_sims = num_simulations or self.config.num_simulations
        
        # Create root node
        root = MCTSNode(position=position)
        
        # Check for immediate terminal
        legal_moves = position.generate_legal_moves()
        if not legal_moves:
            return None, {}, 0.0
        
        if len(legal_moves) == 1:
            return legal_moves[0], {legal_moves[0]: 1.0}, 0.0
        
        # Expand root
        self._expand(root)
        
        # Add Dirichlet noise to root priors
        if add_noise and root.children:
            self._add_dirichlet_noise(root)
        
        # Run simulations
        for i in range(num_sims):
            # Check time limit
            if time_limit_ms is not None:
                elapsed = (time.perf_counter() - start_time) * 1000
                if elapsed >= time_limit_ms:
                    break
            
            # Selection
            node = self._select(root)
            
            # Expansion
            if not node.is_expanded and not node.is_terminal:
                self._expand(node)
                if node.children:
                    # Select first child for evaluation
                    node = next(iter(node.children.values()))
            
            # Evaluation
            if node.is_terminal:
                value = node.terminal_value
            else:
                _, value = self.evaluator(node.position, node.position.generate_legal_moves())
            
            # Backpropagation
            self._backpropagate(node, value)
        
        self.simulations_run = root.visits
        self.time_ms = (time.perf_counter() - start_time) * 1000
        
        # Get visit counts
        move_visits = {
            move: child.visits
            for move, child in root.children.items()
        }
        
        # Select best move based on visit count
        best_move = self._select_move(root)
        
        return best_move, move_visits, root.q_value
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCB formula."""
        while node.is_expanded and not node.is_terminal:
            node = self._select_child(node)
        return node
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select the child with highest UCB score."""
        best_score = float('-inf')
        best_child = None
        
        sqrt_parent_visits = math.sqrt(node.visits)
        
        for child in node.children.values():
            # UCB formula: Q + c_puct * P * sqrt(N_parent) / (1 + N)
            ucb = child.q_value + self.config.c_puct * child.prior * sqrt_parent_visits / (1 + child.visits)
            
            if ucb > best_score:
                best_score = ucb
                best_child = child
        
        return best_child if best_child else node
    
    def _expand(self, node: MCTSNode) -> None:
        """Expand a node by creating child nodes."""
        position = node.position
        legal_moves = position.generate_legal_moves()
        
        if not legal_moves:
            # Terminal node
            node.is_terminal = True
            if position.is_in_check():
                # Checkmate - value from perspective of side to move
                node.terminal_value = -1.0
            else:
                # Stalemate
                node.terminal_value = 0.0
            return
        
        if position.is_draw():
            node.is_terminal = True
            node.terminal_value = 0.0
            return
        
        # Get policy from evaluator
        policy, _ = self.evaluator(position, legal_moves)
        
        # Create child nodes
        for move in legal_moves:
            prior = policy.get(move, 1.0 / len(legal_moves))
            new_pos = position.make_move(move)
            child = MCTSNode(
                position=new_pos,
                parent=node,
                move=move,
                prior=prior,
            )
            node.children[move] = child
    
    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate value up the tree."""
        while node is not None:
            node.visits += 1
            # Negate value as we go up (minimax)
            node.value_sum += value
            value = -value
            node = node.parent
    
    def _add_dirichlet_noise(self, node: MCTSNode) -> None:
        """Add Dirichlet noise to root priors for exploration."""
        if not node.children:
            return
        
        alpha = self.config.dirichlet_alpha
        epsilon = self.config.dirichlet_epsilon
        
        # Generate Dirichlet noise
        noise = self._dirichlet(alpha, len(node.children))
        
        for i, child in enumerate(node.children.values()):
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]
    
    def _dirichlet(self, alpha: float, n: int) -> list[float]:
        """Generate Dirichlet distributed random vector."""
        if n <= 0:
            return []
        if alpha <= 0:
            # Invalid alpha, return uniform distribution
            return [1.0 / n] * n
        
        # Use gamma distribution for Dirichlet sampling
        samples = [random.gammavariate(alpha, 1) for _ in range(n)]
        total = sum(samples)
        if total == 0:
            # All samples are zero (rare), return uniform
            return [1.0 / n] * n
        return [s / total for s in samples]
    
    def _select_move(self, root: MCTSNode) -> Move | None:
        """Select final move from root based on visit counts."""
        if not root.children:
            return None
        
        # Get visit counts
        visits = [(move, child.visits) for move, child in root.children.items()]
        
        # Apply temperature
        temp = self.config.temperature
        if root.position.fullmove_number * 2 > self.config.temperature_threshold:
            temp = 0.0
        
        if temp == 0.0:
            # Greedy selection
            return max(visits, key=lambda x: x[1])[0]
        else:
            # Proportional selection
            visit_counts = [v for _, v in visits]
            total = sum(v ** (1.0 / temp) for v in visit_counts)
            if total == 0:
                return visits[0][0]
            
            probs = [(v ** (1.0 / temp)) / total for v in visit_counts]
            r = random.random()
            cumsum = 0.0
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    return visits[i][0]
            
            return visits[-1][0]
    
    def _default_evaluator(
        self,
        position: Position,
        legal_moves: list[Move]
    ) -> tuple[dict[Move, float], float]:
        """Default evaluator using uniform policy and material value."""
        # Uniform policy
        policy = {move: 1.0 / len(legal_moves) for move in legal_moves}
        
        # Simple material evaluation
        from qratum_chess.core import PieceType
        
        PIECE_VALUES = {
            PieceType.PAWN: 1,
            PieceType.KNIGHT: 3,
            PieceType.BISHOP: 3,
            PieceType.ROOK: 5,
            PieceType.QUEEN: 9,
            PieceType.KING: 0,
        }
        
        value = 0.0
        for color in Color:
            sign = 1 if color == Color.WHITE else -1
            for pt in PieceType:
                pieces = int(position.board.pieces[color, pt])
                count = bin(pieces).count('1')
                value += sign * count * PIECE_VALUES.get(pt, 0)
        
        # Normalize and adjust for side to move
        value = value / 30.0  # Normalize
        if position.side_to_move == Color.BLACK:
            value = -value
        
        # Clamp to [-1, 1]
        value = max(-1.0, min(1.0, value))
        
        return policy, value
    
    def get_pv(self, root: MCTSNode, length: int = 10) -> list[Move]:
        """Get principal variation from search tree.
        
        Args:
            root: Root node.
            length: Maximum PV length.
            
        Returns:
            List of moves in principal variation.
        """
        pv = []
        node = root
        
        for _ in range(length):
            if not node.children:
                break
            
            # Select most visited child
            best_child = max(node.children.values(), key=lambda c: c.visits)
            if best_child.move:
                pv.append(best_child.move)
            node = best_child
        
        return pv
