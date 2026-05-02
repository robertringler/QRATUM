"""AAS Canonical Kernel - Universal Decision Primitive.

This module implements the Asymmetric Adaptive Search (AAS) as a canonical
kernel with invariant interfaces that can be extended to any domain.

The kernel provides:
- state_entropy(state) -> float: Measure state uncertainty/information content
- branch_value(state, move) -> float: Evaluate branching decisions
- resource_allocator(entropy_gradient) -> depth_budget: Allocate compute resources
- multi_agent_split(state) -> [orthogonal_subspaces]: Decompose state for parallel search

These primitives form the foundation for chess-class dominance in any bounded domain.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

# Generic state and action types
State = TypeVar("State")
Action = TypeVar("Action")


@runtime_checkable
class StateProtocol(Protocol):
    """Protocol for state objects compatible with AAS kernel."""

    def get_legal_actions(self) -> list[Any]:
        """Return list of legal actions from this state."""
        ...

    def apply_action(self, action: Any) -> StateProtocol:
        """Apply action and return new state (immutable)."""
        ...


@dataclass
class EntropyGradient:
    """Entropy gradient measurement over time.

    Tracks entropy changes to inform resource allocation.
    """

    current: float = 0.0
    previous: float = 0.0
    gradient: float = 0.0
    history: list[float] = field(default_factory=list)
    max_history: int = 100

    def update(self, new_entropy: float) -> None:
        """Update gradient with new entropy measurement."""
        self.previous = self.current
        self.current = new_entropy
        self.gradient = self.current - self.previous

        self.history.append(new_entropy)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    @property
    def trend(self) -> float:
        """Calculate longer-term entropy trend."""
        if len(self.history) < 2:
            return 0.0
        # Simple linear regression slope
        n = len(self.history)
        x_mean = (n - 1) / 2
        y_mean = sum(self.history) / n
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(self.history))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        return numerator / denominator if denominator != 0 else 0.0


@dataclass
class OrthogonalSubspace:
    """Represents an orthogonal subspace for multi-agent decomposition.

    Each subspace can be searched independently by different agents,
    with results later fused.
    """

    subspace_id: str
    state_slice: Any  # Domain-specific state representation
    priority: float = 1.0
    estimated_complexity: float = 1.0
    dependencies: list[str] = field(default_factory=list)


@dataclass
class DepthBudget:
    """Compute resource allocation for search.

    Specifies how deep and wide to search based on entropy analysis.
    """

    base_depth: int = 10
    depth_extension: int = 0
    width_multiplier: float = 1.0
    time_allocation_ms: float = 1000.0
    node_budget: int = 1_000_000
    selective_depth_bonus: dict[str, int] = field(default_factory=dict)

    @property
    def effective_depth(self) -> int:
        """Total search depth with extensions."""
        return self.base_depth + self.depth_extension


@dataclass
class AASMetrics:
    """Metrics required by QRATUM directive.

    Every AAS operation must report these metrics:
    - OSR: Outcome Superiority Ratio
    - CEI: Compute Efficiency Index
    - SF: Sovereignty Factor (0-1, 1 = fully local computation)
    - HRD: Hallucination Risk Density (should be ~0 for deterministic search)
    """

    outcome_superiority_ratio: float = 0.0
    compute_efficiency_index: float = 0.0
    sovereignty_factor: float = 1.0  # Default 1.0 = fully sovereign
    hallucination_risk_density: float = 0.0  # Target ~0

    def is_valid(self) -> bool:
        """Check if module metrics are valid per QRATUM directive."""
        return (
            self.outcome_superiority_ratio >= 0.0
            and self.compute_efficiency_index >= 0.0
            and 0.0 <= self.sovereignty_factor <= 1.0
            and self.hallucination_risk_density >= 0.0
        )


class AASKernel(ABC, Generic[State, Action]):
    """Abstract AAS Canonical Kernel.

    Implements the invariant kernel interfaces for Asymmetric Adaptive Search.
    Domain-specific implementations inherit from this class.
    """

    def __init__(self) -> None:
        """Initialize kernel state."""
        self.entropy_gradient = EntropyGradient()
        self.metrics = AASMetrics()
        self._computation_count = 0
        self._external_calls = 0

    @abstractmethod
    def state_entropy(self, state: State) -> float:
        """Calculate entropy (uncertainty/information content) of a state.

        Higher entropy indicates more uncertainty in the state.
        Used for entropy-gradient directed search.

        Args:
            state: The state to measure.

        Returns:
            Entropy value >= 0. Higher = more uncertain.
        """
        ...

    @abstractmethod
    def branch_value(self, state: State, action: Action) -> float:
        """Evaluate the value of taking an action from a state.

        Used for non-uniform branching asymmetry - prioritizing
        promising branches over others.

        Args:
            state: Current state.
            action: Action to evaluate.

        Returns:
            Value estimate. Higher = more promising branch.
        """
        ...

    def resource_allocator(self, entropy_gradient: EntropyGradient) -> DepthBudget:
        """Allocate compute resources based on entropy gradient.

        Implements adaptive resource allocation:
        - High entropy gradient -> wider search
        - Low/negative gradient -> deeper search
        - Stable entropy -> balanced approach

        Args:
            entropy_gradient: Current entropy gradient state.

        Returns:
            Depth budget specifying search parameters.
        """
        budget = DepthBudget()

        gradient = entropy_gradient.gradient
        trend = entropy_gradient.trend
        current = entropy_gradient.current

        # Base depth adjustment based on current entropy
        if current < 1.0:
            # Low entropy = clear position, go deeper
            budget.depth_extension = 4
            budget.width_multiplier = 0.8
        elif current > 3.0:
            # High entropy = complex position, go wider
            budget.depth_extension = -2
            budget.width_multiplier = 1.5
        else:
            # Medium entropy = balanced
            budget.depth_extension = 0
            budget.width_multiplier = 1.0

        # Gradient-based adjustment
        if gradient > 0.5:
            # Entropy increasing rapidly - widen search
            budget.width_multiplier *= 1.2
            budget.time_allocation_ms *= 1.3
        elif gradient < -0.5:
            # Entropy decreasing - deepen search
            budget.depth_extension += 2

        # Trend-based adjustment
        if trend > 0.1:
            # Long-term entropy increase - allocate more resources
            budget.node_budget = int(budget.node_budget * 1.5)
        elif trend < -0.1:
            # Long-term entropy decrease - can be more aggressive
            budget.depth_extension += 1

        return budget

    @abstractmethod
    def multi_agent_split(self, state: State) -> list[OrthogonalSubspace]:
        """Decompose state into orthogonal subspaces for parallel search.

        Enables multi-agent search by identifying independent aspects
        of the position that can be analyzed in parallel.

        Args:
            state: State to decompose.

        Returns:
            List of orthogonal subspaces that can be searched independently.
        """
        ...

    def adaptive_heuristic_mutation(self, feedback: dict[str, float]) -> None:
        """Mutate heuristics at runtime based on feedback.

        Enables self-modifying behavior based on search results.

        Args:
            feedback: Dictionary of metric_name -> value pairs.
        """
        # Update metrics based on feedback
        if "osr" in feedback:
            self.metrics.outcome_superiority_ratio = feedback["osr"]
        if "cei" in feedback:
            self.metrics.compute_efficiency_index = feedback["cei"]

    def get_metrics(self) -> AASMetrics:
        """Return current metrics (required by QRATUM directive)."""
        # Update CEI based on computation
        if self._computation_count > 0:
            # CEI = outcome / (compute * external_dependence)
            external_factor = 1.0 + (self._external_calls / max(1, self._computation_count))
            self.metrics.compute_efficiency_index = (
                self.metrics.outcome_superiority_ratio / external_factor
            )
            # SF decreases with external calls
            self.metrics.sovereignty_factor = 1.0 / external_factor

        return self.metrics

    def _track_computation(self, external: bool = False) -> None:
        """Track computation for metrics."""
        self._computation_count += 1
        if external:
            self._external_calls += 1


# Import chess-specific types when available
try:
    from qratum_chess.core import Color, PieceType
    from qratum_chess.core.position import Move, Position

    _CHESS_AVAILABLE = True
except ImportError:
    _CHESS_AVAILABLE = False
    Position = Any  # type: ignore
    Move = Any  # type: ignore
    Color = None  # type: ignore
    PieceType = None  # type: ignore


# Chess board constants
BOARD_SIZE = 8
CENTER_MIN = 2  # d-file (files a=0 through h=7)
CENTER_MAX = 5  # e-file


class ChessAASKernel(AASKernel[Position, Move]):
    """Chess-specific AAS kernel implementation.

    Implements the canonical kernel interfaces for chess domain.
    """

    def __init__(self, evaluator=None) -> None:
        """Initialize chess kernel.

        Args:
            evaluator: Optional position evaluator function.
        """
        super().__init__()
        self.evaluator = evaluator

        # Branching heuristic weights (adaptive)
        self._capture_weight = 2.0
        self._check_weight = 1.5
        self._center_weight = 1.2
        self._development_weight = 1.3
        self._promotion_weight = 3.0

    def state_entropy(self, state: Position) -> float:
        """Calculate entropy of a chess position.

        Entropy is based on:
        - Number of legal moves
        - Evaluation spread across moves
        - King safety uncertainty
        - Material balance uncertainty

        Args:
            state: Chess position.

        Returns:
            Position entropy (higher = more uncertain).
        """
        self._track_computation()

        if not _CHESS_AVAILABLE:
            return 1.0

        legal_moves = state.generate_legal_moves()
        num_moves = len(legal_moves)

        if num_moves == 0:
            return 0.0  # Game over - no uncertainty
        if num_moves == 1:
            return 0.1  # Forced move - minimal uncertainty

        # Base entropy from move count
        base_entropy = math.log(num_moves + 1)

        # Evaluation spread entropy
        eval_entropy = 0.0
        if self.evaluator and num_moves > 1:
            evaluations = []
            for move in legal_moves[:20]:  # Limit for efficiency
                new_pos = state.make_move(move)
                val = -self.evaluator(new_pos)
                evaluations.append(val)

            if evaluations:
                # Softmax-based entropy
                min_val = min(evaluations)
                shifted = [e - min_val + 0.01 for e in evaluations]
                total = sum(shifted)
                probs = [s / total for s in shifted]
                eval_entropy = -sum(p * math.log(p + 1e-10) for p in probs)

        # Combine entropies
        total_entropy = 0.6 * base_entropy + 0.4 * eval_entropy

        # Update gradient tracker
        self.entropy_gradient.update(total_entropy)

        return total_entropy

    def branch_value(self, state: Position, action: Move) -> float:
        """Evaluate the value of a move for branching priority.

        Args:
            state: Current position.
            action: Move to evaluate.

        Returns:
            Branch value (higher = more promising).
        """
        self._track_computation()

        if not _CHESS_AVAILABLE:
            return 1.0

        value = 1.0

        # Capture bonus
        target_sq = action.to_square
        if state.board.piece_at(target_sq):
            value *= self._capture_weight

        # Check bonus
        new_pos = state.make_move(action)
        if new_pos.is_check():
            value *= self._check_weight

        # Center control bonus (squares d3-e6 / c3-f6)
        to_rank, to_file = target_sq // BOARD_SIZE, target_sq % BOARD_SIZE
        if CENTER_MIN <= to_rank <= CENTER_MAX and CENTER_MIN <= to_file <= CENTER_MAX:
            value *= self._center_weight

        # Promotion bonus
        if action.promotion:
            value *= self._promotion_weight

        return value

    def multi_agent_split(self, state: Position) -> list[OrthogonalSubspace]:
        """Split chess position into orthogonal subspaces.

        Chess positions can be decomposed into:
        - Kingside: King safety and kingside pieces
        - Queenside: Queenside pieces and expansion
        - Center: Central control and pawn structure
        - Tactical: Forcing moves and combinations

        Args:
            state: Chess position.

        Returns:
            List of orthogonal subspaces for parallel analysis.
        """
        self._track_computation()

        subspaces = []

        if not _CHESS_AVAILABLE:
            # Return single subspace if chess not available
            return [
                OrthogonalSubspace(
                    subspace_id="full_state",
                    state_slice=state,
                    priority=1.0,
                    estimated_complexity=1.0,
                )
            ]

        legal_moves = state.generate_legal_moves()

        # 1. Kingside subspace
        kingside_moves = [m for m in legal_moves if m.to_square % 8 >= 4]  # Files e-h
        if kingside_moves:
            subspaces.append(
                OrthogonalSubspace(
                    subspace_id="kingside",
                    state_slice={"moves": kingside_moves, "focus": "kingside"},
                    priority=1.2,  # King safety is important
                    estimated_complexity=len(kingside_moves) / max(1, len(legal_moves)),
                )
            )

        # 2. Queenside subspace
        queenside_moves = [m for m in legal_moves if m.to_square % 8 < 4]  # Files a-d
        if queenside_moves:
            subspaces.append(
                OrthogonalSubspace(
                    subspace_id="queenside",
                    state_slice={"moves": queenside_moves, "focus": "queenside"},
                    priority=1.0,
                    estimated_complexity=len(queenside_moves) / max(1, len(legal_moves)),
                )
            )

        # 3. Tactical subspace (captures and checks)
        tactical_moves = []
        for m in legal_moves:
            # Capture
            if state.board.piece_at(m.to_square):
                tactical_moves.append(m)
            else:
                # Check
                new_pos = state.make_move(m)
                if new_pos.is_check():
                    tactical_moves.append(m)

        if tactical_moves:
            subspaces.append(
                OrthogonalSubspace(
                    subspace_id="tactical",
                    state_slice={"moves": tactical_moves, "focus": "forcing"},
                    priority=1.5,  # Tactics are critical
                    estimated_complexity=0.5,  # Usually fewer moves to consider
                    dependencies=["kingside", "queenside"],  # Results affect both
                )
            )

        # 4. Positional subspace (remaining quiet moves)
        tactical_set = set(id(m) for m in tactical_moves)
        positional_moves = [m for m in legal_moves if id(m) not in tactical_set]
        if positional_moves:
            subspaces.append(
                OrthogonalSubspace(
                    subspace_id="positional",
                    state_slice={"moves": positional_moves, "focus": "strategic"},
                    priority=0.8,
                    estimated_complexity=len(positional_moves) / max(1, len(legal_moves)),
                )
            )

        return (
            subspaces
            if subspaces
            else [
                OrthogonalSubspace(
                    subspace_id="full_state",
                    state_slice=state,
                    priority=1.0,
                    estimated_complexity=1.0,
                )
            ]
        )

    def adaptive_heuristic_mutation(self, feedback: dict[str, float]) -> None:
        """Update heuristic weights based on search feedback.

        Args:
            feedback: Search result feedback including:
                - capture_accuracy: How often captures were best
                - check_accuracy: How often checks were best
                - center_accuracy: How often center moves were best
        """
        super().adaptive_heuristic_mutation(feedback)

        learning_rate = 0.1

        if "capture_accuracy" in feedback:
            accuracy = feedback["capture_accuracy"]
            if accuracy > 0.5:
                self._capture_weight *= 1 + learning_rate * (accuracy - 0.5)
            else:
                self._capture_weight *= 1 - learning_rate * (0.5 - accuracy)

        if "check_accuracy" in feedback:
            accuracy = feedback["check_accuracy"]
            if accuracy > 0.5:
                self._check_weight *= 1 + learning_rate * (accuracy - 0.5)
            else:
                self._check_weight *= 1 - learning_rate * (0.5 - accuracy)

        if "center_accuracy" in feedback:
            accuracy = feedback["center_accuracy"]
            if accuracy > 0.5:
                self._center_weight *= 1 + learning_rate * (accuracy - 0.5)
            else:
                self._center_weight *= 1 - learning_rate * (0.5 - accuracy)


# Factory function for domain-agnostic kernel creation
def create_aas_kernel(domain: str, **kwargs) -> AASKernel:
    """Create an AAS kernel for the specified domain.

    Args:
        domain: Domain identifier ("chess", etc.)
        **kwargs: Domain-specific configuration.

    Returns:
        Domain-specific AAS kernel instance.
    """
    if domain.lower() == "chess":
        return ChessAASKernel(**kwargs)
    else:
        raise ValueError(f"Unknown domain: {domain}. Available: chess")
