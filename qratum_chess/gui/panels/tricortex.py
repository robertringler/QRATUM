"""Tri-Cortex Diagnostics Panel: Visualization of tri-modal cognitive outputs.

Features:
- Tactical Cortex: Heatmap of move evaluations and short-term tactical probabilities
- Strategic Cortex: Graph of long-horizon evaluation values
- Conceptual Cortex: Vector visualization of motif abstraction and pattern recognition
- Cortex contribution bars showing percentage influence per move
- Entropy/confidence overlay to indicate move uncertainty and novelty
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from qratum_chess.core.position import Position


class CortexType(Enum):
    """Types of cognitive cortices."""

    TACTICAL = "tactical"
    STRATEGIC = "strategic"
    CONCEPTUAL = "conceptual"


@dataclass
class HeatmapData:
    """Heatmap visualization data for a cortex."""

    values: np.ndarray  # 8x8 array
    min_value: float = 0.0
    max_value: float = 1.0
    colormap: str = "viridis"
    label: str = ""


@dataclass
class MoveEvaluation:
    """Evaluation data for a single move."""

    move_uci: str
    tactical_score: float
    strategic_score: float
    conceptual_score: float
    combined_score: float
    confidence: float
    novelty: float
    motifs: list[str] = field(default_factory=list)


@dataclass
class CortexContribution:
    """Contribution percentages from each cortex."""

    tactical: float = 0.33
    strategic: float = 0.33
    conceptual: float = 0.34

    def normalize(self) -> None:
        """Normalize contributions to sum to 1.0."""
        total = self.tactical + self.strategic + self.conceptual
        if total > 0:
            self.tactical /= total
            self.strategic /= total
            self.conceptual /= total


@dataclass
class EntropyMetrics:
    """Entropy and uncertainty metrics."""

    move_entropy: float = 0.0  # Uncertainty in move selection
    position_entropy: float = 0.0  # Position complexity
    confidence: float = 1.0  # Overall confidence
    novelty_score: float = 0.0  # How novel is the position


@dataclass
class TimeSeriesPoint:
    """Single point in a time series graph."""

    move_number: int
    value: float
    label: str = ""


@dataclass
class TriCortexPanelState:
    """Complete state of the tri-cortex diagnostics panel."""

    # Heatmaps for each cortex
    tactical_heatmap: HeatmapData | None = None
    strategic_heatmap: HeatmapData | None = None
    conceptual_heatmap: HeatmapData | None = None

    # Move evaluations
    move_evaluations: list[MoveEvaluation] = field(default_factory=list)
    top_moves: list[MoveEvaluation] = field(default_factory=list)

    # Cortex contributions
    contributions: CortexContribution = field(default_factory=CortexContribution)

    # Entropy and confidence
    entropy: EntropyMetrics = field(default_factory=EntropyMetrics)

    # Time series data
    tactical_history: list[TimeSeriesPoint] = field(default_factory=list)
    strategic_history: list[TimeSeriesPoint] = field(default_factory=list)
    conceptual_history: list[TimeSeriesPoint] = field(default_factory=list)

    # Active cortex for detailed view
    active_cortex: CortexType = CortexType.TACTICAL

    # Display options
    show_heatmap: bool = True
    show_contribution_bars: bool = True
    show_entropy_overlay: bool = True
    show_time_series: bool = True
    heatmap_opacity: float = 0.7


class TriCortexPanel:
    """Tri-Cortex Diagnostics Panel for visualizing cognitive outputs.

    This panel displays:
    - Tactical Cortex: Short-term move evaluations as heatmaps
    - Strategic Cortex: Long-horizon planning visualizations
    - Conceptual Cortex: Motif abstractions and pattern visualizations
    - Contribution bars showing influence of each cortex
    - Entropy overlays for uncertainty visualization
    """

    # Color schemes for each cortex
    CORTEX_COLORS = {
        CortexType.TACTICAL: {
            "primary": (1.0, 0.4, 0.4),  # Red
            "secondary": (1.0, 0.7, 0.3),  # Orange
            "colormap": "hot",
        },
        CortexType.STRATEGIC: {
            "primary": (0.3, 0.7, 1.0),  # Blue
            "secondary": (0.5, 0.9, 0.9),  # Cyan
            "colormap": "cool",
        },
        CortexType.CONCEPTUAL: {
            "primary": (0.7, 0.3, 1.0),  # Purple
            "secondary": (0.9, 0.5, 0.9),  # Pink
            "colormap": "plasma",
        },
    }

    def __init__(self, width: int = 400, height: int = 600) -> None:
        """Initialize tri-cortex diagnostics panel.

        Args:
            width: Panel width in pixels
            height: Panel height in pixels
        """
        self.width = width
        self.height = height
        self.state = TriCortexPanelState()

        # Initialize with empty heatmaps
        self._init_heatmaps()

    def _init_heatmaps(self) -> None:
        """Initialize empty heatmaps."""
        empty_map = np.zeros((8, 8), dtype=np.float32)

        self.state.tactical_heatmap = HeatmapData(
            values=empty_map.copy(),
            colormap="hot",
            label="Tactical Activity",
        )
        self.state.strategic_heatmap = HeatmapData(
            values=empty_map.copy(),
            colormap="cool",
            label="Strategic Value",
        )
        self.state.conceptual_heatmap = HeatmapData(
            values=empty_map.copy(),
            colormap="plasma",
            label="Conceptual Patterns",
        )

    def update_from_trimodal(self, diagnostics: dict[str, Any]) -> None:
        """Update panel state from TriModalCore diagnostics.

        Args:
            diagnostics: Dictionary from TriModalCore.evaluate()
        """
        # Update contributions
        self.state.contributions = CortexContribution(
            tactical=diagnostics.get("tactical_weight", 0.33),
            strategic=diagnostics.get("strategic_weight", 0.33),
            conceptual=diagnostics.get("conceptual_weight", 0.34),
        )

        # Update entropy metrics
        self.state.entropy = EntropyMetrics(
            move_entropy=diagnostics.get("entropy", 0.0),
            position_entropy=diagnostics.get("position_complexity", 0.0),
            confidence=diagnostics.get("confidence", 1.0),
            novelty_score=diagnostics.get("novelty_pressure", 0.0),
        )

        # Update heatmaps if available
        if "tactical_heatmap" in diagnostics:
            self.state.tactical_heatmap.values = np.array(diagnostics["tactical_heatmap"]).reshape(
                8, 8
            )

        if "strategic_heatmap" in diagnostics:
            self.state.strategic_heatmap.values = np.array(
                diagnostics["strategic_heatmap"]
            ).reshape(8, 8)

        if "conceptual_heatmap" in diagnostics:
            self.state.conceptual_heatmap.values = np.array(
                diagnostics["conceptual_heatmap"]
            ).reshape(8, 8)

    def update_move_evaluations(self, evaluations: list[dict[str, Any]]) -> None:
        """Update move evaluation list.

        Args:
            evaluations: List of move evaluation dictionaries
        """
        self.state.move_evaluations.clear()

        for eval_data in evaluations:
            move_eval = MoveEvaluation(
                move_uci=eval_data.get("move", ""),
                tactical_score=eval_data.get("tactical", 0.0),
                strategic_score=eval_data.get("strategic", 0.0),
                conceptual_score=eval_data.get("conceptual", 0.0),
                combined_score=eval_data.get("combined", 0.0),
                confidence=eval_data.get("confidence", 1.0),
                novelty=eval_data.get("novelty", 0.0),
                motifs=eval_data.get("motifs", []),
            )
            self.state.move_evaluations.append(move_eval)

        # Sort by combined score and get top moves
        sorted_moves = sorted(
            self.state.move_evaluations, key=lambda x: x.combined_score, reverse=True
        )
        self.state.top_moves = sorted_moves[:5]

    def update_tactical_heatmap(self, position: Position, evaluator: Any) -> None:
        """Update tactical heatmap from position analysis.

        Args:
            position: Current chess position
            evaluator: Tactical evaluator (NNUE-style)
        """
        heatmap = np.zeros((8, 8), dtype=np.float32)

        # Evaluate tactical activity per square
        for square in range(64):
            file = square % 8
            rank = square // 8

            # Get piece at square
            piece = position.board.piece_at(square)
            if piece:
                # Tactical value based on piece activity
                value = self._evaluate_tactical_square(position, square)
                heatmap[rank, file] = value

        self.state.tactical_heatmap.values = heatmap
        self.state.tactical_heatmap.min_value = float(heatmap.min())
        self.state.tactical_heatmap.max_value = float(heatmap.max())

    def _evaluate_tactical_square(self, position: Position, square: int) -> float:
        """Evaluate tactical activity at a square.

        Args:
            position: Chess position
            square: Square index (0-63)

        Returns:
            Tactical activity value (0-1)
        """
        # Get attacks to/from this square using position's bitboard methods
        # Note: Uses position.board which may have get_attacks_from/to or attackers/attacks methods
        try:
            # Try using bitboard attack methods if available
            if hasattr(position.board, "get_attacks_from"):
                attacks_from = bin(position.board.get_attacks_from(square)).count("1")
                attacks_to = bin(position.board.get_attacks_to(square)).count("1")
            else:
                # Fallback: estimate activity based on piece presence
                piece = position.board.piece_at(square)
                attacks_from = 4 if piece else 0  # Simplified estimate
                attacks_to = 2  # Central squares typically have more attacks
        except (AttributeError, TypeError):
            # Default fallback
            attacks_from = 0
            attacks_to = 0

        # Normalize
        activity = (attacks_from + attacks_to) / 16.0
        return min(1.0, activity)

    def update_strategic_heatmap(self, position: Position, evaluator: Any) -> None:
        """Update strategic heatmap from position analysis.

        Args:
            position: Current chess position
            evaluator: Strategic evaluator (RL policy-value network)
        """
        heatmap = np.zeros((8, 8), dtype=np.float32)

        # Evaluate strategic value per square
        for square in range(64):
            file = square % 8
            rank = square // 8

            # Strategic value based on piece influence and control
            value = self._evaluate_strategic_square(position, square)
            heatmap[rank, file] = value

        self.state.strategic_heatmap.values = heatmap
        self.state.strategic_heatmap.min_value = float(heatmap.min())
        self.state.strategic_heatmap.max_value = float(heatmap.max())

    def _evaluate_strategic_square(self, position: Position, square: int) -> float:
        """Evaluate strategic value at a square.

        Args:
            position: Chess position
            square: Square index (0-63)

        Returns:
            Strategic value (-1 to 1)
        """
        file = square % 8
        rank = square // 8

        # Center control is strategically important
        center_dist = abs(file - 3.5) + abs(rank - 3.5)
        center_value = 1.0 - (center_dist / 7.0)

        # Piece presence adds strategic value
        piece = position.board.piece_at(square)
        piece_value = 0.0
        if piece:
            # Handle both string and object piece representations
            try:
                if isinstance(piece, str):
                    is_white = piece.isupper()
                    piece_type = piece.upper()
                else:
                    # Assume piece object with color and piece_type attributes
                    is_white = getattr(piece, "color", True)  # True = white
                    piece_symbol = getattr(piece, "symbol", lambda: "P")
                    piece_type = (
                        piece_symbol().upper()
                        if callable(piece_symbol)
                        else str(piece_symbol).upper()
                    )

                piece_values = {"P": 0.1, "N": 0.3, "B": 0.3, "R": 0.5, "Q": 0.9, "K": 0.2}
                piece_value = piece_values.get(piece_type, 0.0)
                if not is_white:
                    piece_value = -piece_value
            except (AttributeError, TypeError):
                piece_value = 0.0

        return center_value * 0.3 + piece_value * 0.7

    def update_conceptual_heatmap(self, position: Position, patterns: dict[str, Any]) -> None:
        """Update conceptual heatmap from pattern analysis.

        Args:
            position: Current chess position
            patterns: Detected patterns from Conceptual Cortex
        """
        heatmap = np.zeros((8, 8), dtype=np.float32)

        # Mark squares involved in detected patterns
        for pattern_name, pattern_data in patterns.items():
            if "squares" in pattern_data:
                intensity = pattern_data.get("intensity", 0.5)
                for square in pattern_data["squares"]:
                    file = square % 8
                    rank = square // 8
                    heatmap[rank, file] = max(heatmap[rank, file], intensity)

        self.state.conceptual_heatmap.values = heatmap
        self.state.conceptual_heatmap.min_value = 0.0
        self.state.conceptual_heatmap.max_value = 1.0

    def add_time_series_point(
        self, move_number: int, tactical: float, strategic: float, conceptual: float
    ) -> None:
        """Add a point to the time series graphs.

        Args:
            move_number: Current move number
            tactical: Tactical evaluation
            strategic: Strategic evaluation
            conceptual: Conceptual evaluation
        """
        self.state.tactical_history.append(TimeSeriesPoint(move_number, tactical))
        self.state.strategic_history.append(TimeSeriesPoint(move_number, strategic))
        self.state.conceptual_history.append(TimeSeriesPoint(move_number, conceptual))

        # Keep only last 100 moves
        max_points = 100
        if len(self.state.tactical_history) > max_points:
            self.state.tactical_history = self.state.tactical_history[-max_points:]
            self.state.strategic_history = self.state.strategic_history[-max_points:]
            self.state.conceptual_history = self.state.conceptual_history[-max_points:]

    def set_active_cortex(self, cortex: CortexType) -> None:
        """Set the active cortex for detailed view.

        Args:
            cortex: Cortex type to activate
        """
        self.state.active_cortex = cortex

    def get_render_data(self) -> dict[str, Any]:
        """Get all data needed for rendering.

        Returns:
            Dictionary with panel state for visualization
        """
        # Convert numpy arrays to lists for JSON serialization
        tactical_heatmap = None
        strategic_heatmap = None
        conceptual_heatmap = None

        if self.state.tactical_heatmap:
            tactical_heatmap = {
                "values": self.state.tactical_heatmap.values.tolist(),
                "min": self.state.tactical_heatmap.min_value,
                "max": self.state.tactical_heatmap.max_value,
                "colormap": self.state.tactical_heatmap.colormap,
                "label": self.state.tactical_heatmap.label,
            }

        if self.state.strategic_heatmap:
            strategic_heatmap = {
                "values": self.state.strategic_heatmap.values.tolist(),
                "min": self.state.strategic_heatmap.min_value,
                "max": self.state.strategic_heatmap.max_value,
                "colormap": self.state.strategic_heatmap.colormap,
                "label": self.state.strategic_heatmap.label,
            }

        if self.state.conceptual_heatmap:
            conceptual_heatmap = {
                "values": self.state.conceptual_heatmap.values.tolist(),
                "min": self.state.conceptual_heatmap.min_value,
                "max": self.state.conceptual_heatmap.max_value,
                "colormap": self.state.conceptual_heatmap.colormap,
                "label": self.state.conceptual_heatmap.label,
            }

        # Convert move evaluations
        move_evals = [
            {
                "move": me.move_uci,
                "tactical": me.tactical_score,
                "strategic": me.strategic_score,
                "conceptual": me.conceptual_score,
                "combined": me.combined_score,
                "confidence": me.confidence,
                "novelty": me.novelty,
                "motifs": me.motifs,
            }
            for me in self.state.move_evaluations
        ]

        top_moves = [
            {
                "move": me.move_uci,
                "tactical": me.tactical_score,
                "strategic": me.strategic_score,
                "conceptual": me.conceptual_score,
                "combined": me.combined_score,
                "confidence": me.confidence,
                "novelty": me.novelty,
                "motifs": me.motifs,
            }
            for me in self.state.top_moves
        ]

        # Convert time series
        time_series = {
            "tactical": [
                {"move": p.move_number, "value": p.value} for p in self.state.tactical_history
            ],
            "strategic": [
                {"move": p.move_number, "value": p.value} for p in self.state.strategic_history
            ],
            "conceptual": [
                {"move": p.move_number, "value": p.value} for p in self.state.conceptual_history
            ],
        }

        return {
            "width": self.width,
            "height": self.height,
            "active_cortex": self.state.active_cortex.value,
            "heatmaps": {
                "tactical": tactical_heatmap,
                "strategic": strategic_heatmap,
                "conceptual": conceptual_heatmap,
            },
            "contributions": {
                "tactical": self.state.contributions.tactical,
                "strategic": self.state.contributions.strategic,
                "conceptual": self.state.contributions.conceptual,
            },
            "entropy": {
                "move_entropy": self.state.entropy.move_entropy,
                "position_entropy": self.state.entropy.position_entropy,
                "confidence": self.state.entropy.confidence,
                "novelty": self.state.entropy.novelty_score,
            },
            "move_evaluations": move_evals,
            "top_moves": top_moves,
            "time_series": time_series,
            "display_options": {
                "show_heatmap": self.state.show_heatmap,
                "show_contribution_bars": self.state.show_contribution_bars,
                "show_entropy_overlay": self.state.show_entropy_overlay,
                "show_time_series": self.state.show_time_series,
                "heatmap_opacity": self.state.heatmap_opacity,
            },
            "cortex_colors": {
                "tactical": self.CORTEX_COLORS[CortexType.TACTICAL],
                "strategic": self.CORTEX_COLORS[CortexType.STRATEGIC],
                "conceptual": self.CORTEX_COLORS[CortexType.CONCEPTUAL],
            },
        }

    def to_json(self) -> str:
        """Serialize render data to JSON."""
        return json.dumps(self.get_render_data())
