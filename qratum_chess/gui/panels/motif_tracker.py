"""Motif & Novelty Tracker Panel: Emergent pattern detection and highlighting.

Features:
- Highlight emergent tactical and strategic motifs during play
- Novelty density graphs and cross-cortex motif correlation
- Historical motif heatmaps showing frequency and novelty across games
- Pattern discovery timeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from qratum_chess.core.position import Position


class MotifType(Enum):
    """Types of chess motifs."""

    # Tactical motifs
    FORK = "fork"
    PIN = "pin"
    SKEWER = "skewer"
    DISCOVERY = "discovery"
    DEFLECTION = "deflection"
    DECOY = "decoy"
    OVERLOADING = "overloading"
    X_RAY = "x_ray"
    ZWISCHENZUG = "zwischenzug"

    # Strategic motifs
    OUTPOST = "outpost"
    WEAK_SQUARE = "weak_square"
    OPEN_FILE = "open_file"
    BACKWARD_PAWN = "backward_pawn"
    ISOLATED_PAWN = "isolated_pawn"
    PASSED_PAWN = "passed_pawn"
    PAWN_CHAIN = "pawn_chain"
    MINORITY_ATTACK = "minority_attack"
    SPACE_ADVANTAGE = "space_advantage"

    # Conceptual patterns
    KINGSIDE_ATTACK = "kingside_attack"
    QUEENSIDE_ATTACK = "queenside_attack"
    CENTRAL_CONTROL = "central_control"
    PROPHYLAXIS = "prophylaxis"
    PIECE_COORDINATION = "piece_coordination"

    # Novel/emergent
    NOVEL_PATTERN = "novel_pattern"
    EMERGENT_MOTIF = "emergent_motif"


@dataclass
class MotifInstance:
    """A single instance of a motif in a game."""

    motif_type: MotifType
    move_number: int
    squares: list[int]  # Squares involved in the motif
    pieces: list[str]  # Pieces involved
    score: float  # How well-executed or effective (0-1)
    novelty: float  # How novel this instance is (0-1)
    description: str = ""
    cortex_source: str = ""  # Which cortex detected it


@dataclass
class MotifStatistics:
    """Statistics for a specific motif type."""

    total_count: int = 0
    successful_count: int = 0
    average_score: float = 0.0
    average_novelty: float = 0.0
    frequency_by_phase: dict[str, int] = field(
        default_factory=lambda: {"opening": 0, "middlegame": 0, "endgame": 0}
    )


@dataclass
class NoveltyEvent:
    """A novelty event in the game."""

    move_number: int
    move_uci: str
    novelty_score: float
    description: str
    related_motifs: list[MotifType] = field(default_factory=list)


@dataclass
class MotifTrackerState:
    """Complete state of the motif tracker panel."""

    # Current game motifs
    active_motifs: list[MotifInstance] = field(default_factory=list)
    motif_timeline: list[MotifInstance] = field(default_factory=list)

    # Novelty tracking
    novelty_events: list[NoveltyEvent] = field(default_factory=list)
    current_novelty_score: float = 0.0
    novelty_threshold: float = 0.3  # Above this is considered novel

    # Statistics
    motif_stats: dict[MotifType, MotifStatistics] = field(default_factory=dict)

    # Historical data (across games)
    historical_frequency: dict[MotifType, list[float]] = field(default_factory=dict)
    historical_novelty: list[float] = field(default_factory=list)

    # Display options
    show_tactical: bool = True
    show_strategic: bool = True
    show_conceptual: bool = True
    show_novel: bool = True
    highlight_active: bool = True
    show_timeline: bool = True
    show_heatmap: bool = True

    # Heatmap data
    motif_heatmap: np.ndarray = field(default_factory=lambda: np.zeros((8, 8)))


class MotifTracker:
    """Motif & Novelty Tracker Panel for detecting and visualizing chess patterns.

    This panel provides:
    - Real-time motif detection and highlighting
    - Novelty scoring and event tracking
    - Historical motif frequency analysis
    - Pattern correlation across cortices
    - Visual timeline of detected patterns
    """

    # Motif categories
    TACTICAL_MOTIFS = {
        MotifType.FORK,
        MotifType.PIN,
        MotifType.SKEWER,
        MotifType.DISCOVERY,
        MotifType.DEFLECTION,
        MotifType.DECOY,
        MotifType.OVERLOADING,
        MotifType.X_RAY,
        MotifType.ZWISCHENZUG,
    }

    STRATEGIC_MOTIFS = {
        MotifType.OUTPOST,
        MotifType.WEAK_SQUARE,
        MotifType.OPEN_FILE,
        MotifType.BACKWARD_PAWN,
        MotifType.ISOLATED_PAWN,
        MotifType.PASSED_PAWN,
        MotifType.PAWN_CHAIN,
        MotifType.MINORITY_ATTACK,
        MotifType.SPACE_ADVANTAGE,
    }

    CONCEPTUAL_MOTIFS = {
        MotifType.KINGSIDE_ATTACK,
        MotifType.QUEENSIDE_ATTACK,
        MotifType.CENTRAL_CONTROL,
        MotifType.PROPHYLAXIS,
        MotifType.PIECE_COORDINATION,
    }

    # Colors for different motif types
    MOTIF_COLORS = {
        "tactical": (1.0, 0.4, 0.4, 0.8),  # Red
        "strategic": (0.3, 0.7, 1.0, 0.8),  # Blue
        "conceptual": (0.7, 0.3, 1.0, 0.8),  # Purple
        "novel": (0.0, 1.0, 0.5, 0.9),  # Green
    }

    def __init__(self, width: int = 350, height: int = 500) -> None:
        """Initialize motif tracker panel.

        Args:
            width: Panel width in pixels
            height: Panel height in pixels
        """
        self.width = width
        self.height = height
        self.state = MotifTrackerState()

        # Initialize statistics for all motif types
        for motif_type in MotifType:
            self.state.motif_stats[motif_type] = MotifStatistics()

    def reset_game(self) -> None:
        """Reset tracker for a new game."""
        self.state.active_motifs.clear()
        self.state.motif_timeline.clear()
        self.state.novelty_events.clear()
        self.state.current_novelty_score = 0.0
        self.state.motif_heatmap = np.zeros((8, 8))

    def detect_motifs(self, position: Position, move_number: int) -> list[MotifInstance]:
        """Detect motifs in the current position.

        Args:
            position: Current chess position
            move_number: Current move number

        Returns:
            List of detected motif instances
        """
        detected = []

        # Detect tactical motifs
        detected.extend(self._detect_tactical_motifs(position, move_number))

        # Detect strategic motifs
        detected.extend(self._detect_strategic_motifs(position, move_number))

        # Detect conceptual patterns
        detected.extend(self._detect_conceptual_motifs(position, move_number))

        # Update active motifs
        self.state.active_motifs = detected

        # Add to timeline
        for motif in detected:
            self.state.motif_timeline.append(motif)

        # Update heatmap
        self._update_heatmap(detected)

        # Update statistics
        self._update_statistics(detected)

        return detected

    def _detect_tactical_motifs(self, position: Position, move_number: int) -> list[MotifInstance]:
        """Detect tactical motifs in position.

        Args:
            position: Chess position
            move_number: Current move number

        Returns:
            List of tactical motif instances
        """
        detected = []

        # Detect forks
        fork_data = self._detect_forks(position)
        if fork_data:
            detected.append(
                MotifInstance(
                    motif_type=MotifType.FORK,
                    move_number=move_number,
                    squares=fork_data["squares"],
                    pieces=fork_data["pieces"],
                    score=fork_data["score"],
                    novelty=self._compute_novelty(MotifType.FORK),
                    description=f"Fork: {fork_data['description']}",
                    cortex_source="tactical",
                )
            )

        # Detect pins
        pin_data = self._detect_pins(position)
        if pin_data:
            detected.append(
                MotifInstance(
                    motif_type=MotifType.PIN,
                    move_number=move_number,
                    squares=pin_data["squares"],
                    pieces=pin_data["pieces"],
                    score=pin_data["score"],
                    novelty=self._compute_novelty(MotifType.PIN),
                    description=f"Pin: {pin_data['description']}",
                    cortex_source="tactical",
                )
            )

        return detected

    def _detect_strategic_motifs(self, position: Position, move_number: int) -> list[MotifInstance]:
        """Detect strategic motifs in position.

        Args:
            position: Chess position
            move_number: Current move number

        Returns:
            List of strategic motif instances
        """
        detected = []

        # Detect passed pawns
        passed_pawns = self._detect_passed_pawns(position)
        if passed_pawns:
            detected.append(
                MotifInstance(
                    motif_type=MotifType.PASSED_PAWN,
                    move_number=move_number,
                    squares=passed_pawns["squares"],
                    pieces=["P"] * len(passed_pawns["squares"]),
                    score=passed_pawns["score"],
                    novelty=self._compute_novelty(MotifType.PASSED_PAWN),
                    description=f"Passed pawn on {self._squares_to_notation(passed_pawns['squares'])}",
                    cortex_source="strategic",
                )
            )

        # Detect open files
        open_files = self._detect_open_files(position)
        if open_files:
            detected.append(
                MotifInstance(
                    motif_type=MotifType.OPEN_FILE,
                    move_number=move_number,
                    squares=open_files["squares"],
                    pieces=[],
                    score=open_files["score"],
                    novelty=self._compute_novelty(MotifType.OPEN_FILE),
                    description=f"Open file: {open_files['description']}",
                    cortex_source="strategic",
                )
            )

        return detected

    def _detect_conceptual_motifs(
        self, position: Position, move_number: int
    ) -> list[MotifInstance]:
        """Detect conceptual patterns in position.

        Args:
            position: Chess position
            move_number: Current move number

        Returns:
            List of conceptual motif instances
        """
        detected = []

        # Detect kingside attack
        kingside = self._detect_kingside_attack(position)
        if kingside:
            detected.append(
                MotifInstance(
                    motif_type=MotifType.KINGSIDE_ATTACK,
                    move_number=move_number,
                    squares=kingside["squares"],
                    pieces=kingside["pieces"],
                    score=kingside["score"],
                    novelty=self._compute_novelty(MotifType.KINGSIDE_ATTACK),
                    description="Kingside attack brewing",
                    cortex_source="conceptual",
                )
            )

        return detected

    def _detect_forks(self, position: Position) -> dict[str, Any] | None:
        """Detect potential fork motifs."""
        # Simplified fork detection
        # In production, would analyze attack patterns
        return None

    def _detect_pins(self, position: Position) -> dict[str, Any] | None:
        """Detect pin motifs."""
        return None

    def _detect_passed_pawns(self, position: Position) -> dict[str, Any] | None:
        """Detect passed pawns."""
        passed_squares = []

        def _get_piece_symbol(piece) -> str:
            """Get piece symbol handling both string and object representations."""
            if piece is None:
                return ""
            if isinstance(piece, str):
                return piece
            # Handle piece object
            try:
                symbol = getattr(piece, "symbol", None)
                if callable(symbol):
                    return symbol()
                return str(piece)
            except (AttributeError, TypeError):
                return str(piece)

        def _is_white_piece(piece) -> bool:
            """Check if piece is white handling both representations."""
            if piece is None:
                return False
            if isinstance(piece, str):
                return piece.isupper()
            try:
                return getattr(piece, "color", True)  # True = white in python-chess
            except AttributeError:
                return True

        # Check each pawn
        for square in range(64):
            piece = position.board.piece_at(square)
            piece_sym = _get_piece_symbol(piece)
            if piece_sym and piece_sym.upper() == "P":
                file = square % 8
                rank = square // 8
                is_white = _is_white_piece(piece)

                # Check if passed (no opposing pawns ahead on same or adjacent files)
                is_passed = True
                for check_file in [max(0, file - 1), file, min(7, file + 1)]:
                    for check_rank in range(
                        rank + (1 if is_white else -8), 8 if is_white else -1, 1 if is_white else -1
                    ):
                        check_sq = check_rank * 8 + check_file
                        if 0 <= check_sq < 64:
                            check_piece = position.board.piece_at(check_sq)
                            check_sym = _get_piece_symbol(check_piece)
                            if check_sym:
                                check_is_white = _is_white_piece(check_piece)
                                is_opponent_pawn = check_sym.upper() == "P" and (
                                    is_white != check_is_white
                                )
                                if is_opponent_pawn:
                                    is_passed = False
                                    break
                    if not is_passed:
                        break

                if is_passed and ((is_white and rank >= 4) or (not is_white and rank <= 3)):
                    passed_squares.append(square)

        if passed_squares:
            return {
                "squares": passed_squares,
                "score": 0.5 + 0.1 * len(passed_squares),
            }
        return None

    def _detect_open_files(self, position: Position) -> dict[str, Any] | None:
        """Detect open files."""
        open_files = []

        for file in range(8):
            has_pawn = False
            for rank in range(8):
                square = rank * 8 + file
                piece = position.board.piece_at(square)
                if piece and piece.upper() == "P":
                    has_pawn = True
                    break

            if not has_pawn:
                open_files.append(file)

        if open_files:
            return {
                "squares": [file for file in open_files],  # File indices
                "score": 0.4 + 0.1 * len(open_files),
                "description": f"Files {', '.join(chr(ord('a') + f) for f in open_files)}",
            }
        return None

    def _detect_kingside_attack(self, position: Position) -> dict[str, Any] | None:
        """Detect kingside attack patterns."""
        # Simplified: count pieces on kingside
        kingside_pieces = 0
        kingside_squares = []

        for file in range(5, 8):  # f, g, h files
            for rank in range(8):
                square = rank * 8 + file
                piece = position.board.piece_at(square)
                if piece and piece.isupper():  # White pieces
                    if piece.upper() not in ["K", "P"]:
                        kingside_pieces += 1
                        kingside_squares.append(square)

        if kingside_pieces >= 3:
            return {
                "squares": kingside_squares,
                "pieces": [
                    position.board.piece_at(s)
                    for s in kingside_squares
                    if position.board.piece_at(s)
                ],
                "score": 0.3 + 0.15 * kingside_pieces,
            }
        return None

    def _compute_novelty(self, motif_type: MotifType) -> float:
        """Compute novelty score for a motif occurrence.

        Args:
            motif_type: Type of motif

        Returns:
            Novelty score (0-1)
        """
        stats = self.state.motif_stats.get(motif_type)
        if not stats or stats.total_count == 0:
            return 1.0  # First occurrence is highly novel

        # Novelty decreases with frequency
        base_novelty = 1.0 / (1 + np.log(stats.total_count + 1))

        # Add some randomness for variation
        return min(1.0, base_novelty + np.random.uniform(-0.1, 0.1))

    def _squares_to_notation(self, squares: list[int]) -> str:
        """Convert square indices to algebraic notation."""
        notations = []
        for sq in squares:
            file = chr(ord("a") + (sq % 8))
            rank = str((sq // 8) + 1)
            notations.append(f"{file}{rank}")
        return ", ".join(notations)

    def _update_heatmap(self, motifs: list[MotifInstance]) -> None:
        """Update motif heatmap with new detections."""
        for motif in motifs:
            for square in motif.squares:
                if 0 <= square < 64:
                    file = square % 8
                    rank = square // 8
                    self.state.motif_heatmap[rank, file] += motif.score

        # Normalize heatmap
        max_val = self.state.motif_heatmap.max()
        if max_val > 0:
            self.state.motif_heatmap /= max_val

    def _update_statistics(self, motifs: list[MotifInstance]) -> None:
        """Update motif statistics with new detections."""
        for motif in motifs:
            stats = self.state.motif_stats[motif.motif_type]
            stats.total_count += 1
            stats.average_score = (
                stats.average_score * (stats.total_count - 1) + motif.score
            ) / stats.total_count
            stats.average_novelty = (
                stats.average_novelty * (stats.total_count - 1) + motif.novelty
            ) / stats.total_count

    def add_novelty_event(
        self, move_number: int, move_uci: str, novelty_score: float, description: str = ""
    ) -> None:
        """Record a novelty event.

        Args:
            move_number: Move number
            move_uci: UCI move notation
            novelty_score: Novelty score (0-1)
            description: Event description
        """
        if novelty_score >= self.state.novelty_threshold:
            event = NoveltyEvent(
                move_number=move_number,
                move_uci=move_uci,
                novelty_score=novelty_score,
                description=description or f"Novel move at move {move_number}",
            )
            self.state.novelty_events.append(event)

        self.state.current_novelty_score = novelty_score
        self.state.historical_novelty.append(novelty_score)

    def get_motif_category(self, motif_type: MotifType) -> str:
        """Get category for a motif type."""
        if motif_type in self.TACTICAL_MOTIFS:
            return "tactical"
        elif motif_type in self.STRATEGIC_MOTIFS:
            return "strategic"
        elif motif_type in self.CONCEPTUAL_MOTIFS:
            return "conceptual"
        else:
            return "novel"

    def get_render_data(self) -> dict[str, Any]:
        """Get all data needed for rendering.

        Returns:
            Dictionary with panel state for visualization
        """
        # Filter active motifs by display settings
        visible_motifs = []
        for motif in self.state.active_motifs:
            category = self.get_motif_category(motif.motif_type)
            if (
                category == "tactical"
                and self.state.show_tactical
                or category == "strategic"
                and self.state.show_strategic
                or category == "conceptual"
                and self.state.show_conceptual
                or category == "novel"
                and self.state.show_novel
            ):
                visible_motifs.append(motif)

        # Convert motif instances to dictionaries
        active_motifs_data = [
            {
                "type": m.motif_type.value,
                "category": self.get_motif_category(m.motif_type),
                "move_number": m.move_number,
                "squares": m.squares,
                "pieces": m.pieces,
                "score": m.score,
                "novelty": m.novelty,
                "description": m.description,
                "cortex_source": m.cortex_source,
                "color": self.MOTIF_COLORS[self.get_motif_category(m.motif_type)],
            }
            for m in visible_motifs
        ]

        # Timeline data (last 20 motifs)
        timeline_data = [
            {
                "type": m.motif_type.value,
                "category": self.get_motif_category(m.motif_type),
                "move_number": m.move_number,
                "score": m.score,
                "novelty": m.novelty,
            }
            for m in self.state.motif_timeline[-20:]
        ]

        # Novelty events
        novelty_data = [
            {
                "move_number": e.move_number,
                "move": e.move_uci,
                "score": e.novelty_score,
                "description": e.description,
            }
            for e in self.state.novelty_events[-10:]
        ]

        # Statistics summary
        stats_summary = {}
        for motif_type, stats in self.state.motif_stats.items():
            if stats.total_count > 0:
                stats_summary[motif_type.value] = {
                    "count": stats.total_count,
                    "avg_score": stats.average_score,
                    "avg_novelty": stats.average_novelty,
                }

        return {
            "width": self.width,
            "height": self.height,
            "active_motifs": active_motifs_data,
            "timeline": timeline_data,
            "novelty_events": novelty_data,
            "current_novelty": self.state.current_novelty_score,
            "novelty_threshold": self.state.novelty_threshold,
            "heatmap": self.state.motif_heatmap.tolist(),
            "statistics": stats_summary,
            "display_options": {
                "show_tactical": self.state.show_tactical,
                "show_strategic": self.state.show_strategic,
                "show_conceptual": self.state.show_conceptual,
                "show_novel": self.state.show_novel,
                "highlight_active": self.state.highlight_active,
                "show_timeline": self.state.show_timeline,
                "show_heatmap": self.state.show_heatmap,
            },
            "motif_colors": self.MOTIF_COLORS,
            "total_motifs": len(self.state.motif_timeline),
            "total_novelty_events": len(self.state.novelty_events),
        }

    def to_json(self) -> str:
        """Serialize render data to JSON."""
        import json

        return json.dumps(self.get_render_data())
