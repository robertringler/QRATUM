"""Board Panel: 2D/3D interactive chess board with move highlighting and PV lines.

Features:
- 2D/3D toggle with smooth camera transitions
- Trajectory overlays for PV lines and move previews
- Legal move highlighting and move input (click, drag & drop, algebraic)
- Evaluation bar reflecting current positional advantage
- Clock/timer display for competitive games
- GPU-accelerated rendering (≥120 FPS target)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from qratum_chess.core.position import Position


class RenderMode(Enum):
    """Board rendering mode."""

    MODE_2D = "2d"
    MODE_3D = "3d"


class BoardTheme(Enum):
    """Board visual theme."""

    CLASSIC = "classic"
    MODERN = "modern"
    DARK = "dark"
    QUANTUM = "quantum"
    NEON = "neon"


@dataclass
class CameraState:
    """3D camera state for board rendering."""

    position: tuple[float, float, float] = (0.0, 8.0, 8.0)
    target: tuple[float, float, float] = (3.5, 0.0, 3.5)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov: float = 45.0
    near: float = 0.1
    far: float = 100.0

    def rotate_around_target(self, angle_x: float, angle_y: float) -> None:
        """Rotate camera around target point."""
        px, py, pz = self.position
        tx, ty, tz = self.target

        # Translate to origin
        dx, dy, dz = px - tx, py - ty, pz - tz

        # Rotate around Y axis (horizontal)
        cos_y = math.cos(angle_y)
        sin_y = math.sin(angle_y)
        dx_new = dx * cos_y - dz * sin_y
        dz_new = dx * sin_y + dz * cos_y
        dx, dz = dx_new, dz_new

        # Rotate around X axis (vertical) - constrained
        dist_xz = math.sqrt(dx * dx + dz * dz)
        current_pitch = math.atan2(dy, dist_xz)
        new_pitch = max(0.1, min(1.4, current_pitch + angle_x))  # Clamp pitch

        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        dy = dist * math.sin(new_pitch)
        scale = math.cos(new_pitch) / max(0.001, math.cos(current_pitch))
        dx *= scale
        dz *= scale

        # Translate back
        self.position = (tx + dx, ty + dy, tz + dz)

    def zoom(self, factor: float) -> None:
        """Zoom camera in/out."""
        px, py, pz = self.position
        tx, ty, tz = self.target

        dx, dy, dz = px - tx, py - ty, pz - tz
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        new_dist = max(3.0, min(30.0, dist * factor))  # Clamp distance

        scale = new_dist / dist
        self.position = (tx + dx * scale, ty + dy * scale, tz + dz * scale)


@dataclass
class SquareHighlight:
    """Highlight state for a board square."""

    square: int  # 0-63
    color: tuple[float, float, float, float]  # RGBA
    style: str = "solid"  # solid, dashed, glow, pulse
    intensity: float = 1.0


@dataclass
class ArrowOverlay:
    """Arrow overlay for showing moves or PV lines."""

    from_square: int
    to_square: int
    color: tuple[float, float, float, float]
    width: float = 0.15
    style: str = "solid"  # solid, dashed, gradient
    label: str = ""


@dataclass
class EvaluationBar:
    """Evaluation bar state."""

    value: float = 0.0  # -1 to +1 (or centipawns)
    is_mate: bool = False
    mate_in: int = 0
    confidence: float = 1.0


@dataclass
class ClockState:
    """Chess clock state."""

    white_time: float = 600.0  # seconds
    black_time: float = 600.0
    white_active: bool = True
    increment: float = 0.0
    is_running: bool = False


@dataclass
class BoardPanelState:
    """Complete state of the board panel."""

    position_fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    render_mode: RenderMode = RenderMode.MODE_2D
    theme: BoardTheme = BoardTheme.MODERN
    camera: CameraState = field(default_factory=CameraState)
    highlights: list[SquareHighlight] = field(default_factory=list)
    arrows: list[ArrowOverlay] = field(default_factory=list)
    evaluation: EvaluationBar = field(default_factory=EvaluationBar)
    clock: ClockState = field(default_factory=ClockState)
    flipped: bool = False
    show_coordinates: bool = True
    show_legal_moves: bool = True
    show_last_move: bool = True
    animation_speed: float = 0.3  # seconds per move animation
    selected_square: int | None = None
    dragging_piece: int | None = None


class BoardPanel:
    """Interactive chess board panel with 2D/3D rendering.

    This panel provides:
    - 2D and 3D board visualization with smooth transitions
    - Move highlighting and legal move display
    - Principal Variation (PV) line visualization
    - Evaluation bar with confidence indication
    - Clock display for timed games
    - GPU-accelerated rendering support
    """

    # Piece Unicode symbols
    PIECE_SYMBOLS = {
        "K": "♔",
        "Q": "♕",
        "R": "♖",
        "B": "♗",
        "N": "♘",
        "P": "♙",
        "k": "♚",
        "q": "♛",
        "r": "♜",
        "b": "♝",
        "n": "♞",
        "p": "♟",
    }

    # Square colors for different themes
    THEMES = {
        BoardTheme.CLASSIC: {
            "light": (0.94, 0.85, 0.71, 1.0),
            "dark": (0.71, 0.53, 0.39, 1.0),
            "highlight": (0.93, 0.93, 0.5, 0.8),
            "last_move": (0.8, 0.8, 0.2, 0.5),
            "legal_move": (0.2, 0.8, 0.2, 0.4),
            "check": (1.0, 0.3, 0.3, 0.7),
        },
        BoardTheme.MODERN: {
            "light": (0.93, 0.93, 0.88, 1.0),
            "dark": (0.47, 0.59, 0.68, 1.0),
            "highlight": (0.0, 0.96, 1.0, 0.6),
            "last_move": (0.5, 0.8, 1.0, 0.5),
            "legal_move": (0.0, 1.0, 0.5, 0.4),
            "check": (1.0, 0.2, 0.2, 0.7),
        },
        BoardTheme.DARK: {
            "light": (0.35, 0.35, 0.40, 1.0),
            "dark": (0.18, 0.18, 0.22, 1.0),
            "highlight": (0.0, 0.8, 1.0, 0.6),
            "last_move": (0.4, 0.6, 1.0, 0.5),
            "legal_move": (0.0, 1.0, 0.4, 0.4),
            "check": (1.0, 0.3, 0.3, 0.7),
        },
        BoardTheme.QUANTUM: {
            "light": (0.1, 0.15, 0.25, 1.0),
            "dark": (0.05, 0.08, 0.15, 1.0),
            "highlight": (0.0, 0.96, 1.0, 0.8),
            "last_move": (0.48, 0.17, 0.75, 0.6),
            "legal_move": (0.0, 1.0, 0.53, 0.5),
            "check": (1.0, 0.0, 0.43, 0.8),
        },
        BoardTheme.NEON: {
            "light": (0.05, 0.05, 0.1, 1.0),
            "dark": (0.02, 0.02, 0.05, 1.0),
            "highlight": (1.0, 0.0, 1.0, 0.7),
            "last_move": (0.0, 1.0, 1.0, 0.5),
            "legal_move": (0.0, 1.0, 0.0, 0.5),
            "check": (1.0, 0.0, 0.5, 0.8),
        },
    }

    def __init__(
        self,
        width: int = 800,
        height: int = 800,
        on_move: Callable[[str], None] | None = None,
        on_square_click: Callable[[int], None] | None = None,
    ) -> None:
        """Initialize board panel.

        Args:
            width: Panel width in pixels
            height: Panel height in pixels
            on_move: Callback when a move is made (UCI format)
            on_square_click: Callback when a square is clicked
        """
        self.width = width
        self.height = height
        self.state = BoardPanelState()
        self.on_move = on_move
        self.on_square_click = on_square_click

        # Animation state
        self._animation_start_time: float = 0.0
        self._animating_piece: str | None = None
        self._animation_from: int | None = None
        self._animation_to: int | None = None
        self._animation_progress: float = 1.0

        # Position cache
        self._board_array: list[str] = [""] * 64
        self._update_board_from_fen()

    def _update_board_from_fen(self) -> None:
        """Update internal board array from FEN string."""
        fen_parts = self.state.position_fen.split()
        board_part = fen_parts[0]

        self._board_array = [""] * 64
        rank = 7
        file = 0

        for char in board_part:
            if char == "/":
                rank -= 1
                file = 0
            elif char.isdigit():
                file += int(char)
            else:
                square = rank * 8 + file
                self._board_array[square] = char
                file += 1

    def set_position(self, fen: str) -> None:
        """Set board position from FEN string."""
        self.state.position_fen = fen
        self._update_board_from_fen()
        self.state.highlights.clear()
        self.state.selected_square = None

    def set_position_from_position(self, position: Position) -> None:
        """Set board position from Position object."""
        self.set_position(position.to_fen())

    def animate_move(self, from_sq: int, to_sq: int, piece: str) -> None:
        """Start move animation."""
        self._animation_start_time = time.time()
        self._animating_piece = piece
        self._animation_from = from_sq
        self._animation_to = to_sq
        self._animation_progress = 0.0

    def highlight_move(
        self, from_sq: int, to_sq: int, color: tuple[float, float, float, float] | None = None
    ) -> None:
        """Add highlight for a move (from and to squares)."""
        if color is None:
            color = self.THEMES[self.state.theme]["last_move"]

        self.state.highlights.append(SquareHighlight(from_sq, color, style="solid"))
        self.state.highlights.append(SquareHighlight(to_sq, color, style="solid"))

    def highlight_legal_moves(self, squares: list[int]) -> None:
        """Highlight legal move destinations."""
        color = self.THEMES[self.state.theme]["legal_move"]
        for sq in squares:
            self.state.highlights.append(SquareHighlight(sq, color, style="glow"))

    def add_pv_arrows(
        self, moves: list[tuple[int, int]], base_color: tuple[float, float, float, float]
    ) -> None:
        """Add arrows for principal variation."""
        for i, (from_sq, to_sq) in enumerate(moves):
            # Fade color for deeper moves
            alpha = base_color[3] * (0.9**i)
            color = (base_color[0], base_color[1], base_color[2], alpha)
            width = max(0.08, 0.15 - i * 0.02)

            arrow = ArrowOverlay(
                from_square=from_sq,
                to_square=to_sq,
                color=color,
                width=width,
                style="gradient" if i == 0 else "dashed",
                label=f"{i+1}" if i < 3 else "",
            )
            self.state.arrows.append(arrow)

    def clear_overlays(self) -> None:
        """Clear all highlights and arrows."""
        self.state.highlights.clear()
        self.state.arrows.clear()

    def set_evaluation(
        self, value: float, is_mate: bool = False, mate_in: int = 0, confidence: float = 1.0
    ) -> None:
        """Update evaluation bar."""
        self.state.evaluation = EvaluationBar(
            value=value,
            is_mate=is_mate,
            mate_in=mate_in,
            confidence=confidence,
        )

    def toggle_render_mode(self) -> None:
        """Toggle between 2D and 3D rendering."""
        if self.state.render_mode == RenderMode.MODE_2D:
            self.state.render_mode = RenderMode.MODE_3D
        else:
            self.state.render_mode = RenderMode.MODE_2D

    def set_theme(self, theme: BoardTheme) -> None:
        """Set board visual theme."""
        self.state.theme = theme

    def flip_board(self) -> None:
        """Flip board orientation."""
        self.state.flipped = not self.state.flipped

    def square_to_coords(self, square: int) -> tuple[int, int]:
        """Convert square index to file/rank."""
        file = square % 8
        rank = square // 8
        if self.state.flipped:
            file = 7 - file
            rank = 7 - rank
        return file, rank

    def coords_to_square(self, file: int, rank: int) -> int:
        """Convert file/rank to square index."""
        if self.state.flipped:
            file = 7 - file
            rank = 7 - rank
        return rank * 8 + file

    def pixel_to_square(self, x: float, y: float) -> int | None:
        """Convert pixel coordinates to square index."""
        # Assuming board takes full panel width/height
        square_size = min(self.width, self.height) / 8
        margin_x = (self.width - square_size * 8) / 2
        margin_y = (self.height - square_size * 8) / 2

        file = int((x - margin_x) / square_size)
        rank = 7 - int((y - margin_y) / square_size)

        if 0 <= file < 8 and 0 <= rank < 8:
            return self.coords_to_square(file, rank)
        return None

    def handle_click(self, x: float, y: float) -> None:
        """Handle mouse click on board."""
        square = self.pixel_to_square(x, y)
        if square is None:
            self.state.selected_square = None
            self.state.highlights.clear()
            return

        if self.on_square_click:
            self.on_square_click(square)

        if self.state.selected_square is not None:
            # Try to make a move
            from_sq = self.state.selected_square
            to_sq = square

            if from_sq != to_sq and self.on_move:
                # Convert to UCI
                from_file = chr(ord("a") + (from_sq % 8))
                from_rank = str((from_sq // 8) + 1)
                to_file = chr(ord("a") + (to_sq % 8))
                to_rank = str((to_sq // 8) + 1)
                uci_move = f"{from_file}{from_rank}{to_file}{to_rank}"
                self.on_move(uci_move)

            self.state.selected_square = None
            self.state.highlights.clear()
        else:
            # Select piece
            piece = self._board_array[square]
            if piece:
                self.state.selected_square = square
                color = self.THEMES[self.state.theme]["highlight"]
                self.state.highlights = [SquareHighlight(square, color)]

    def get_render_data(self) -> dict[str, Any]:
        """Get all data needed for rendering.

        Returns:
            Dictionary with board state, pieces, highlights, arrows, etc.
        """
        # Update animation
        if self._animation_progress < 1.0:
            elapsed = time.time() - self._animation_start_time
            self._animation_progress = min(1.0, elapsed / self.state.animation_speed)

        # Get theme colors
        theme_colors = self.THEMES[self.state.theme]

        # Build squares data
        squares = []
        for square in range(64):
            file, rank = self.square_to_coords(square)
            is_light = (file + rank) % 2 == 1
            base_color = theme_colors["light"] if is_light else theme_colors["dark"]

            # Check for highlights
            highlight_color = None
            for hl in self.state.highlights:
                if hl.square == square:
                    highlight_color = hl.color
                    break

            # Get piece at square
            piece = self._board_array[square]

            # Check if this piece is animating
            is_animating = (
                self._animation_progress < 1.0
                and self._animation_from == square
                and self._animating_piece == piece
            )

            squares.append(
                {
                    "index": square,
                    "file": file,
                    "rank": rank,
                    "color": base_color,
                    "highlight": highlight_color,
                    "piece": piece if not is_animating else "",
                    "piece_symbol": self.PIECE_SYMBOLS.get(piece, "") if not is_animating else "",
                }
            )

        # Build arrows data
        arrows_data = []
        for arrow in self.state.arrows:
            from_file, from_rank = self.square_to_coords(arrow.from_square)
            to_file, to_rank = self.square_to_coords(arrow.to_square)
            arrows_data.append(
                {
                    "from": {"file": from_file, "rank": from_rank},
                    "to": {"file": to_file, "rank": to_rank},
                    "color": arrow.color,
                    "width": arrow.width,
                    "style": arrow.style,
                    "label": arrow.label,
                }
            )

        # Animation data
        animation_data = None
        if self._animation_progress < 1.0:
            from_file, from_rank = self.square_to_coords(self._animation_from or 0)
            to_file, to_rank = self.square_to_coords(self._animation_to or 0)

            # Interpolate position
            t = self._animation_progress
            t = t * t * (3 - 2 * t)  # Smooth step

            current_file = from_file + (to_file - from_file) * t
            current_rank = from_rank + (to_rank - from_rank) * t

            animation_data = {
                "piece": self._animating_piece,
                "piece_symbol": self.PIECE_SYMBOLS.get(self._animating_piece or "", ""),
                "file": current_file,
                "rank": current_rank,
                "progress": self._animation_progress,
            }

        return {
            "mode": self.state.render_mode.value,
            "theme": self.state.theme.value,
            "width": self.width,
            "height": self.height,
            "flipped": self.state.flipped,
            "show_coordinates": self.state.show_coordinates,
            "squares": squares,
            "arrows": arrows_data,
            "animation": animation_data,
            "evaluation": {
                "value": self.state.evaluation.value,
                "is_mate": self.state.evaluation.is_mate,
                "mate_in": self.state.evaluation.mate_in,
                "confidence": self.state.evaluation.confidence,
            },
            "clock": {
                "white_time": self.state.clock.white_time,
                "black_time": self.state.clock.black_time,
                "white_active": self.state.clock.white_active,
                "is_running": self.state.clock.is_running,
            },
            "camera": {
                "position": self.state.camera.position,
                "target": self.state.camera.target,
                "up": self.state.camera.up,
                "fov": self.state.camera.fov,
            },
            "selected_square": self.state.selected_square,
        }

    def to_json(self) -> str:
        """Serialize render data to JSON."""
        return json.dumps(self.get_render_data())
