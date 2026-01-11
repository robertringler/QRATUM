"""GUI Manager: Coordinates all GUI panels and provides unified interface.

The GUIManager integrates all panel components and provides:
- Unified state management
- Panel coordination
- Event routing
- Render data aggregation for web frontend
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable
import json
import time

from qratum_chess.gui.panels.board import BoardPanel, BoardTheme, RenderMode
from qratum_chess.gui.panels.tricortex import TriCortexPanel
from qratum_chess.gui.panels.search_tree import SearchTreePanel
from qratum_chess.gui.panels.motif_tracker import MotifTracker
from qratum_chess.gui.panels.quantum import QuantumPanel
from qratum_chess.gui.panels.anti_holographic import AntiHolographicPanel
from qratum_chess.gui.panels.telemetry import TelemetryPanel
from qratum_chess.gui.panels.control import ControlPanel, GameMode
from qratum_chess.gui.snapshot.manager import SnapshotManager
from qratum_chess.gui.snapshot.capture import SnapshotConfig

if TYPE_CHECKING:
    from qratum_chess.core.position import Position, Move


@dataclass
class GUIConfig:
    """GUI configuration."""
    # Panel sizes
    board_size: int = 600
    panel_width: int = 350
    panel_height: int = 500
    
    # Theme
    board_theme: BoardTheme = BoardTheme.QUANTUM
    dark_mode: bool = True
    
    # Features
    enable_quantum_panel: bool = False
    enable_snapshot_capture: bool = True
    snapshot_output_dir: str = "snapshots"
    
    # Performance
    target_fps: int = 60
    animation_speed: float = 0.3


class GUIManager:
    """Main GUI Manager for QRATUM-Chess visualization system.
    
    This class:
    - Creates and manages all GUI panel instances
    - Coordinates state updates across panels
    - Aggregates render data for web frontend
    - Handles snapshot capture integration
    - Routes events between panels
    
    Usage:
        gui = GUIManager()
        gui.new_game()
        gui.update_position(position)
        render_data = gui.get_full_render_data()
    """
    
    def __init__(self, config: GUIConfig | None = None) -> None:
        """Initialize GUI manager.
        
        Args:
            config: GUI configuration
        """
        self.config = config or GUIConfig()
        
        # Create panels
        self.board = BoardPanel(
            width=self.config.board_size,
            height=self.config.board_size,
            on_move=self._on_move_from_board,
        )
        self.board.set_theme(self.config.board_theme)
        
        self.tricortex = TriCortexPanel(
            width=self.config.panel_width,
            height=self.config.panel_height,
        )
        
        self.search_tree = SearchTreePanel(
            width=self.config.panel_width + 100,
            height=self.config.panel_height,
        )
        
        self.motif_tracker = MotifTracker(
            width=self.config.panel_width,
            height=self.config.panel_height,
        )
        
        self.quantum = QuantumPanel(
            width=self.config.panel_width,
            height=self.config.panel_height - 100,
        )
        self.quantum.set_enabled(self.config.enable_quantum_panel)
        
        self.anti_holographic = AntiHolographicPanel(
            width=self.config.panel_width,
            height=self.config.panel_height - 100,
        )
        
        self.telemetry = TelemetryPanel(
            width=self.config.panel_width,
            height=self.config.panel_height,
        )
        
        self.control = ControlPanel(
            width=self.config.panel_width,
            height=self.config.panel_height + 100,
            on_new_game=self.new_game,
            on_stop_game=self.stop_game,
            on_settings_change=self._on_settings_change,
        )
        
        # Snapshot manager
        self.snapshot_manager: SnapshotManager | None = None
        if self.config.enable_snapshot_capture:
            snapshot_config = SnapshotConfig(
                output_dir=self.config.snapshot_output_dir,
            )
            self.snapshot_manager = SnapshotManager(
                config=snapshot_config,
                on_snapshot=self._on_snapshot_captured,
            )
            self.snapshot_manager.set_panels(
                board=self.board,
                tricortex=self.tricortex,
                search_tree=self.search_tree,
                motif_tracker=self.motif_tracker,
                quantum=self.quantum,
                anti_holographic=self.anti_holographic,
                telemetry=self.telemetry,
                control=self.control,
            )
        
        # Callbacks
        self._on_move_callback: Callable[[str], None] | None = None
        self._on_snapshot_callback: Callable[[Any], None] | None = None
        
        # Current game state
        self._current_position: Position | None = None
        self._game_active = False
    
    def set_on_move_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for when a move is made via GUI.
        
        Args:
            callback: Callback function receiving UCI move string
        """
        self._on_move_callback = callback
    
    def set_on_snapshot_callback(self, callback: Callable[[Any], None]) -> None:
        """Set callback for when a snapshot is captured.
        
        Args:
            callback: Callback function receiving snapshot data
        """
        self._on_snapshot_callback = callback
    
    def new_game(self, fen: str | None = None) -> None:
        """Start a new game.
        
        Args:
            fen: Starting position FEN (default: standard starting position)
        """
        from qratum_chess.core.position import Position
        
        if fen:
            self._current_position = Position.from_fen(fen)
        else:
            self._current_position = Position.starting()
            fen = self._current_position.to_fen()
        
        # Reset all panels
        self.board.set_position(fen)
        self.board.clear_overlays()
        
        self.motif_tracker.reset_game()
        self.anti_holographic.reset()
        self.telemetry.reset_session()
        self.search_tree.clear()
        
        self._game_active = True
        
        # Start snapshot capture
        if self.snapshot_manager:
            self.snapshot_manager.start()
            self.snapshot_manager.new_game(fen)
    
    def stop_game(self) -> None:
        """Stop current game."""
        self._game_active = False
        
        if self.snapshot_manager:
            if self._current_position:
                self.snapshot_manager.game_ended("*", self._current_position.to_fen())
            self.snapshot_manager.stop()
    
    def update_position(
        self,
        position: 'Position',
        last_move: 'Move | None' = None,
        pv: list['Move'] | None = None,
        evaluation: float = 0.0,
    ) -> None:
        """Update the displayed position.
        
        Args:
            position: New position
            last_move: Last move made (for highlighting)
            pv: Principal variation (for arrow display)
            evaluation: Position evaluation
        """
        self._current_position = position
        fen = position.to_fen()
        
        # Update board
        self.board.set_position(fen)
        self.board.clear_overlays()
        
        if last_move:
            self.board.highlight_move(last_move.from_sq, last_move.to_sq)
            
            # Animate the move
            piece = position.board.piece_at(last_move.to_sq)
            if piece:
                self.board.animate_move(last_move.from_sq, last_move.to_sq, piece)
        
        if pv:
            pv_tuples = [(m.from_sq, m.to_sq) for m in pv[:5]]
            self.board.add_pv_arrows(pv_tuples, (0.0, 0.8, 1.0, 0.7))
        
        # Update evaluation bar
        self.board.set_evaluation(evaluation)
        
        # Detect motifs
        self.motif_tracker.detect_motifs(
            position,
            move_number=self.control.state.current_move_number,
        )
        
        # Notify snapshot manager
        if self.snapshot_manager and last_move:
            self.snapshot_manager.on_move(
                last_move.to_uci(),
                fen,
                evaluation,
            )
    
    def update_cortex_diagnostics(self, diagnostics: dict[str, Any]) -> None:
        """Update tri-cortex diagnostics panel.
        
        Args:
            diagnostics: Diagnostics from TriModalCore.evaluate()
        """
        self.tricortex.update_from_trimodal(diagnostics)
        
        # Add time series point
        move_num = self.control.state.current_move_number
        self.tricortex.add_time_series_point(
            move_num,
            diagnostics.get('tactical_weight', 0.33),
            diagnostics.get('strategic_weight', 0.33),
            diagnostics.get('conceptual_weight', 0.34),
        )
    
    def update_search_tree(self, root_node: Any) -> None:
        """Update search tree visualization.
        
        Args:
            root_node: MCTS root node
        """
        self.search_tree.update_from_mcts(root_node)
    
    def update_telemetry(
        self,
        nodes: int,
        evaluations: int,
        elapsed: float,
        threads: int = 1,
        hash_hits: int = 0,
        hash_probes: int = 0,
    ) -> None:
        """Update telemetry dashboard.
        
        Args:
            nodes: Nodes searched
            evaluations: Evaluations performed
            elapsed: Time elapsed in seconds
            threads: Active threads
            hash_hits: Hash table hits
            hash_probes: Hash table probes
        """
        self.telemetry.update_performance(nodes, evaluations, elapsed)
        self.telemetry.update_threads(threads, threads, 0.8)
        
        if hash_probes > 0:
            self.telemetry.update_cache(hash_hits, hash_probes, 256, hash_hits, hash_probes * 10)
    
    def update_anti_holographic(
        self,
        move_probabilities: list[float],
        selected_rank: int,
        evaluation: float,
        previous_eval: float,
    ) -> None:
        """Update anti-holographic indicators.
        
        Args:
            move_probabilities: Probability distribution over moves
            selected_rank: Rank of selected move (0 = best)
            evaluation: Current evaluation
            previous_eval: Previous evaluation
        """
        total_moves = len(move_probabilities)
        
        self.anti_holographic.update_stochasticity(
            move_probabilities,
            selected_rank,
            total_moves,
        )
        
        self.anti_holographic.update_evaluation_drift(
            previous_eval,
            evaluation,
        )
        
        # Compute destabilization (simplified)
        complexity = min(1.0, total_moves / 40)
        eval_variance = abs(evaluation - previous_eval)
        
        self.anti_holographic.update_destabilization(
            complexity,
            eval_variance,
            1.0,  # Time pressure ratio
            0.3,  # Style disruption
        )
    
    def update_quantum(
        self,
        num_qubits: int,
        amplitudes: list[complex],
        entangled_pairs: list[tuple[int, int]] | None = None,
    ) -> None:
        """Update quantum panel.
        
        Args:
            num_qubits: Number of qubits
            amplitudes: Quantum state amplitudes
            entangled_pairs: Entangled qubit pairs
        """
        if self.quantum.state.enabled:
            self.quantum.update_circuit_state(num_qubits, amplitudes, entangled_pairs)
    
    def highlight_legal_moves(self, position: 'Position', from_square: int) -> None:
        """Highlight legal moves from a square.
        
        Args:
            position: Current position
            from_square: Source square
        """
        legal_moves = position.generate_legal_moves()
        destinations = [m.to_sq for m in legal_moves if m.from_sq == from_square]
        self.board.highlight_legal_moves(destinations)
    
    def toggle_quantum_panel(self) -> None:
        """Toggle quantum panel visibility."""
        self.quantum.set_enabled(not self.quantum.state.enabled)
        self.control.toggle_quantum(self.quantum.state.enabled)
    
    def toggle_3d_view(self) -> None:
        """Toggle between 2D and 3D board view."""
        self.board.toggle_render_mode()
    
    def set_board_theme(self, theme: BoardTheme) -> None:
        """Set board visual theme.
        
        Args:
            theme: Theme to apply
        """
        self.board.set_theme(theme)
    
    def flip_board(self) -> None:
        """Flip board orientation."""
        self.board.flip_board()
    
    def manual_snapshot(self, description: str = "") -> Any:
        """Manually capture a snapshot.
        
        Args:
            description: Snapshot description
            
        Returns:
            Captured snapshot
        """
        if self.snapshot_manager:
            return self.snapshot_manager.manual_capture(description)
        return None
    
    def generate_snapshot_report(self, format: str = "html") -> str | None:
        """Generate snapshot report.
        
        Args:
            format: Report format (html, json)
            
        Returns:
            Path to generated report
        """
        if self.snapshot_manager:
            return str(self.snapshot_manager.generate_report(format))
        return None
    
    def get_full_render_data(self) -> dict[str, Any]:
        """Get combined render data from all panels.
        
        Returns:
            Dictionary with all panel data for web frontend
        """
        return {
            'timestamp': time.time(),
            'game_active': self._game_active,
            'panels': {
                'board': self.board.get_render_data(),
                'tricortex': self.tricortex.get_render_data(),
                'search_tree': self.search_tree.get_render_data(),
                'motif_tracker': self.motif_tracker.get_render_data(),
                'quantum': self.quantum.get_render_data(),
                'anti_holographic': self.anti_holographic.get_render_data(),
                'telemetry': self.telemetry.get_render_data(),
                'control': self.control.get_render_data(),
            },
            'snapshot_stats': self.snapshot_manager.get_stats() if self.snapshot_manager else None,
        }
    
    def to_json(self) -> str:
        """Serialize full render data to JSON.
        
        Returns:
            JSON string
        """
        return json.dumps(self.get_full_render_data(), default=str)
    
    def _on_move_from_board(self, uci_move: str) -> None:
        """Handle move input from board panel.
        
        Args:
            uci_move: UCI move string
        """
        if self._on_move_callback:
            self._on_move_callback(uci_move)
    
    def _on_settings_change(self, settings: dict[str, Any]) -> None:
        """Handle settings change from control panel.
        
        Args:
            settings: New settings
        """
        # Apply cortex weight changes
        cortex = settings.get('cortex', {})
        # These would be applied to the engine configuration
        
        # Update quantum panel
        quantum = settings.get('quantum', {})
        if quantum.get('enabled') != self.quantum.state.enabled:
            self.quantum.set_enabled(quantum.get('enabled', False))
    
    def _on_snapshot_captured(self, snapshot: Any) -> None:
        """Handle snapshot capture event.
        
        Args:
            snapshot: Captured snapshot
        """
        if self._on_snapshot_callback:
            self._on_snapshot_callback(snapshot)
