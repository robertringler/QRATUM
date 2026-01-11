"""Snapshot Manager for coordinating automatic capture.

Monitors game state and automatically triggers snapshot capture
based on significant events like evaluation shifts, motif discovery,
phase transitions, and novelty detection.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from qratum_chess.gui.snapshot.events import (
    EventDetector,
    GamePhase,
    SnapshotEvent,
    SnapshotTrigger,
)
from qratum_chess.gui.snapshot.capture import FullSnapshot, SnapshotCapture, SnapshotConfig

if TYPE_CHECKING:
    from qratum_chess.core.position import Position


@dataclass
class ManagerState:
    """State of the snapshot manager."""
    active: bool = False
    game_in_progress: bool = False
    current_position: str = ""
    current_move_number: int = 0
    current_phase: GamePhase = GamePhase.OPENING
    last_capture_time: float = 0.0
    total_captures: int = 0
    
    # Interval capture
    interval_timer_active: bool = False
    last_interval_capture: float = 0.0


class SnapshotManager:
    """Manages automatic snapshot capture during QRATUM-Chess operation.
    
    This class:
    - Monitors game state for significant events
    - Automatically triggers snapshots at appropriate moments
    - Coordinates interval-based capturing
    - Manages capture rate limiting
    - Provides hooks for manual capture
    
    Usage:
        manager = SnapshotManager(config)
        manager.set_panels(board, tricortex, ...)
        manager.start()
        
        # During game:
        manager.on_move(move, position, evaluation)
        manager.on_evaluation_update(eval, pv)
        manager.on_motif_detected(motifs)
        
        manager.stop()
        report = manager.generate_report()
    """
    
    def __init__(
        self,
        config: SnapshotConfig | None = None,
        on_snapshot: Callable[[FullSnapshot], None] | None = None,
    ) -> None:
        """Initialize snapshot manager.
        
        Args:
            config: Snapshot configuration
            on_snapshot: Callback when snapshot is captured
        """
        self.config = config or SnapshotConfig()
        self.on_snapshot = on_snapshot
        
        # Create capture and detector instances
        self._capture = SnapshotCapture(config=self.config)
        self._detector = EventDetector(
            eval_threshold=self.config.eval_threshold,
            novelty_threshold=self.config.novelty_threshold,
        )
        
        # State
        self._state = ManagerState()
        
        # Captured snapshots
        self._snapshots: list[FullSnapshot] = []
        
        # Rate limiting
        self._min_capture_interval = 1.0  # Minimum seconds between captures
        
        # Interval capture thread
        self._interval_thread: threading.Thread | None = None
        self._stop_interval = threading.Event()
    
    @property
    def session_id(self) -> str:
        """Get current session ID."""
        return self._capture.session_id
    
    @property
    def snapshots(self) -> list[FullSnapshot]:
        """Get all captured snapshots."""
        return self._snapshots.copy()
    
    def set_panels(self, **panels) -> None:
        """Set panel references for capture.
        
        Args:
            **panels: Panel keyword arguments (board, tricortex, etc.)
        """
        self._capture.set_panels(**panels)
    
    def start(self) -> None:
        """Start snapshot manager."""
        self._state.active = True
        self._state.total_captures = 0
        self._snapshots.clear()
        self._detector.reset()
        
        # Start interval capture if configured
        if self.config.time_interval > 0:
            self._start_interval_capture()
    
    def stop(self) -> None:
        """Stop snapshot manager."""
        self._state.active = False
        self._stop_interval.set()
        
        if self._interval_thread and self._interval_thread.is_alive():
            self._interval_thread.join(timeout=2.0)
    
    def new_game(self, fen: str = "") -> None:
        """Called when a new game starts.
        
        Args:
            fen: Starting position FEN
        """
        self._state.game_in_progress = True
        self._state.current_position = fen
        self._state.current_move_number = 0
        self._state.current_phase = GamePhase.OPENING
        self._detector.reset()
        
        # Capture game start
        if self._state.active and self.config.auto_trigger:
            event = SnapshotEvent(
                trigger=SnapshotTrigger.GAME_START,
                fen=fen,
                description="New game started",
            )
            self._do_capture(event)
    
    def game_ended(self, result: str, fen: str) -> None:
        """Called when game ends.
        
        Args:
            result: Game result (1-0, 0-1, 1/2-1/2)
            fen: Final position FEN
        """
        self._state.game_in_progress = False
        
        # Capture game end
        if self._state.active and self.config.auto_trigger:
            event = SnapshotEvent(
                trigger=SnapshotTrigger.GAME_END,
                move_number=self._state.current_move_number,
                phase=self._state.current_phase,
                fen=fen,
                metrics={'result': result},
                description=f"Game ended: {result}",
            )
            self._do_capture(event)
    
    def on_move(
        self,
        move: str,
        fen: str,
        evaluation: float,
        is_check: bool = False,
        is_capture: bool = False,
        is_promotion: bool = False,
    ) -> None:
        """Called when a move is made.
        
        Args:
            move: UCI move notation
            fen: Position FEN after move
            evaluation: Position evaluation
            is_check: Whether move gives check
            is_capture: Whether move is a capture
            is_promotion: Whether move is a promotion
        """
        if not self._state.active:
            return
        
        self._state.current_position = fen
        self._state.current_move_number += 1
        self._detector.on_move()
        
        # Check for critical moment
        if self.config.auto_trigger:
            event = self._detector.check_critical_moment(
                move, is_check, is_capture, is_promotion, fen, evaluation
            )
            if event:
                self._do_capture(event)
    
    def on_evaluation_update(
        self,
        evaluation: float,
        pv: list[str],
        fen: str,
    ) -> None:
        """Called when evaluation is updated.
        
        Args:
            evaluation: Current position evaluation
            pv: Principal variation
            fen: Current position FEN
        """
        if not self._state.active or not self.config.auto_trigger:
            return
        
        # Check for evaluation shift
        event = self._detector.check_evaluation_shift(evaluation, fen, pv)
        if event:
            self._do_capture(event)
    
    def on_phase_change(
        self,
        piece_count: int,
        queens_present: bool,
        fen: str,
        evaluation: float,
    ) -> None:
        """Called to check/update game phase.
        
        Args:
            piece_count: Total pieces on board
            queens_present: Whether queens are present
            fen: Current position FEN
            evaluation: Current evaluation
        """
        if not self._state.active or not self.config.auto_trigger:
            return
        
        # Determine current phase
        phase = self._detector.determine_phase(piece_count, queens_present)
        
        # Check for phase transition
        event = self._detector.check_phase_transition(phase, fen, evaluation)
        if event:
            self._state.current_phase = phase
            self._do_capture(event)
    
    def on_motif_detected(
        self,
        motifs: list[dict[str, Any]],
        fen: str,
        evaluation: float,
    ) -> None:
        """Called when motifs are detected.
        
        Args:
            motifs: List of detected motifs
            fen: Current position FEN
            evaluation: Current evaluation
        """
        if not self._state.active or not self.config.auto_trigger:
            return
        
        event = self._detector.check_motif_discovery(motifs, fen, evaluation)
        if event:
            self._do_capture(event)
    
    def on_novelty_detected(
        self,
        novelty_score: float,
        move: str,
        fen: str,
        evaluation: float,
    ) -> None:
        """Called when novelty is detected.
        
        Args:
            novelty_score: Novelty score (0-1)
            move: Move that was novel
            fen: Current position FEN
            evaluation: Current evaluation
        """
        if not self._state.active or not self.config.auto_trigger:
            return
        
        event = self._detector.check_novelty(novelty_score, fen, evaluation, move)
        if event:
            self._do_capture(event)
    
    def manual_capture(self, description: str = "") -> FullSnapshot | None:
        """Manually trigger a snapshot capture.
        
        Args:
            description: Description for the snapshot
            
        Returns:
            Captured snapshot or None if manager not active
        """
        if not self._state.active:
            return None
        
        event = SnapshotEvent(
            trigger=SnapshotTrigger.MANUAL,
            move_number=self._state.current_move_number,
            phase=self._state.current_phase,
            fen=self._state.current_position,
            description=description or "Manual snapshot capture",
        )
        
        return self._do_capture(event)
    
    def benchmark_checkpoint(
        self,
        games_complete: int,
        total_games: int,
        elo_estimate: float,
        metrics: dict[str, Any],
    ) -> FullSnapshot | None:
        """Capture benchmark checkpoint.
        
        Args:
            games_complete: Number of games completed
            total_games: Total games in benchmark
            elo_estimate: Current Elo estimate
            metrics: Additional benchmark metrics
            
        Returns:
            Captured snapshot
        """
        if not self._state.active:
            return None
        
        event = SnapshotEvent(
            trigger=SnapshotTrigger.BENCHMARK_CHECKPOINT,
            metrics={
                'games_complete': games_complete,
                'total_games': total_games,
                'elo_estimate': elo_estimate,
                **metrics,
            },
            description=f"Benchmark checkpoint: {games_complete}/{total_games} games, Elo: {elo_estimate:.0f}",
        )
        
        return self._do_capture(event)
    
    def _do_capture(self, event: SnapshotEvent) -> FullSnapshot | None:
        """Perform snapshot capture with rate limiting.
        
        Args:
            event: Triggering event
            
        Returns:
            Captured snapshot or None if rate limited
        """
        # Rate limiting
        now = time.time()
        if now - self._state.last_capture_time < self._min_capture_interval:
            return None
        
        # Capture
        snapshot = self._capture.capture(event)
        self._snapshots.append(snapshot)
        
        self._state.last_capture_time = now
        self._state.total_captures += 1
        
        # Notify callback
        if self.on_snapshot:
            self.on_snapshot(snapshot)
        
        return snapshot
    
    def _start_interval_capture(self) -> None:
        """Start interval-based capture thread."""
        self._stop_interval.clear()
        self._state.interval_timer_active = True
        
        self._interval_thread = threading.Thread(
            target=self._interval_capture_loop,
            daemon=True,
        )
        self._interval_thread.start()
    
    def _interval_capture_loop(self) -> None:
        """Interval capture thread main loop."""
        while not self._stop_interval.is_set():
            # Wait for interval
            if self._stop_interval.wait(timeout=self.config.time_interval):
                break
            
            # Capture if game in progress
            if self._state.game_in_progress and self._state.active:
                event = SnapshotEvent(
                    trigger=SnapshotTrigger.TIME_INTERVAL,
                    move_number=self._state.current_move_number,
                    phase=self._state.current_phase,
                    fen=self._state.current_position,
                    description=f"Interval capture ({self.config.time_interval}s)",
                )
                self._do_capture(event)
                self._state.last_interval_capture = time.time()
        
        self._state.interval_timer_active = False
    
    def generate_report(self, format: str = "html") -> Path:
        """Generate report from captured snapshots.
        
        Args:
            format: Report format (html, json)
            
        Returns:
            Path to generated report
        """
        from qratum_chess.gui.snapshot.capture import OutputFormat
        
        fmt = OutputFormat.HTML if format.lower() == "html" else OutputFormat.JSON
        return self._capture.generate_report(self._snapshots, fmt)
    
    def get_stats(self) -> dict[str, Any]:
        """Get capture statistics.
        
        Returns:
            Dictionary of statistics
        """
        trigger_counts: dict[str, int] = {}
        for snap in self._snapshots:
            trigger = snap.event.trigger.value
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
        
        return {
            'session_id': self.session_id,
            'active': self._state.active,
            'total_captures': self._state.total_captures,
            'game_in_progress': self._state.game_in_progress,
            'current_move': self._state.current_move_number,
            'current_phase': self._state.current_phase.value,
            'trigger_counts': trigger_counts,
            'interval_active': self._state.interval_timer_active,
        }
