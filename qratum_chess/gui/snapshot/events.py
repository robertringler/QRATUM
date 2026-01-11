"""Snapshot event definitions and detection logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time


class SnapshotTrigger(Enum):
    """Types of events that trigger snapshots."""
    MANUAL = "manual"
    EVALUATION_SHIFT = "evaluation_shift"
    MOTIF_DISCOVERY = "motif_discovery"
    PHASE_TRANSITION = "phase_transition"
    NOVELTY_DETECTION = "novelty_detection"
    TIME_INTERVAL = "time_interval"
    MOVE_MADE = "move_made"
    GAME_START = "game_start"
    GAME_END = "game_end"
    CRITICAL_MOMENT = "critical_moment"
    BENCHMARK_CHECKPOINT = "benchmark_checkpoint"


class GamePhase(Enum):
    """Chess game phases."""
    OPENING = "opening"
    MIDDLEGAME = "middlegame"
    ENDGAME = "endgame"


@dataclass
class SnapshotEvent:
    """Event that triggered a snapshot capture.
    
    Attributes:
        trigger: Type of event that triggered the snapshot
        timestamp: Unix timestamp of the event
        move_number: Current move number
        phase: Current game phase
        fen: FEN string of the position
        evaluation: Current position evaluation
        pv_line: Principal variation moves
        metrics: Additional metrics from cortices and telemetry
        description: Human-readable event description
    """
    trigger: SnapshotTrigger
    timestamp: float = field(default_factory=time.time)
    move_number: int = 0
    phase: GamePhase = GamePhase.OPENING
    fen: str = ""
    evaluation: float = 0.0
    pv_line: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'trigger': self.trigger.value,
            'timestamp': self.timestamp,
            'timestamp_formatted': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp)),
            'move_number': self.move_number,
            'phase': self.phase.value,
            'fen': self.fen,
            'evaluation': self.evaluation,
            'pv_line': self.pv_line,
            'metrics': self.metrics,
            'description': self.description,
        }


class EventDetector:
    """Detects significant events that warrant snapshots.
    
    Monitors game state and telemetry to identify:
    - Large evaluation shifts (Δ > threshold)
    - Motif emergence
    - Phase transitions
    - Novelty detection
    - Critical moments (checks, captures, promotions)
    """
    
    def __init__(
        self,
        eval_threshold: float = 1.0,
        novelty_threshold: float = 0.3,
    ) -> None:
        """Initialize event detector.
        
        Args:
            eval_threshold: Evaluation change threshold for triggering
            novelty_threshold: Novelty score threshold for triggering
        """
        self.eval_threshold = eval_threshold
        self.novelty_threshold = novelty_threshold
        
        # State tracking
        self._last_evaluation: float | None = None
        self._last_phase: GamePhase | None = None
        self._detected_motifs: set[str] = set()
        self._move_count = 0
    
    def reset(self) -> None:
        """Reset detector state for new game."""
        self._last_evaluation = None
        self._last_phase = None
        self._detected_motifs.clear()
        self._move_count = 0
    
    def check_evaluation_shift(
        self,
        current_eval: float,
        fen: str,
        pv: list[str],
    ) -> SnapshotEvent | None:
        """Check for significant evaluation shift.
        
        Args:
            current_eval: Current position evaluation
            fen: Current position FEN
            pv: Principal variation
            
        Returns:
            SnapshotEvent if shift detected, None otherwise
        """
        if self._last_evaluation is not None:
            delta = abs(current_eval - self._last_evaluation)
            
            if delta >= self.eval_threshold:
                direction = "improved" if current_eval > self._last_evaluation else "worsened"
                event = SnapshotEvent(
                    trigger=SnapshotTrigger.EVALUATION_SHIFT,
                    move_number=self._move_count,
                    fen=fen,
                    evaluation=current_eval,
                    pv_line=pv,
                    metrics={
                        'previous_eval': self._last_evaluation,
                        'delta': delta,
                    },
                    description=f"Evaluation {direction} by {delta:.2f} (from {self._last_evaluation:.2f} to {current_eval:.2f})",
                )
                self._last_evaluation = current_eval
                return event
        
        self._last_evaluation = current_eval
        return None
    
    def check_phase_transition(
        self,
        phase: GamePhase,
        fen: str,
        evaluation: float,
    ) -> SnapshotEvent | None:
        """Check for game phase transition.
        
        Args:
            phase: Current game phase
            fen: Current position FEN
            evaluation: Current evaluation
            
        Returns:
            SnapshotEvent if transition detected, None otherwise
        """
        if self._last_phase is not None and phase != self._last_phase:
            event = SnapshotEvent(
                trigger=SnapshotTrigger.PHASE_TRANSITION,
                move_number=self._move_count,
                phase=phase,
                fen=fen,
                evaluation=evaluation,
                metrics={
                    'previous_phase': self._last_phase.value,
                },
                description=f"Phase transition: {self._last_phase.value} → {phase.value}",
            )
            self._last_phase = phase
            return event
        
        self._last_phase = phase
        return None
    
    def check_motif_discovery(
        self,
        motifs: list[dict[str, Any]],
        fen: str,
        evaluation: float,
    ) -> SnapshotEvent | None:
        """Check for new motif discovery.
        
        Args:
            motifs: List of detected motifs
            fen: Current position FEN
            evaluation: Current evaluation
            
        Returns:
            SnapshotEvent if new motif detected, None otherwise
        """
        new_motifs = []
        for motif in motifs:
            motif_key = f"{motif.get('type', '')}_{motif.get('squares', [])}"
            if motif_key not in self._detected_motifs:
                self._detected_motifs.add(motif_key)
                new_motifs.append(motif)
        
        if new_motifs:
            return SnapshotEvent(
                trigger=SnapshotTrigger.MOTIF_DISCOVERY,
                move_number=self._move_count,
                fen=fen,
                evaluation=evaluation,
                metrics={
                    'new_motifs': new_motifs,
                    'total_motifs': len(self._detected_motifs),
                },
                description=f"New motifs discovered: {[m.get('type', 'unknown') for m in new_motifs]}",
            )
        
        return None
    
    def check_novelty(
        self,
        novelty_score: float,
        fen: str,
        evaluation: float,
        move: str,
    ) -> SnapshotEvent | None:
        """Check for novelty detection.
        
        Args:
            novelty_score: Current novelty score (0-1)
            fen: Current position FEN
            evaluation: Current evaluation
            move: Move that was played
            
        Returns:
            SnapshotEvent if novelty detected, None otherwise
        """
        if novelty_score >= self.novelty_threshold:
            return SnapshotEvent(
                trigger=SnapshotTrigger.NOVELTY_DETECTION,
                move_number=self._move_count,
                fen=fen,
                evaluation=evaluation,
                metrics={
                    'novelty_score': novelty_score,
                    'move': move,
                },
                description=f"Novel move detected: {move} (score: {novelty_score:.2f})",
            )
        
        return None
    
    def check_critical_moment(
        self,
        move: str,
        is_check: bool,
        is_capture: bool,
        is_promotion: bool,
        fen: str,
        evaluation: float,
    ) -> SnapshotEvent | None:
        """Check for critical moment (check, capture, promotion).
        
        Args:
            move: Move played
            is_check: Whether move gives check
            is_capture: Whether move is a capture
            is_promotion: Whether move is a promotion
            fen: Current position FEN
            evaluation: Current evaluation
            
        Returns:
            SnapshotEvent if critical moment, None otherwise
        """
        reasons = []
        if is_check:
            reasons.append("check")
        if is_capture:
            reasons.append("capture")
        if is_promotion:
            reasons.append("promotion")
        
        # Only trigger for combined critical moments (check + capture, etc.)
        if len(reasons) >= 2 or is_promotion:
            return SnapshotEvent(
                trigger=SnapshotTrigger.CRITICAL_MOMENT,
                move_number=self._move_count,
                fen=fen,
                evaluation=evaluation,
                metrics={
                    'move': move,
                    'is_check': is_check,
                    'is_capture': is_capture,
                    'is_promotion': is_promotion,
                },
                description=f"Critical moment: {move} ({', '.join(reasons)})",
            )
        
        return None
    
    def on_move(self) -> None:
        """Called when a move is made."""
        self._move_count += 1
    
    def determine_phase(self, piece_count: int, queens_present: bool) -> GamePhase:
        """Determine game phase from position characteristics.
        
        Args:
            piece_count: Total number of pieces
            queens_present: Whether queens are on the board
            
        Returns:
            Current game phase
        """
        if self._move_count < 15:
            return GamePhase.OPENING
        elif piece_count <= 12 or not queens_present:
            return GamePhase.ENDGAME
        else:
            return GamePhase.MIDDLEGAME
