"""Control Panel: Game modes, settings, and command interface.

Features:
- Human vs AI, AI vs AI, batch benchmarking, replay control
- Log export, quantum toggle, cortex visibility toggle, audit mode
- Simulation settings and fault-injection triggers
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class GameMode(Enum):
    """Game mode options."""

    HUMAN_VS_AI = "human_vs_ai"
    AI_VS_AI = "ai_vs_ai"
    ANALYSIS = "analysis"
    BATCH_BENCHMARK = "batch_benchmark"
    REPLAY = "replay"


class TimeControl(Enum):
    """Time control presets."""

    BULLET_1_0 = "1+0"
    BULLET_2_1 = "2+1"
    BLITZ_3_0 = "3+0"
    BLITZ_5_0 = "5+0"
    RAPID_10_0 = "10+0"
    RAPID_15_10 = "15+10"
    CLASSICAL_30_0 = "30+0"
    UNLIMITED = "unlimited"


class SearchMode(Enum):
    """Search algorithm mode."""

    ADAPTIVE = "adaptive"  # AAS
    ALPHA_BETA = "alpha_beta"
    MCTS = "mcts"
    HYBRID = "hybrid"


@dataclass
class GameSettings:
    """Game configuration settings."""

    mode: GameMode = GameMode.HUMAN_VS_AI
    time_control: TimeControl = TimeControl.BLITZ_5_0
    search_mode: SearchMode = SearchMode.ADAPTIVE

    # Engine settings
    search_depth: int = 0  # 0 = unlimited
    search_time_ms: int = 0  # 0 = use time control
    num_threads: int = 1
    hash_size_mb: int = 256

    # AI settings
    multipv: int = 1  # Number of principal variations
    ponder: bool = False

    # Display settings
    show_engine_output: bool = True
    show_pv_lines: bool = True
    auto_flip_board: bool = True


@dataclass
class CortexSettings:
    """Tri-cortex visibility and weight settings."""

    show_tactical: bool = True
    show_strategic: bool = True
    show_conceptual: bool = True

    tactical_weight: float = 0.33
    strategic_weight: float = 0.33
    conceptual_weight: float = 0.34

    # Override mode
    weight_override: bool = False


@dataclass
class QuantumSettings:
    """Quantum panel settings."""

    enabled: bool = False
    show_amplitudes: bool = True
    show_entanglement: bool = True
    show_qubo: bool = False
    quantum_assist_search: bool = False


@dataclass
class AuditSettings:
    """Audit and logging settings."""

    enabled: bool = False
    log_moves: bool = True
    log_evaluations: bool = True
    log_cortex_outputs: bool = False
    log_quantum_data: bool = False
    export_format: str = "json"  # json, csv, pgn


@dataclass
class FaultInjectionSettings:
    """Fault injection for resilience testing."""

    enabled: bool = False
    corrupt_hash_rate: float = 0.0
    inject_nan_rate: float = 0.0
    drop_thread_rate: float = 0.0
    latency_injection_ms: float = 0.0


@dataclass
class BenchmarkSettings:
    """Benchmark mode settings."""

    num_games: int = 100
    opponent_engine: str = "self"
    time_control: TimeControl = TimeControl.BLITZ_3_0
    opening_book: bool = True
    record_games: bool = True
    parallel_games: int = 1


@dataclass
class ControlPanelState:
    """Complete state of the control panel."""

    # Settings
    game: GameSettings = field(default_factory=GameSettings)
    cortex: CortexSettings = field(default_factory=CortexSettings)
    quantum: QuantumSettings = field(default_factory=QuantumSettings)
    audit: AuditSettings = field(default_factory=AuditSettings)
    fault_injection: FaultInjectionSettings = field(default_factory=FaultInjectionSettings)
    benchmark: BenchmarkSettings = field(default_factory=BenchmarkSettings)

    # Game state
    game_in_progress: bool = False
    game_paused: bool = False
    current_move_number: int = 0

    # Replay state
    replay_position: int = 0
    replay_length: int = 0
    replay_playing: bool = False
    replay_speed: float = 1.0

    # Benchmark state
    benchmark_running: bool = False
    benchmark_progress: float = 0.0
    benchmark_games_complete: int = 0

    # Status
    last_command: str = ""
    last_error: str = ""
    engine_status: str = "idle"


class ControlPanel:
    """Control Panel for game modes, settings, and commands.

    This panel provides:
    - Game mode selection (Human vs AI, AI vs AI, Analysis, etc.)
    - Time control and engine settings
    - Cortex visibility toggles and weight adjustment
    - Quantum panel enable/disable
    - Audit logging and export
    - Fault injection for testing
    - Replay controls
    """

    def __init__(
        self,
        width: int = 350,
        height: int = 600,
        on_new_game: Callable[[], None] | None = None,
        on_stop_game: Callable[[], None] | None = None,
        on_settings_change: Callable[[dict], None] | None = None,
    ) -> None:
        """Initialize control panel.

        Args:
            width: Panel width in pixels
            height: Panel height in pixels
            on_new_game: Callback when new game is started
            on_stop_game: Callback when game is stopped
            on_settings_change: Callback when settings change
        """
        self.width = width
        self.height = height
        self.state = ControlPanelState()

        self.on_new_game = on_new_game
        self.on_stop_game = on_stop_game
        self.on_settings_change = on_settings_change

        # Command history
        self.command_history: list[str] = []

    def new_game(self) -> bool:
        """Start a new game.

        Returns:
            True if game started successfully
        """
        if self.state.game_in_progress:
            self.stop_game()

        self.state.game_in_progress = True
        self.state.game_paused = False
        self.state.current_move_number = 0
        self.state.engine_status = "thinking"
        self.state.last_command = "new_game"

        self._add_to_history("New game started")

        if self.on_new_game:
            self.on_new_game()

        return True

    def stop_game(self) -> bool:
        """Stop current game.

        Returns:
            True if game stopped successfully
        """
        self.state.game_in_progress = False
        self.state.game_paused = False
        self.state.engine_status = "idle"
        self.state.last_command = "stop_game"

        self._add_to_history("Game stopped")

        if self.on_stop_game:
            self.on_stop_game()

        return True

    def pause_game(self) -> bool:
        """Pause/resume current game.

        Returns:
            True if state changed
        """
        if self.state.game_in_progress:
            self.state.game_paused = not self.state.game_paused
            self.state.engine_status = "paused" if self.state.game_paused else "thinking"
            self._add_to_history(f"Game {'paused' if self.state.game_paused else 'resumed'}")
            return True
        return False

    def set_game_mode(self, mode: GameMode) -> None:
        """Set game mode.

        Args:
            mode: Game mode to set
        """
        self.state.game.mode = mode
        self._notify_settings_change()
        self._add_to_history(f"Mode set to {mode.value}")

    def set_time_control(self, tc: TimeControl) -> None:
        """Set time control.

        Args:
            tc: Time control preset
        """
        self.state.game.time_control = tc
        self._notify_settings_change()
        self._add_to_history(f"Time control: {tc.value}")

    def set_search_mode(self, mode: SearchMode) -> None:
        """Set search algorithm mode.

        Args:
            mode: Search mode
        """
        self.state.game.search_mode = mode
        self._notify_settings_change()
        self._add_to_history(f"Search mode: {mode.value}")

    def set_threads(self, num_threads: int) -> None:
        """Set number of search threads.

        Args:
            num_threads: Number of threads
        """
        self.state.game.num_threads = max(1, min(64, num_threads))
        self._notify_settings_change()

    def set_hash_size(self, size_mb: int) -> None:
        """Set hash table size.

        Args:
            size_mb: Hash table size in MB
        """
        self.state.game.hash_size_mb = max(1, min(4096, size_mb))
        self._notify_settings_change()

    def toggle_cortex(self, cortex: str, visible: bool | None = None) -> None:
        """Toggle cortex visibility.

        Args:
            cortex: Cortex name (tactical, strategic, conceptual)
            visible: Force visibility state (None to toggle)
        """
        if cortex == "tactical":
            self.state.cortex.show_tactical = (
                visible if visible is not None else not self.state.cortex.show_tactical
            )
        elif cortex == "strategic":
            self.state.cortex.show_strategic = (
                visible if visible is not None else not self.state.cortex.show_strategic
            )
        elif cortex == "conceptual":
            self.state.cortex.show_conceptual = (
                visible if visible is not None else not self.state.cortex.show_conceptual
            )

        self._notify_settings_change()

    def set_cortex_weights(self, tactical: float, strategic: float, conceptual: float) -> None:
        """Set cortex weights (must sum to 1.0).

        Args:
            tactical: Tactical cortex weight
            strategic: Strategic cortex weight
            conceptual: Conceptual cortex weight
        """
        total = tactical + strategic + conceptual
        if total > 0:
            self.state.cortex.tactical_weight = tactical / total
            self.state.cortex.strategic_weight = strategic / total
            self.state.cortex.conceptual_weight = conceptual / total

        self._notify_settings_change()

    def toggle_quantum(self, enabled: bool | None = None) -> None:
        """Toggle quantum panel.

        Args:
            enabled: Force state (None to toggle)
        """
        self.state.quantum.enabled = (
            enabled if enabled is not None else not self.state.quantum.enabled
        )
        self._notify_settings_change()
        self._add_to_history(f"Quantum {'enabled' if self.state.quantum.enabled else 'disabled'}")

    def toggle_audit(self, enabled: bool | None = None) -> None:
        """Toggle audit logging.

        Args:
            enabled: Force state (None to toggle)
        """
        self.state.audit.enabled = enabled if enabled is not None else not self.state.audit.enabled
        self._notify_settings_change()
        self._add_to_history(f"Audit {'enabled' if self.state.audit.enabled else 'disabled'}")

    def set_fault_injection(self, enabled: bool, **settings) -> None:
        """Configure fault injection for resilience testing.

        Args:
            enabled: Whether fault injection is enabled
            **settings: Fault injection parameters
        """
        self.state.fault_injection.enabled = enabled

        if "corrupt_hash_rate" in settings:
            self.state.fault_injection.corrupt_hash_rate = settings["corrupt_hash_rate"]
        if "inject_nan_rate" in settings:
            self.state.fault_injection.inject_nan_rate = settings["inject_nan_rate"]
        if "drop_thread_rate" in settings:
            self.state.fault_injection.drop_thread_rate = settings["drop_thread_rate"]
        if "latency_injection_ms" in settings:
            self.state.fault_injection.latency_injection_ms = settings["latency_injection_ms"]

        self._notify_settings_change()

    def start_benchmark(self) -> bool:
        """Start benchmark run.

        Returns:
            True if benchmark started
        """
        if self.state.benchmark_running:
            return False

        self.state.benchmark_running = True
        self.state.benchmark_progress = 0.0
        self.state.benchmark_games_complete = 0
        self.state.game.mode = GameMode.BATCH_BENCHMARK

        self._add_to_history(f"Benchmark started: {self.state.benchmark.num_games} games")

        return True

    def stop_benchmark(self) -> None:
        """Stop benchmark run."""
        self.state.benchmark_running = False
        self._add_to_history("Benchmark stopped")

    def update_benchmark_progress(self, games_complete: int) -> None:
        """Update benchmark progress.

        Args:
            games_complete: Number of games completed
        """
        self.state.benchmark_games_complete = games_complete
        self.state.benchmark_progress = games_complete / max(1, self.state.benchmark.num_games)

    # Replay controls
    def load_replay(self, moves: list[str]) -> None:
        """Load game for replay.

        Args:
            moves: List of moves in UCI format
        """
        self.state.game.mode = GameMode.REPLAY
        self.state.replay_length = len(moves)
        self.state.replay_position = 0
        self.state.replay_playing = False
        self._add_to_history(f"Loaded replay: {len(moves)} moves")

    def replay_step(self, direction: int = 1) -> int:
        """Step through replay.

        Args:
            direction: 1 for forward, -1 for backward

        Returns:
            New position
        """
        new_pos = self.state.replay_position + direction
        self.state.replay_position = max(0, min(self.state.replay_length, new_pos))
        return self.state.replay_position

    def replay_goto(self, position: int) -> int:
        """Go to specific replay position.

        Args:
            position: Target position

        Returns:
            New position
        """
        self.state.replay_position = max(0, min(self.state.replay_length, position))
        return self.state.replay_position

    def replay_play_pause(self) -> bool:
        """Toggle replay playback.

        Returns:
            True if now playing
        """
        self.state.replay_playing = not self.state.replay_playing
        return self.state.replay_playing

    def export_log(self, format: str = "json") -> dict[str, Any]:
        """Export game log.

        Args:
            format: Export format (json, csv, pgn)

        Returns:
            Export data
        """
        self.state.audit.export_format = format

        return {
            "format": format,
            "timestamp": time.time(),
            "game_mode": self.state.game.mode.value,
            "command_history": self.command_history[-100:],
            "settings": {
                "game": {
                    "mode": self.state.game.mode.value,
                    "time_control": self.state.game.time_control.value,
                    "search_mode": self.state.game.search_mode.value,
                    "threads": self.state.game.num_threads,
                    "hash_mb": self.state.game.hash_size_mb,
                },
                "cortex": {
                    "tactical_weight": self.state.cortex.tactical_weight,
                    "strategic_weight": self.state.cortex.strategic_weight,
                    "conceptual_weight": self.state.cortex.conceptual_weight,
                },
                "quantum_enabled": self.state.quantum.enabled,
                "audit_enabled": self.state.audit.enabled,
            },
        }

    def _add_to_history(self, message: str) -> None:
        """Add message to command history."""
        timestamp = time.strftime("%H:%M:%S")
        self.command_history.append(f"[{timestamp}] {message}")
        if len(self.command_history) > 1000:
            self.command_history = self.command_history[-1000:]

    def _notify_settings_change(self) -> None:
        """Notify settings change callback."""
        if self.on_settings_change:
            self.on_settings_change(self.get_settings())

    def get_settings(self) -> dict[str, Any]:
        """Get current settings as dictionary."""
        return {
            "game": {
                "mode": self.state.game.mode.value,
                "time_control": self.state.game.time_control.value,
                "search_mode": self.state.game.search_mode.value,
                "search_depth": self.state.game.search_depth,
                "search_time_ms": self.state.game.search_time_ms,
                "num_threads": self.state.game.num_threads,
                "hash_size_mb": self.state.game.hash_size_mb,
                "multipv": self.state.game.multipv,
                "ponder": self.state.game.ponder,
            },
            "cortex": {
                "show_tactical": self.state.cortex.show_tactical,
                "show_strategic": self.state.cortex.show_strategic,
                "show_conceptual": self.state.cortex.show_conceptual,
                "tactical_weight": self.state.cortex.tactical_weight,
                "strategic_weight": self.state.cortex.strategic_weight,
                "conceptual_weight": self.state.cortex.conceptual_weight,
            },
            "quantum": {
                "enabled": self.state.quantum.enabled,
                "show_amplitudes": self.state.quantum.show_amplitudes,
                "show_entanglement": self.state.quantum.show_entanglement,
                "quantum_assist": self.state.quantum.quantum_assist_search,
            },
            "audit": {
                "enabled": self.state.audit.enabled,
                "log_moves": self.state.audit.log_moves,
                "log_evaluations": self.state.audit.log_evaluations,
            },
        }

    def get_render_data(self) -> dict[str, Any]:
        """Get all data needed for rendering.

        Returns:
            Dictionary with panel state for visualization
        """
        return {
            "width": self.width,
            "height": self.height,
            "game": {
                "mode": self.state.game.mode.value,
                "time_control": self.state.game.time_control.value,
                "search_mode": self.state.game.search_mode.value,
                "in_progress": self.state.game_in_progress,
                "paused": self.state.game_paused,
                "move_number": self.state.current_move_number,
                "threads": self.state.game.num_threads,
                "hash_mb": self.state.game.hash_size_mb,
            },
            "cortex": {
                "show_tactical": self.state.cortex.show_tactical,
                "show_strategic": self.state.cortex.show_strategic,
                "show_conceptual": self.state.cortex.show_conceptual,
                "weights": {
                    "tactical": self.state.cortex.tactical_weight,
                    "strategic": self.state.cortex.strategic_weight,
                    "conceptual": self.state.cortex.conceptual_weight,
                },
            },
            "quantum": {
                "enabled": self.state.quantum.enabled,
            },
            "audit": {
                "enabled": self.state.audit.enabled,
            },
            "fault_injection": {
                "enabled": self.state.fault_injection.enabled,
            },
            "benchmark": {
                "running": self.state.benchmark_running,
                "progress": self.state.benchmark_progress,
                "games_complete": self.state.benchmark_games_complete,
                "total_games": self.state.benchmark.num_games,
            },
            "replay": {
                "position": self.state.replay_position,
                "length": self.state.replay_length,
                "playing": self.state.replay_playing,
                "speed": self.state.replay_speed,
            },
            "status": {
                "engine": self.state.engine_status,
                "last_command": self.state.last_command,
                "last_error": self.state.last_error,
            },
            "history": self.command_history[-20:],
            "available_modes": [m.value for m in GameMode],
            "available_time_controls": [tc.value for tc in TimeControl],
            "available_search_modes": [sm.value for sm in SearchMode],
        }

    def to_json(self) -> str:
        """Serialize render data to JSON."""
        import json

        return json.dumps(self.get_render_data())
