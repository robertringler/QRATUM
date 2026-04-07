"""Quantum Panel: Optional visualization of quantum-assisted evaluation.

Features:
- Visualization of qubit encodings and amplitude distributions
- Quantum-assisted motif evaluation results (QUBO solutions)
- Decision vector overlay showing entangled move evaluations
- Toggle to enable/disable quantum visualization
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class QuantumVisualizationMode(Enum):
    """Quantum visualization modes."""

    AMPLITUDE_BARS = "amplitude_bars"
    QUBIT_GRID = "qubit_grid"
    BLOCH_SPHERES = "bloch_spheres"
    ENTANGLEMENT_GRAPH = "entanglement_graph"
    QUBO_HEATMAP = "qubo_heatmap"


@dataclass
class QubitState:
    """State of a single qubit."""

    index: int
    amplitude_0: complex
    amplitude_1: complex
    phase: float = 0.0

    @property
    def probability_0(self) -> float:
        return abs(self.amplitude_0) ** 2

    @property
    def probability_1(self) -> float:
        return abs(self.amplitude_1) ** 2

    def bloch_coordinates(self) -> tuple[float, float, float]:
        """Get Bloch sphere coordinates (x, y, z)."""
        # Simplified Bloch coordinates
        theta = 2 * np.arccos(abs(self.amplitude_0))
        phi = np.angle(self.amplitude_1) - np.angle(self.amplitude_0)

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        return (float(x), float(y), float(z))


@dataclass
class QuantumCircuitSnapshot:
    """Snapshot of quantum circuit state."""

    qubits: list[QubitState]
    entanglement_pairs: list[tuple[int, int]]  # Pairs of entangled qubits
    measurement_results: dict[str, float]  # Basis state -> probability
    circuit_depth: int = 0
    gate_count: int = 0


@dataclass
class QUBOSolution:
    """QUBO optimization solution."""

    energy: float
    configuration: dict[int, int]  # Variable -> binary value
    quality: float  # Solution quality metric (0-1)
    iterations: int = 0


@dataclass
class QuantumDecisionVector:
    """Quantum-derived move decision vector."""

    moves: list[str]  # UCI moves
    amplitudes: list[complex]
    probabilities: list[float]
    entanglement_scores: list[float]
    selected_move: str = ""


@dataclass
class QuantumPanelState:
    """Complete state of the quantum panel."""

    enabled: bool = False

    # Current quantum state
    circuit_snapshot: QuantumCircuitSnapshot | None = None

    # QUBO state
    qubo_solution: QUBOSolution | None = None
    qubo_matrix: np.ndarray | None = None

    # Decision vector
    decision_vector: QuantumDecisionVector | None = None

    # Visualization mode
    visualization_mode: QuantumVisualizationMode = QuantumVisualizationMode.AMPLITUDE_BARS

    # History
    amplitude_history: list[list[float]] = field(default_factory=list)
    energy_history: list[float] = field(default_factory=list)

    # Display options
    show_amplitudes: bool = True
    show_phases: bool = True
    show_entanglement: bool = True
    show_qubo: bool = False
    animate_transitions: bool = True
    color_by_phase: bool = True


class QuantumPanel:
    """Quantum Panel for visualizing quantum-assisted evaluation.

    This panel provides:
    - Qubit state visualization (amplitude bars, Bloch spheres, grids)
    - Entanglement network graphs
    - QUBO optimization heatmaps
    - Quantum decision vector overlays
    - Real-time amplitude distribution updates
    """

    # Color schemes
    COLORS = {
        "amplitude_0": (0.0, 0.5, 1.0),  # Blue
        "amplitude_1": (1.0, 0.3, 0.0),  # Orange
        "entangled": (0.8, 0.0, 1.0),  # Purple
        "superposition": (0.0, 1.0, 0.8),  # Cyan
        "measured": (0.0, 1.0, 0.3),  # Green
        "energy_low": (0.2, 0.8, 0.2),  # Low energy - good
        "energy_high": (0.8, 0.2, 0.2),  # High energy - bad
    }

    def __init__(self, width: int = 350, height: int = 400) -> None:
        """Initialize quantum panel.

        Args:
            width: Panel width in pixels
            height: Panel height in pixels
        """
        self.width = width
        self.height = height
        self.state = QuantumPanelState()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable quantum visualization.

        Args:
            enabled: Whether quantum panel is enabled
        """
        self.state.enabled = enabled

    def update_circuit_state(
        self,
        num_qubits: int,
        amplitudes: list[complex],
        entangled_pairs: list[tuple[int, int]] | None = None,
    ) -> None:
        """Update quantum circuit state.

        Args:
            num_qubits: Number of qubits
            amplitudes: Complex amplitudes for each basis state
            entangled_pairs: List of entangled qubit pairs
        """
        if not self.state.enabled:
            return

        # Create qubit states
        qubits = []
        for i in range(num_qubits):
            # Extract single-qubit amplitudes (simplified)
            # In reality would need partial trace
            amp_0 = complex(np.cos(i * 0.1), 0)
            amp_1 = complex(np.sin(i * 0.1), 0)

            qubits.append(
                QubitState(
                    index=i,
                    amplitude_0=amp_0,
                    amplitude_1=amp_1,
                )
            )

        # Create measurement results from amplitudes
        measurement_results = {}
        for i, amp in enumerate(amplitudes[: min(16, len(amplitudes))]):
            basis_state = format(i, f"0{num_qubits}b")
            measurement_results[basis_state] = abs(amp) ** 2

        self.state.circuit_snapshot = QuantumCircuitSnapshot(
            qubits=qubits,
            entanglement_pairs=entangled_pairs or [],
            measurement_results=measurement_results,
            circuit_depth=num_qubits * 3,  # Estimate
            gate_count=num_qubits * 5,
        )

        # Update history
        probs = [abs(a) ** 2 for a in amplitudes[:8]]
        self.state.amplitude_history.append(probs)
        if len(self.state.amplitude_history) > 100:
            self.state.amplitude_history = self.state.amplitude_history[-100:]

    def update_qubo_solution(
        self,
        matrix: np.ndarray,
        solution: dict[int, int],
        energy: float,
        quality: float,
    ) -> None:
        """Update QUBO optimization result.

        Args:
            matrix: QUBO matrix
            solution: Binary solution configuration
            energy: Solution energy
            quality: Solution quality (0-1)
        """
        if not self.state.enabled:
            return

        self.state.qubo_matrix = matrix
        self.state.qubo_solution = QUBOSolution(
            energy=energy,
            configuration=solution,
            quality=quality,
        )

        # Update energy history
        self.state.energy_history.append(energy)
        if len(self.state.energy_history) > 100:
            self.state.energy_history = self.state.energy_history[-100:]

    def update_decision_vector(
        self,
        moves: list[str],
        amplitudes: list[complex],
        entanglement_scores: list[float] | None = None,
    ) -> None:
        """Update quantum decision vector for moves.

        Args:
            moves: List of UCI moves
            amplitudes: Complex amplitudes for each move
            entanglement_scores: Entanglement contribution scores
        """
        if not self.state.enabled:
            return

        # Normalize probabilities
        probs = [abs(a) ** 2 for a in amplitudes]
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]

        # Find selected move
        selected_idx = np.argmax(probs)

        self.state.decision_vector = QuantumDecisionVector(
            moves=moves,
            amplitudes=amplitudes,
            probabilities=probs,
            entanglement_scores=entanglement_scores or [0.0] * len(moves),
            selected_move=moves[selected_idx] if moves else "",
        )

    def set_visualization_mode(self, mode: QuantumVisualizationMode) -> None:
        """Set visualization mode.

        Args:
            mode: Visualization mode to use
        """
        self.state.visualization_mode = mode

    def _get_amplitude_color(self, amplitude: complex) -> tuple[float, float, float]:
        """Get color for a complex amplitude.

        Args:
            amplitude: Complex amplitude

        Returns:
            RGB color tuple
        """
        # Color by phase if enabled
        if self.state.color_by_phase:
            phase = np.angle(amplitude)
            # Map phase to hue (0 to 2π -> 0 to 1)
            hue = (phase + np.pi) / (2 * np.pi)

            # HSV to RGB (simplified)
            h = hue * 6
            x = 1 - abs(h % 2 - 1)
            if h < 1:
                return (1, x, 0)
            elif h < 2:
                return (x, 1, 0)
            elif h < 3:
                return (0, 1, x)
            elif h < 4:
                return (0, x, 1)
            elif h < 5:
                return (x, 0, 1)
            else:
                return (1, 0, x)

        # Default color based on magnitude
        mag = abs(amplitude)
        return (
            self.COLORS["amplitude_0"][0] * (1 - mag) + self.COLORS["amplitude_1"][0] * mag,
            self.COLORS["amplitude_0"][1] * (1 - mag) + self.COLORS["amplitude_1"][1] * mag,
            self.COLORS["amplitude_0"][2] * (1 - mag) + self.COLORS["amplitude_1"][2] * mag,
        )

    def get_render_data(self) -> dict[str, Any]:
        """Get all data needed for rendering.

        Returns:
            Dictionary with panel state for visualization
        """
        if not self.state.enabled:
            return {
                "enabled": False,
                "width": self.width,
                "height": self.height,
            }

        # Circuit data
        circuit_data = None
        if self.state.circuit_snapshot:
            snap = self.state.circuit_snapshot

            qubits_data = [
                {
                    "index": q.index,
                    "prob_0": q.probability_0,
                    "prob_1": q.probability_1,
                    "phase": q.phase,
                    "bloch": q.bloch_coordinates(),
                }
                for q in snap.qubits
            ]

            circuit_data = {
                "qubits": qubits_data,
                "entanglement_pairs": snap.entanglement_pairs,
                "measurements": snap.measurement_results,
                "depth": snap.circuit_depth,
                "gates": snap.gate_count,
            }

        # QUBO data
        qubo_data = None
        if self.state.qubo_solution and self.state.qubo_matrix is not None:
            qubo_data = {
                "matrix": self.state.qubo_matrix.tolist(),
                "solution": self.state.qubo_solution.configuration,
                "energy": self.state.qubo_solution.energy,
                "quality": self.state.qubo_solution.quality,
            }

        # Decision vector data
        decision_data = None
        if self.state.decision_vector:
            dv = self.state.decision_vector

            # Convert amplitudes to magnitude and phase
            amp_data = [
                {
                    "move": dv.moves[i] if i < len(dv.moves) else "",
                    "magnitude": abs(a),
                    "phase": np.angle(a),
                    "probability": dv.probabilities[i] if i < len(dv.probabilities) else 0,
                    "entanglement": (
                        dv.entanglement_scores[i] if i < len(dv.entanglement_scores) else 0
                    ),
                    "color": self._get_amplitude_color(a),
                }
                for i, a in enumerate(dv.amplitudes)
            ]

            decision_data = {
                "amplitudes": amp_data,
                "selected_move": dv.selected_move,
                "total_moves": len(dv.moves),
            }

        # History data
        amplitude_history = (
            self.state.amplitude_history[-50:] if self.state.amplitude_history else []
        )
        energy_history = self.state.energy_history[-50:] if self.state.energy_history else []

        return {
            "enabled": True,
            "width": self.width,
            "height": self.height,
            "visualization_mode": self.state.visualization_mode.value,
            "circuit": circuit_data,
            "qubo": qubo_data,
            "decision_vector": decision_data,
            "history": {
                "amplitudes": amplitude_history,
                "energy": energy_history,
            },
            "display_options": {
                "show_amplitudes": self.state.show_amplitudes,
                "show_phases": self.state.show_phases,
                "show_entanglement": self.state.show_entanglement,
                "show_qubo": self.state.show_qubo,
                "animate_transitions": self.state.animate_transitions,
                "color_by_phase": self.state.color_by_phase,
            },
            "colors": self.COLORS,
        }

    def to_json(self) -> str:
        """Serialize render data to JSON."""

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, complex):
                return {"real": obj.real, "imag": obj.imag}
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        return json.dumps(self.get_render_data(), default=convert)
