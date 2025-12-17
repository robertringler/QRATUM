"""Quantum simulation adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from qubic.visualization.adapters.base import SimulationAdapter
from qubic.visualization.core.data_model import VisualizationData


class QuantumSimulationAdapter(SimulationAdapter):
    """Adapter for quantum circuit simulation data.

    Supports visualization of:
    - Quantum state amplitudes
    - Probability distributions
    - Qubit correlations
    - Circuit execution results
    """

    def load_data(self, source: dict[str, Any] | np.ndarray) -> VisualizationData:
        """Load quantum simulation data.

        Args:
            source: Either a dictionary with quantum state data or
                   a numpy array of state amplitudes

        Returns:
            VisualizationData for quantum amplitude visualization

        Raises:
            ValueError: If source format is invalid
        """

        if isinstance(source, np.ndarray):
            return self._create_amplitude_bars(source)
        elif isinstance(source, dict):
            if "amplitudes" in source:
                return self._create_amplitude_bars(source["amplitudes"], metadata=source)
            else:
                raise ValueError("Dictionary must contain 'amplitudes' key")
        else:
            raise ValueError(
                f"Unsupported source type: {type(source)}. Expected numpy array or dictionary."
            )

    def _create_amplitude_bars(
        self, amplitudes: np.ndarray, metadata: dict[str, Any] | None = None
    ) -> VisualizationData:
        """Create 3D bar chart for amplitude visualization.

        Args:
            amplitudes: Complex amplitudes of quantum state
            metadata: Optional metadata about the quantum state

        Returns:
            VisualizationData representing amplitude bars
        """

        n_states = len(amplitudes)

        # Create bar geometry for each amplitude
        vertices_list = []
        faces_list = []
        probabilities = []

        for i, amp in enumerate(amplitudes):
            # Probability (height of bar)
            prob = np.abs(amp) ** 2

            # Position along x-axis
            x_pos = i - n_states / 2

            # Create rectangular bar (simplified as two triangles forming a quad)
            bar_width = 0.8
            base_vertices = np.array(
                [
                    [x_pos - bar_width / 2, 0, -bar_width / 2],
                    [x_pos + bar_width / 2, 0, -bar_width / 2],
                    [x_pos + bar_width / 2, 0, bar_width / 2],
                    [x_pos - bar_width / 2, 0, bar_width / 2],
                    [x_pos - bar_width / 2, prob, -bar_width / 2],
                    [x_pos + bar_width / 2, prob, -bar_width / 2],
                    [x_pos + bar_width / 2, prob, bar_width / 2],
                    [x_pos - bar_width / 2, prob, bar_width / 2],
                ]
            )

            v_offset = len(vertices_list)
            vertices_list.extend(base_vertices)

            # Faces for the bar (12 triangles = 6 faces * 2 triangles each)
            bar_faces = [
                # Bottom
                [v_offset + 0, v_offset + 1, v_offset + 2],
                [v_offset + 0, v_offset + 2, v_offset + 3],
                # Top
                [v_offset + 4, v_offset + 6, v_offset + 5],
                [v_offset + 4, v_offset + 7, v_offset + 6],
                # Front
                [v_offset + 0, v_offset + 5, v_offset + 1],
                [v_offset + 0, v_offset + 4, v_offset + 5],
                # Back
                [v_offset + 2, v_offset + 7, v_offset + 3],
                [v_offset + 2, v_offset + 6, v_offset + 7],
                # Left
                [v_offset + 0, v_offset + 7, v_offset + 4],
                [v_offset + 0, v_offset + 3, v_offset + 7],
                # Right
                [v_offset + 1, v_offset + 5, v_offset + 6],
                [v_offset + 1, v_offset + 6, v_offset + 2],
            ]

            faces_list.extend(bar_faces)

            # Store probability for each vertex of this bar
            probabilities.extend([prob] * 8)

        vertices = np.array(vertices_list)
        faces = np.array(faces_list)
        probabilities = np.array(probabilities)

        # Phase information
        phases = np.angle(amplitudes)
        phase_field = np.repeat(phases, 8)  # One phase per bar (8 vertices each)

        vis_metadata = {
            "adapter_type": "quantum",
            "n_states": n_states,
            "visualization_type": "amplitude_bars",
        }
        if metadata:
            vis_metadata.update(metadata)

        return VisualizationData(
            vertices=vertices,
            faces=faces,
            scalar_fields={"probability": probabilities, "phase": phase_field},
            metadata=vis_metadata,
        )

    def validate_source(self, source: Any) -> bool:
        """Validate quantum simulation data source.

        Args:
            source: Data source to validate

        Returns:
            True if source is valid for this adapter
        """

        if isinstance(source, np.ndarray):
            # Should be 1D array of complex numbers
            return source.ndim == 1 and np.iscomplexobj(source)
        elif isinstance(source, dict):
            return "amplitudes" in source
        return False

    def create_synthetic_state(
        self, n_qubits: int = 3, state_type: str = "superposition"
    ) -> VisualizationData:
        """Create synthetic quantum state for testing.

        Args:
            n_qubits: Number of qubits
            state_type: Type of state ('superposition', 'entangled', 'basis')

        Returns:
            VisualizationData with synthetic quantum state
        """

        n_states = 2**n_qubits

        if state_type == "superposition":
            # Equal superposition with random phases
            amplitudes = np.exp(1j * np.random.uniform(0, 2 * np.pi, n_states))
            amplitudes = amplitudes / np.sqrt(n_states)

        elif state_type == "entangled":
            # Bell-like state
            amplitudes = np.zeros(n_states, dtype=complex)
            amplitudes[0] = 1 / np.sqrt(2)
            amplitudes[-1] = 1 / np.sqrt(2)

        elif state_type == "basis":
            # Single basis state
            amplitudes = np.zeros(n_states, dtype=complex)
            amplitudes[0] = 1.0

        else:
            raise ValueError(f"Unknown state type: {state_type}")

        metadata = {"n_qubits": n_qubits, "state_type": state_type}

        return self.load_data({"amplitudes": amplitudes, **metadata})
