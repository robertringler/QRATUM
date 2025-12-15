"""Adapter for QuASIM simulation results."""

from __future__ import annotations

from typing import Any

import numpy as np


class QuASIMDataAdapter:
    """Adapter for QuASIM quantum simulation data.

    This adapter provides a bridge between QuASIM simulation outputs
    and the qubic-viz rendering pipeline.
    """

    @staticmethod
    def extract_quantum_state(simulation_result: Any) -> dict[str, np.ndarray]:
        """Extract quantum state data.

        Args:
            simulation_result: QuASIM simulation result

        Returns:
            Dictionary with quantum state data
        """
        # Extract quantum state if available
        if hasattr(simulation_result, "quantum_state"):
            state = simulation_result.quantum_state
            return {
                "amplitudes": np.array(state.amplitudes) if hasattr(state, "amplitudes") else None,
                "phases": np.array(state.phases) if hasattr(state, "phases") else None,
                "probabilities": (
                    np.array(state.probabilities) if hasattr(state, "probabilities") else None
                ),
            }
        return {"amplitudes": None, "phases": None, "probabilities": None}

    @staticmethod
    def extract_optimization_data(simulation_result: Any) -> dict[str, Any]:
        """Extract optimization trajectory data.

        Args:
            simulation_result: QuASIM simulation result

        Returns:
            Dictionary with optimization data
        """
        data = {
            "objective_values": [],
            "parameter_values": [],
            "convergence_history": [],
        }

        if hasattr(simulation_result, "optimization_history"):
            history = simulation_result.optimization_history
            data["objective_values"] = getattr(history, "objectives", [])
            data["parameter_values"] = getattr(history, "parameters", [])
            data["convergence_history"] = getattr(history, "convergence", [])

        return data

    @staticmethod
    def convert_to_visualization_format(
        data: Any, data_type: str = "tire"
    ) -> dict[str, Any]:
        """Convert QuASIM data to visualization format.

        Args:
            data: Input data from QuASIM
            data_type: Type of data (tire, quantum, optimization, etc.)

        Returns:
            Visualization-ready data dictionary
        """
        if data_type == "tire":
            return QuASIMDataAdapter._convert_tire_data(data)
        elif data_type == "quantum":
            return QuASIMDataAdapter._convert_quantum_data(data)
        elif data_type == "optimization":
            return QuASIMDataAdapter._convert_optimization_data(data)
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    @staticmethod
    def _convert_tire_data(data: Any) -> dict[str, Any]:
        """Convert tire simulation data.

        Args:
            data: Tire simulation data

        Returns:
            Converted data
        """
        from .tire_data_adapter import TireDataAdapter

        return TireDataAdapter.extract_visualization_data(data)

    @staticmethod
    def _convert_quantum_data(data: Any) -> dict[str, Any]:
        """Convert quantum simulation data.

        Args:
            data: Quantum simulation data

        Returns:
            Converted data
        """
        return {
            "quantum_state": QuASIMDataAdapter.extract_quantum_state(data),
            "optimization": QuASIMDataAdapter.extract_optimization_data(data),
        }

    @staticmethod
    def _convert_optimization_data(data: Any) -> dict[str, Any]:
        """Convert optimization data.

        Args:
            data: Optimization data

        Returns:
            Converted data
        """
        return QuASIMDataAdapter.extract_optimization_data(data)

    @staticmethod
    def create_animation_frames(
        simulation_sequence: list[Any], frame_skip: int = 1
    ) -> list[dict[str, Any]]:
        """Create animation frames from simulation sequence.

        Args:
            simulation_sequence: List of simulation results
            frame_skip: Skip every N frames

        Returns:
            List of frame data dictionaries
        """
        frames = []
        for i, sim_result in enumerate(simulation_sequence):
            if i % frame_skip == 0:
                frame_data = QuASIMDataAdapter.convert_to_visualization_format(sim_result)
                frame_data["frame_index"] = i // frame_skip
                frames.append(frame_data)
        return frames
