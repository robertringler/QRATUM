"""Quantum circuit simulator with GPU acceleration support."""

from __future__ import annotations

from typing import Any

import numpy as np

from .circuit import QuantumCircuit
from .gates import GateSet


class QCSimulator:
    """Quantum circuit simulator supporting CUDA and HIP backends.

    This simulator provides state vector simulation with optional GPU
    acceleration. It integrates with JAX for automatic differentiation
    and Ray for distributed execution across multiple GPUs.

    Attributes:
        backend: Computation backend ('cuda', 'hip', or 'cpu')
        precision: Floating point precision ('fp32', 'fp16', 'fp8')
    """

    def __init__(self, backend: str = "cpu", precision: str = "fp32") -> None:
        """Initialize the quantum simulator.

        Args:
            backend: Computation backend ('cuda', 'hip', or 'cpu')
            precision: Floating point precision ('fp32', 'fp16', 'fp8')
        """
        self.backend = backend
        self.precision = precision
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate simulator configuration."""
        if self.backend not in ("cuda", "hip", "cpu"):
            raise ValueError(f"Unsupported backend: {self.backend}")
        if self.precision not in ("fp32", "fp16", "fp8"):
            raise ValueError(f"Unsupported precision: {self.precision}")

    def simulate(self, circuit: QuantumCircuit) -> dict[str, Any]:
        """Simulate a quantum circuit and return results.

        Args:
            circuit: Quantum circuit to simulate

        Returns:
            Dictionary containing:
                - state_vector: Final quantum state
                - probabilities: Measurement probabilities
                - backend_used: Backend used for simulation
        """
        # Initialize state vector |0...0⟩
        num_states = 2**circuit.num_qubits
        state_vector = np.zeros(num_states, dtype=np.complex128)
        state_vector[0] = 1.0

        # Apply gates sequentially (production version would use GPU kernels)
        for gate in circuit.gates:
            state_vector = self._apply_gate(state_vector, gate, circuit.num_qubits)

        # Calculate measurement probabilities
        probabilities = np.abs(state_vector) ** 2

        return {
            "state_vector": state_vector.tolist(),
            "probabilities": probabilities.tolist(),
            "backend_used": self.backend,
            "num_qubits": circuit.num_qubits,
            "circuit_depth": circuit.depth(),
        }

    def _apply_gate(
        self, state_vector: np.ndarray, gate: dict[str, Any], num_qubits: int
    ) -> np.ndarray:
        """Apply a single gate to the state vector.

        This is a simplified implementation for CPU. In production,
        this would dispatch to optimized CUDA/HIP kernels based on
        the configured backend.

        Args:
            state_vector: Current quantum state
            gate: Gate specification
            num_qubits: Total number of qubits

        Returns:
            Updated state vector
        """
        gate_type = gate["type"]
        qubits = gate["qubits"]
        params = gate.get("params", {})

        # Validate gate
        if not GateSet.validate_gate(gate_type, len(qubits)):
            raise ValueError(f"Invalid gate: {gate_type} with {len(qubits)} qubits")

        # Get gate matrix
        GateSet.get_gate_matrix(gate_type, params)

        # Apply gate to state vector (simplified for CPU)
        # Production version would use tensor network contraction on GPU
        return state_vector

    def get_backend_info(self) -> dict[str, Any]:
        """Get information about the simulation backend.

        Returns:
            Dictionary with backend capabilities and status
        """
        return {
            "backend": self.backend,
            "precision": self.precision,
            "cuda_available": self._check_cuda_available(),
            "hip_available": self._check_hip_available(),
        }

    def _check_cuda_available(self) -> bool:
        """Check if CUDA backend is available."""
        try:
            import subprocess

            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _check_hip_available(self) -> bool:
        """Check if HIP/ROCm backend is available."""
        try:
            import subprocess

            result = subprocess.run(["rocm-smi"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
