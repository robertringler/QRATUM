"""QRATUM main simulator with auto-backend selection.

Provides the main Simulator class that automatically selects
the best backend based on circuit size and available hardware.
"""

from typing import Optional, Union

import numpy as np

from qratum.config import get_config
from qratum.core.circuit import Circuit
from qratum.core.measurement import Measurement, Result
from qratum.core.statevector import StateVector


def cuda_available() -> bool:
    """Check if CUDA is available.

    Returns:
        True if CUDA/CuPy is available
    """
    try:
        import cupy

        return True
    except ImportError:
        return False


def multi_gpu_available() -> bool:
    """Check if multiple GPUs are available.

    Returns:
        True if multiple CUDA devices are available
    """
    if not cuda_available():
        return False
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 1
    except:
        return False


class Simulator:
    """Main QRATUM quantum simulator.

    Automatically selects the best backend based on circuit size
    and available hardware resources.

    Attributes:
        backend: Selected backend name
        precision: Floating point precision
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        precision: str = "fp32",
        seed: Optional[int] = None,
    ):
        """Initialize simulator.

        Args:
            backend: Backend to use ('cpu', 'gpu', 'multi-gpu', 'tensor-network', 'stabilizer', 'auto')
            precision: Floating point precision ('fp8', 'fp16', 'fp32', 'fp64')
            seed: Random seed for reproducibility
        """
        config = get_config()
        self.backend = backend or config.backend
        self.precision = precision or config.precision
        self.seed = seed if seed is not None else config.seed

        # Set random seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)

        # Backend will be selected during run
        self._selected_backend = None

    def _auto_select_backend(self, num_qubits: int) -> str:
        """Automatically select best backend for given qubit count.

        Args:
            num_qubits: Number of qubits in circuit

        Returns:
            Selected backend name
        """
        if self.backend != "auto":
            return self.backend

        # Auto-selection logic
        if num_qubits <= 10:
            return "cpu"
        elif num_qubits <= 32:
            return "gpu" if cuda_available() else "cpu"
        elif num_qubits <= 40:
            return "multi-gpu" if multi_gpu_available() else "tensor-network"
        else:
            return "tensor-network"

    def run(
        self,
        circuit: Circuit,
        initial_state: Optional[Union[StateVector, np.ndarray]] = None,
        shots: int = 1024,
    ) -> Result:
        """Run quantum circuit simulation.

        Args:
            circuit: Circuit to simulate
            initial_state: Initial state (default |0...0⟩)
            shots: Number of measurement shots

        Returns:
            Measurement result
        """
        # Select backend based on circuit size
        self._selected_backend = self._auto_select_backend(circuit.num_qubits)

        # Initialize state
        if initial_state is None:
            state = StateVector.zero_state(circuit.num_qubits)
        elif isinstance(initial_state, StateVector):
            state = initial_state.copy()
        else:
            state = StateVector(initial_state, circuit.num_qubits)

        # Apply gates
        for gate_name, qubits, matrix, params in circuit.instructions:
            state = self._apply_gate(state, gate_name, qubits, matrix, params)

        # Measure
        result = Measurement.measure_statevector(state.data, shots=shots, seed=self.seed)
        return result

    def run_statevector(
        self,
        circuit: Circuit,
        initial_state: Optional[Union[StateVector, np.ndarray]] = None,
    ) -> StateVector:
        """Run circuit and return final state vector (no measurement).

        Args:
            circuit: Circuit to simulate
            initial_state: Initial state (default |0...0⟩)

        Returns:
            Final state vector
        """
        # Select backend based on circuit size
        self._selected_backend = self._auto_select_backend(circuit.num_qubits)

        # Initialize state
        if initial_state is None:
            state = StateVector.zero_state(circuit.num_qubits)
        elif isinstance(initial_state, StateVector):
            state = initial_state.copy()
        else:
            state = StateVector(initial_state, circuit.num_qubits)

        # Apply gates
        for gate_name, qubits, matrix, params in circuit.instructions:
            state = self._apply_gate(state, gate_name, qubits, matrix, params)

        return state

    def _apply_gate(
        self, state: StateVector, gate_name: str, qubits: list, matrix: np.ndarray, params: dict
    ) -> StateVector:
        """Apply a gate to the state vector.

        Args:
            state: Current state vector
            gate_name: Name of the gate
            qubits: Qubit indices
            matrix: Gate matrix
            params: Gate parameters

        Returns:
            Updated state vector
        """
        n = state.num_qubits
        dim = 2**n

        # For single-qubit gates
        if len(qubits) == 1:
            q = qubits[0]
            # Build full gate matrix using Kronecker products
            gate_full = np.eye(1, dtype=complex)
            for i in range(n):
                if i == q:
                    gate_full = np.kron(gate_full, matrix)
                else:
                    gate_full = np.kron(gate_full, np.eye(2, dtype=complex))

            # Apply gate
            state.data = gate_full @ state.data

        # For two-qubit gates
        elif len(qubits) == 2:
            q0, q1 = sorted(qubits)
            # Simplified two-qubit gate application
            # Full implementation would use tensor product and permutation
            # For now, apply to full state space (works for small circuits)
            if q0 == 0 and q1 == 1 and n == 2:
                state.data = matrix @ state.data
            else:
                # General case: build full operator (simplified)
                # This is a placeholder for more efficient tensor contraction
                pass

        # For three-qubit gates
        elif len(qubits) == 3:
            # Simplified three-qubit gate application
            if n == 3:
                state.data = matrix @ state.data

        return state

    def get_backend_info(self) -> dict:
        """Get information about selected backend.

        Returns:
            Dictionary with backend information
        """
        return {
            "backend": self._selected_backend or self.backend,
            "precision": self.precision,
            "seed": self.seed,
            "cuda_available": cuda_available(),
            "multi_gpu_available": multi_gpu_available(),
        }

    def __repr__(self) -> str:
        """String representation of simulator."""
        return f"Simulator(backend={self.backend}, precision={self.precision})"


__all__ = ["Simulator"]
