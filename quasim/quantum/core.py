"""Core quantum computing infrastructure for QRATUM.

This module provides backend setup, configuration, and common utilities
for quantum circuit simulation and execution.

Backends supported:
- Simulator: qasm_simulator (Qiskit Aer) - for development and testing
- Real Hardware: IBM Quantum backends (requires API token)

Note: Real quantum hardware execution requires:
1. IBM Quantum account: https://quantum-computing.ibm.com/
2. API token configuration
3. Understanding of queue times and cost considerations
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

# Handle optional dependencies
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumCircuit = Any
    AerSimulator = Any
    NoiseModel = Any


BackendType = Literal["simulator", "ibmq"]


@dataclass
class QuantumConfig:
    """Configuration for quantum computations.

    Attributes:
        backend_type: Type of backend ("simulator" or "ibmq")
        shots: Number of measurement shots (min 1000 for statistical validity)
        optimization_level: Qiskit transpiler optimization (0-3)
        seed: Random seed for reproducibility (simulator only)
        max_experiments: Maximum parallel circuit executions
        ibmq_token: IBM Quantum API token (required for real hardware)
        ibmq_backend_name: Name of IBM backend (e.g., "ibm_brisbane")
    """

    backend_type: BackendType = "simulator"
    shots: int = 1024
    optimization_level: int = 1
    seed: int = 42
    max_experiments: int = 10
    ibmq_token: str | None = None
    ibmq_backend_name: str | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.shots < 100:
            warnings.warn(
                f"shots={self.shots} is very low. Use at least 1000 for reliable statistics.",
                UserWarning,
            )

        if self.backend_type == "ibmq" and not self.ibmq_token:
            raise ValueError("ibmq_token required for IBM Quantum backend")

    @property
    def is_simulator(self) -> bool:
        """Check if using simulator backend."""
        return self.backend_type == "simulator"


class QuantumBackend:
    """Quantum backend wrapper for circuit execution.

    This class provides a unified interface for quantum circuit execution
    on both simulators and real quantum hardware.

    Example:
        >>> config = QuantumConfig(backend_type="simulator", shots=1024)
        >>> backend = QuantumBackend(config)
        >>> # Create and execute quantum circuit
        >>> circuit = QuantumCircuit(2)
        >>> circuit.h(0)
        >>> circuit.cx(0, 1)
        >>> circuit.measure_all()
        >>> result = backend.execute(circuit)
        >>> counts = result.get_counts()
    """

    def __init__(self, config: QuantumConfig):
        """Initialize quantum backend.

        Args:
            config: Quantum configuration

        Raises:
            ImportError: If Qiskit is not installed
            ValueError: If configuration is invalid
        """
        if not QISKIT_AVAILABLE:
            raise ImportError(
                "Qiskit is required for quantum computing. "
                "Install with: pip install qiskit qiskit-aer"
            )

        self.config = config
        self._backend = self._setup_backend()

    def _setup_backend(self) -> Any:
        """Setup the quantum backend based on configuration."""
        if self.config.backend_type == "simulator":
            return self._setup_simulator()
        elif self.config.backend_type == "ibmq":
            return self._setup_ibmq()
        else:
            raise ValueError(f"Unknown backend type: {self.config.backend_type}")

    def _setup_simulator(self) -> AerSimulator:
        """Setup Qiskit Aer simulator."""
        backend = AerSimulator()
        return backend

    def _setup_ibmq(self) -> Any:
        """Setup IBM Quantum backend.

        Requires IBM Quantum account and API token.
        """
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
        except ImportError:
            raise ImportError(
                "IBM Quantum access requires qiskit-ibm-runtime. "
                "Install with: pip install qiskit-ibm-runtime"
            )

        # Save account if token provided
        if self.config.ibmq_token:
            QiskitRuntimeService.save_account(
                channel="ibm_quantum", token=self.config.ibmq_token, overwrite=True
            )

        # Load service and get backend
        service = QiskitRuntimeService(channel="ibm_quantum")

        if self.config.ibmq_backend_name:
            backend = service.backend(self.config.ibmq_backend_name)
        else:
            # Get least busy backend
            backend = service.least_busy(operational=True, simulator=False)
            print(f"Selected backend: {backend.name}")

        return backend

    def execute(self, circuit: QuantumCircuit | list[QuantumCircuit], **kwargs: Any) -> Any:
        """Execute quantum circuit(s) on the backend.

        Args:
            circuit: Single circuit or list of circuits to execute
            **kwargs: Additional execution parameters

        Returns:
            Execution result with measurement counts and metadata
        """
        # Ensure circuit(s) have measurements
        circuits = [circuit] if isinstance(circuit, QuantumCircuit) else circuit

        # Transpile circuits for backend
        transpiled = transpile(
            circuits,
            backend=self._backend,
            optimization_level=self.config.optimization_level,
            seed_transpiler=self.config.seed if self.config.is_simulator else None,
        )

        # Execute
        if self.config.is_simulator:
            job = self._backend.run(
                transpiled, shots=self.config.shots, seed_simulator=self.config.seed, **kwargs
            )
        else:
            job = self._backend.run(transpiled, shots=self.config.shots, **kwargs)

        return job.result()

    @property
    def backend_name(self) -> str:
        """Get the name of the current backend."""
        return self._backend.name if hasattr(self._backend, "name") else "unknown"

    @property
    def num_qubits(self) -> int:
        """Get the number of qubits available on the backend."""
        if hasattr(self._backend, "num_qubits"):
            return self._backend.num_qubits
        # Simulator has no hard limit
        return 32


def create_noise_model(
    gate_error_1q: float = 0.001, gate_error_2q: float = 0.01, readout_error: float = 0.02
) -> NoiseModel | None:
    """Create a simple noise model for simulation.

    Args:
        gate_error_1q: Single-qubit gate error rate
        gate_error_2q: Two-qubit gate error rate
        readout_error: Measurement readout error rate

    Returns:
        Qiskit NoiseModel or None if Qiskit not available
    """
    if not QISKIT_AVAILABLE:
        warnings.warn("Qiskit not available, returning None", UserWarning)
        return None

    from qiskit_aer.noise import depolarizing_error

    noise_model = NoiseModel()

    # Single-qubit gates
    error_1q = depolarizing_error(gate_error_1q, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ["u1", "u2", "u3", "rx", "ry", "rz"])

    # Two-qubit gates
    error_2q = depolarizing_error(gate_error_2q, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cz", "swap"])

    # Readout error
    p0_1 = readout_error  # Prob of measuring 1 when state is 0
    p1_0 = readout_error  # Prob of measuring 0 when state is 1
    noise_model.add_all_qubit_readout_error([[1 - p0_1, p0_1], [p1_0, 1 - p1_0]])

    return noise_model


# Export main classes
__all__ = [
    "QuantumConfig",
    "QuantumBackend",
    "create_noise_model",
    "QISKIT_AVAILABLE",
]
