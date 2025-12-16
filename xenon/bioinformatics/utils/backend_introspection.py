"""Backend introspection for quantum systems.

Provides runtime capability detection and automatic downgrade paths.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class BackendType(Enum):
    """Types of computational backends."""

    CLASSICAL = "classical"
    QISKIT_AER = "qiskit_aer"
    HARDWARE_QPU = "hardware_qpu"


@dataclass
class BackendCapabilities:
    """Backend capability description.

    Attributes:
        backend_type: Type of backend
        max_qubits: Maximum number of qubits
        max_circuit_depth: Maximum circuit depth
        supports_noise: Whether noise models are supported
        gate_set: Supported quantum gates
        available: Whether backend is currently available
    """

    backend_type: BackendType
    max_qubits: int
    max_circuit_depth: int
    supports_noise: bool
    gate_set: List[str]
    available: bool


class BackendIntrospection:
    """Backend capability detection and introspection.

    Provides automatic detection of available quantum backends
    and automatic downgrade paths when requested backend unavailable.
    """

    def __init__(self):
        """Initialize backend introspection."""
        self.backends: Dict[BackendType, BackendCapabilities] = {}
        self._detect_backends()

    def _detect_backends(self) -> None:
        """Detect available backends."""
        # Classical backend always available
        self.backends[BackendType.CLASSICAL] = BackendCapabilities(
            backend_type=BackendType.CLASSICAL,
            max_qubits=1000,
            max_circuit_depth=1000000,
            supports_noise=False,
            gate_set=["all"],
            available=True,
        )

        # Try Qiskit
        try:
            import qiskit
            from qiskit_aer import Aer

            self.backends[BackendType.QISKIT_AER] = BackendCapabilities(
                backend_type=BackendType.QISKIT_AER,
                max_qubits=32,
                max_circuit_depth=10000,
                supports_noise=True,
                gate_set=["x", "y", "z", "h", "cx", "rx", "ry", "rz"],
                available=True,
            )
        except ImportError:
            self.backends[BackendType.QISKIT_AER] = BackendCapabilities(
                backend_type=BackendType.QISKIT_AER,
                max_qubits=0,
                max_circuit_depth=0,
                supports_noise=False,
                gate_set=[],
                available=False,
            )

    def get_backend(self, preferred: BackendType) -> BackendType:
        """Get best available backend.

        Args:
            preferred: Preferred backend type

        Returns:
            Best available backend (may downgrade)
        """
        if preferred in self.backends and self.backends[preferred].available:
            return preferred

        # Downgrade path: QPU -> Aer -> Classical
        if preferred == BackendType.HARDWARE_QPU:
            if self.backends.get(
                BackendType.QISKIT_AER,
                BackendCapabilities(BackendType.QISKIT_AER, 0, 0, False, [], False),
            ).available:
                warnings.warn("Hardware QPU not available, downgrading to Qiskit Aer", UserWarning)
                return BackendType.QISKIT_AER

        # Final fallback to classical
        warnings.warn(f"{preferred.value} not available, using classical fallback", UserWarning)
        return BackendType.CLASSICAL

    def log_execution(
        self,
        backend: BackendType,
        circuit_depth: int,
        gate_count: int,
        execution_time: float,
    ) -> Dict[str, any]:
        """Log execution metrics.

        Args:
            backend: Backend used
            circuit_depth: Circuit depth
            gate_count: Number of gates
            execution_time: Execution time in seconds

        Returns:
            Execution log entry
        """
        return {
            "backend": backend.value,
            "circuit_depth": circuit_depth,
            "gate_count": gate_count,
            "execution_time_s": execution_time,
            "gates_per_second": gate_count / execution_time if execution_time > 0 else 0,
        }
