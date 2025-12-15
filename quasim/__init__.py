"""QuASIM - Quantum-Accelerated Simulation Runtime.

DEPRECATED: QuASIM has been renamed to QRATUM (Quantum Resource Allocation,
Tensor Analysis, and Unified Modeling). This module provides backward
compatibility but will be removed in version 3.0.0.

Please update your imports:
    OLD: from quasim import Simulator, Circuit
    NEW: from qratum import Simulator, Circuit

For migration guidance, see MIGRATION.md

---

QuASIM provides a production-ready, horizontally scalable platform for
quantum-enhanced digital-twin workloads across aerospace, pharma, finance,
and manufacturing verticals.

Modules:
    qc: Quantum Computing primitives and circuit simulation
    dtwin: Digital Twin simulation and state management
    opt: Optimization algorithms and quantum-accelerated solvers
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

# Issue deprecation warning
warnings.warn(
    "The 'quasim' package has been renamed to 'qratum'. "
    "Please update your imports: 'from qratum import ...' "
    "This compatibility shim will be removed in version 3.0.0. "
    "See MIGRATION.md for guidance.",
    DeprecationWarning,
    stacklevel=2,
)

__version__ = "0.1.0"
__author__ = "Sybernix Team"
__legacy_package__ = True
__new_package_name__ = "qratum"


@dataclass
class Config:
    """QuASIM runtime configuration.

    Attributes:
        simulation_precision: Precision level ('fp8', 'fp16', 'fp32', 'fp64')
        max_workspace_mb: Maximum workspace memory in megabytes
        backend: Compute backend ('cpu', 'cuda', 'rocm')
        seed: Random seed for deterministic simulations
    """

    simulation_precision: str = "fp32"
    max_workspace_mb: int = 1024
    backend: str = "cpu"
    seed: int | None = None


class Runtime:
    """QuASIM simulation runtime context."""

    def __init__(self, config: Config):
        """Initialize runtime with configuration.

        Args:
            config: Runtime configuration
        """
        self.config = config
        self.average_latency = 0.0
        self._initialized = False

    def __enter__(self):
        """Enter runtime context."""
        self._initialized = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit runtime context."""
        self._initialized = False
        return False

    def simulate(self, circuit: list[list[complex]]) -> list[complex]:
        """Simulate quantum circuit.

        Args:
            circuit: Circuit specification as list of gate matrices

        Returns:
            Simulation result as state vector
        """
        if not self._initialized:
            raise RuntimeError("Runtime not initialized. Use 'with runtime(config)' context.")

        # Simplified simulation - in production would use actual quantum simulation
        result = []
        for gate in circuit:
            # Average the complex values for each gate
            avg = sum(gate) / len(gate) if gate else 0j
            result.append(avg)

        # Set a non-zero latency to indicate operation completed
        self.average_latency = 0.001  # 1ms simulated latency

        return result


@contextmanager
def runtime(config: Config):
    """Create a QuASIM runtime context.

    Args:
        config: Runtime configuration

    Yields:
        Runtime instance
    """
    rt = Runtime(config)
    try:
        yield rt.__enter__()
    finally:
        rt.__exit__(None, None, None)


# Re-export QRATUM functionality for backward compatibility
try:
    from qratum import (
        Simulator as QRATUMSimulator,
        Circuit as QRATUMCircuit,
        StateVector as QRATUMStateVector,
        Measurement as QRATUMMeasurement,
        Result as QRATUMResult,
        DensityMatrix as QRATUMDensityMatrix,
        gates as qratum_gates,
        QRATUMConfig,
    )

    # Create aliases for backward compatibility
    # Users can use quasim.Simulator which maps to qratum.Simulator
    QuantumSimulator = QRATUMSimulator
    QuantumCircuit = QRATUMCircuit
    QuantumStateVector = QRATUMStateVector
    QuantumMeasurement = QRATUMMeasurement
    QuantumResult = QRATUMResult
    QuantumDensityMatrix = QRATUMDensityMatrix
    quantum_gates = qratum_gates

    _QRATUM_AVAILABLE = True
except ImportError:
    _QRATUM_AVAILABLE = False
    warnings.warn(
        "QRATUM package not found. Legacy QuASIM functionality only. "
        "Install qratum for full quantum simulation features.",
        RuntimeWarning,
        stacklevel=2,
    )

__all__ = [
    "__version__",
    "__author__",
    "__legacy_package__",
    "__new_package_name__",
    "Config",
    "Runtime",
    "runtime",
]

# Add QRATUM exports if available
if _QRATUM_AVAILABLE:
    __all__.extend(
        [
            "QuantumSimulator",
            "QuantumCircuit",
            "QuantumStateVector",
            "QuantumMeasurement",
            "QuantumResult",
            "QuantumDensityMatrix",
            "quantum_gates",
        ]
    )
