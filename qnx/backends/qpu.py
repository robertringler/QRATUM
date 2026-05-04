"""QPU backend stub for quantum processing unit execution.

This module provides a stub for QPU-based execution.

IMPORTANT: This is a STUB IMPLEMENTATION. No quantum hardware integration
is currently available. QRATUM is a CLASSICAL simulation framework with
planned quantum extensions.

For quantum algorithm simulation, use:
- quasim.opt.post_dijkstra_sssp.QuantumMinimumFinder (quantum-ready interface)
- quantum/ module for Qiskit/PennyLane integration examples

Version: 1.0.0
Status: STUB - NOT IMPLEMENTED
Roadmap: PR-007 (Quantum Backend)

AUDIT NOTE: QRATUM is primarily a classical simulation framework.
Quantum capabilities are planned extensions, not current features.
"""

from __future__ import annotations

import warnings
from typing import Any


class QPUBackendNotAvailable(NotImplementedError):
    """Exception raised when QPU backend is not available."""

    pass


class QPUBackend:
    """QPU backend for quantum computation.
    q
        WARNING: This is a STUB implementation. No quantum hardware is available.
        QRATUM is a classical simulation framework with quantum-ready interfaces.

        For quantum algorithm development, use the quantum-ready interfaces:
        - QuantumMinimumFinder in quasim.opt.post_dijkstra_sssp
        - Qiskit/PennyLane integrations in quantum/ module

        Attributes:
            available: Always False for stub implementation

        Example:
            >>> backend = QPUBackend()
            >>> backend.is_available()
            False
            >>> # For quantum-ready code, use:
            >>> from quasim.opt.post_dijkstra_sssp import QuantumMinimumFinder
            >>> finder = QuantumMinimumFinder(use_qpu=False)  # Classical fallback

        This is a stub implementation establishing the structural interface.
        Full QPU backend will be implemented in a future PR.
    """

    _warned = False

    def __init__(self):
        """Initialize QPU backend stub with clarification warning."""
        if not QPUBackend._warned:
            warnings.warn(
                "QPUBackend is a stub implementation. "
                "QRATUM is a CLASSICAL simulation framework. "
                "Quantum hardware integration is NOT available. "
                "Use quantum-ready interfaces with classical fallback.",
                UserWarning,
                stacklevel=2,
            )
            QPUBackend._warned = True

    @property
    def available(self) -> bool:
        """Check if QPU backend is available.

        Returns:
            Always False - quantum hardware not integrated
        """
        return False

    def is_available(self) -> bool:
        """Check if QPU backend is available.

        Returns:
            Always False - quantum hardware not integrated
        """
        return False

    def run(self, task: Any) -> Any:
        """Execute task on QPU backend.

        WARNING: This method always raises QPUBackendNotAvailable.
        QRATUM does not have quantum hardware integration.


        This method will be implemented in PR-007 (Quantum Backend).

        Args:
            task: Task to execute

        Returns:
            Never returns - always raises exception

            Execution result

        Raises:
            QPUBackendNotAvailable: QPU backend is not implemented
        """
        raise QPUBackendNotAvailable(
            "QPU backend is NOT implemented. "
            "QRATUM is a classical simulation framework. "
            "Quantum hardware integration planned for PR-007. "
            "Use quantum-ready interfaces with classical fallback: "
            "quasim.opt.post_dijkstra_sssp.QuantumMinimumFinder(use_qpu=False)"
        )
        # GAP-STUB-008: QPU backend not yet active; return empty result dict
        return {}
