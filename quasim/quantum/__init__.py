"""QRATUM Quantum Computing Module.

This module provides genuine quantum computing capabilities using Qiskit and PennyLane.
It implements validated quantum algorithms for materials science applications.

IMPORTANT: This is a NISQ-era (Noisy Intermediate-Scale Quantum) implementation.
All quantum computations are:
- Probabilistic (require multiple shots for statistics)
- Limited to small systems (~10-50 qubits effectively)
- Subject to noise and error rates
- Validated against classical methods

Current Capabilities:
- VQE (Variational Quantum Eigensolver) for molecular ground states
- QAOA (Quantum Approximate Optimization Algorithm) for combinatorial optimization
- Quantum circuit simulation on classical hardware
- Basic error mitigation techniques

See individual modules for detailed documentation and limitations.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["core", "vqe_molecule", "qaoa_optimization"]

# Quantum module is optional - gracefully handle missing dependencies
try:
    import qiskit

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    import pennylane

    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


def check_quantum_dependencies() -> dict[str, bool]:
    """Check which quantum computing libraries are available.

    Returns:
        Dictionary mapping library names to availability status
    """

    return {
        "qiskit": QISKIT_AVAILABLE,
        "pennylane": PENNYLANE_AVAILABLE,
    }


def get_quantum_status() -> str:
    """Get a human-readable status of quantum capabilities.

    Returns:
        Status string describing available quantum features
    """

    deps = check_quantum_dependencies()

    if not any(deps.values()):
        return "No quantum libraries available. Install with: pip install qiskit pennylane"

    available = [name for name, avail in deps.items() if avail]
    return f"Quantum computing enabled with: {', '.join(available)}"
