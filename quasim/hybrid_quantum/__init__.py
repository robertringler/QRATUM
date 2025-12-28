"""QRATUM Hybrid Quantum-Classical Integration Module.

This module provides safe quantum backend integration with QRATUM's trust-preserving
architecture. All quantum operations are wrapped in provenance tracking, verification,
and rollback capabilities.

Core Principles:
- Quantum outputs **never bypass** verification, provenance, or rollback
- All quantum use is **proposal-only** until approved
- No uncontrolled quantum advantage - speedups bounded by jurisdictional invariants
- Trust is a conserved invariant

Supported Backends:
- IBM Quantum (Qiskit Runtime)
- IonQ (via Amazon Braket or direct API)
- Quantinuum (via Azure Quantum)
- Azure Quantum (multiple providers)
- AWS Braket (simulators and hardware)

Architecture:
- Each backend wrapper inherits from HybridQuantumBackend ABC
- All executions go through QuantumProvenanceWrapper for tracking
- Results require QuantumVerifier approval before being used
- RollbackManager enables deterministic state recovery
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "HybridQuantumBackend",
    "QuantumProvenanceWrapper",
    "QuantumVerifier",
    "RollbackManager",
    "IBMHybridBackend",
    "IonQHybridBackend",
    "QuantinuumHybridBackend",
    "AzureQuantumHybridBackend",
    "BraketHybridBackend",
    "HybridQuantumConfig",
    "ProvenanceRecord",
    "VerificationResult",
    "check_hybrid_backends",
]

# Check for optional dependencies
try:
    from qiskit_ibm_runtime import QiskitRuntimeService

    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False

try:
    from braket.aws import AwsDevice

    BRAKET_AVAILABLE = True
except ImportError:
    BRAKET_AVAILABLE = False

try:
    from azure.quantum import Workspace

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False


def check_hybrid_backends() -> dict[str, bool]:
    """Check which hybrid quantum backends are available.

    Returns:
        Dictionary mapping backend names to availability status
    """
    return {
        "ibm": IBM_AVAILABLE,
        "braket": BRAKET_AVAILABLE,
        "azure": AZURE_AVAILABLE,
        "ionq": BRAKET_AVAILABLE,  # IonQ accessible via Braket
        "quantinuum": AZURE_AVAILABLE,  # Quantinuum accessible via Azure
    }


# Import core components (defined below in submodules)
from .backends import (
    AzureQuantumHybridBackend,
    BraketHybridBackend,
    HybridQuantumBackend,
    HybridQuantumConfig,
    IBMHybridBackend,
    IonQHybridBackend,
    QuantinuumHybridBackend,
)
from .provenance import ProvenanceRecord, QuantumProvenanceWrapper
from .rollback import RollbackManager
from .verification import QuantumVerifier, VerificationResult
