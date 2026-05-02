"""QRATUM Hybrid Quantum-Classical Integration Module.

This module provides safe quantum backend integration with QRATUM's trust-preserving
architecture. All quantum operations are wrapped in provenance tracking, verification,
and rollback capabilities.

Core Principles:
- Quantum outputs **never bypass** verification, provenance, or rollback
- All quantum use is **proposal-only** until approved
- No uncontrolled quantum advantage - speedups bounded by jurisdictional invariants
- Trust is a conserved invariant: ℛ(t) ≥ 0
- Diagnostics read-only; resolutions proposal-only
- Forever non-agentic epistemic compiler

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
- HybridQuantumOrchestrator coordinates quantum-classical workflows
- EnhancedTopologicalObserver provides read-only diagnostics
- ReinjectionEvaluationEngine manages proposal-only reinjection

Priority Enhancement Clusters:
- P0: Quantum-Classical Hybrid Speed (≥10× speedup, zk-proof ≤5s)
- P1: Epistemic Perfection (100% verifiable, ℛ(t) variance ≤0.001)
- P1: Scientific Substrate (fidelity ≥0.999, collapse_index ≥30% reduction)
- P2: Distributed Fortress (finality <1s, ≥10k TXO/s)
- P2: Human Jurisdiction Interface (proposal cycle ≤48h)
- P3: Operational Anti-Entropy (uptime 99.999%, cost ≤$0.01/op)
"""

from __future__ import annotations

__version__ = "1.1.0"
__all__ = [
    # Backends
    "HybridQuantumBackend",
    "IBMHybridBackend",
    "IonQHybridBackend",
    "QuantinuumHybridBackend",
    "AzureQuantumHybridBackend",
    "BraketHybridBackend",
    "HybridQuantumConfig",
    # Provenance
    "QuantumProvenanceWrapper",
    "ProvenanceRecord",
    # Verification
    "QuantumVerifier",
    "VerificationResult",
    "TopologicalDiagnosticObserver",
    # Rollback
    "RollbackManager",
    "DualApprovalGate",
    # Orchestrator
    "HybridQuantumOrchestrator",
    "TrustMetric",
    "FallbackStrategy",
    "ExecutionContext",
    "OrchestratorStatus",
    "ExecutionMode",
    "FailureType",
    "QuantumVerificationError",
    # Enhanced Topological Observer
    "EnhancedTopologicalObserver",
    "TopologicalObservation",
    "CollapseMetrics",
    "FidelityMetrics",
    "DiagnosticFinding",
    "DiagnosticSeverity",
    "DiagnosticCategory",
    # Reinjection Engine
    "ReinjectionEvaluationEngine",
    "ProposalArtifact",
    "ProposalStatus",
    "ProposalCluster",
    "PreValidationScore",
    "MerkleTreeBuilder",
    # Utilities
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

# Import orchestrator components
from .orchestrator import (
    ExecutionContext,
    ExecutionMode,
    FailureType,
    FallbackStrategy,
    HybridQuantumOrchestrator,
    OrchestratorStatus,
    QuantumVerificationError,
    TrustMetric,
)
from .provenance import ProvenanceRecord, QuantumProvenanceWrapper

# Import reinjection engine
from .reinjection_engine import (
    MerkleTreeBuilder,
    PreValidationScore,
    ProposalArtifact,
    ProposalCluster,
    ProposalStatus,
    ReinjectionEvaluationEngine,
)
from .rollback import DualApprovalGate, RollbackManager

# Import enhanced topological observer
from .topological_observer import (
    CollapseMetrics,
    DiagnosticCategory,
    DiagnosticFinding,
    DiagnosticSeverity,
    EnhancedTopologicalObserver,
    FidelityMetrics,
    TopologicalObservation,
)
from .verification import QuantumVerifier, TopologicalDiagnosticObserver, VerificationResult
