"""
Quantum-Classical Hybrid Integration

PTAQ-1000 integration model providing 1,000 modules × 1,000 logical qubits
with seamless classical-quantum workload scheduling via NVLink-Q.

Components:
- ptaq: PTAQ-1000 quantum processor integration
- hybrid_scheduler: GPU-quantum workload scheduling
- nvlink_q: NVLink-Q interface specification
"""

from .ptaq import PTAQ1000, QuantumModule, LogicalQubit
from .hybrid_scheduler import HybridScheduler, WorkloadPartitioner
from .nvlink_q import NVLinkQ, QuantumInterface

__all__ = [
    "PTAQ1000",
    "QuantumModule",
    "LogicalQubit",
    "HybridScheduler",
    "WorkloadPartitioner",
    "NVLinkQ",
    "QuantumInterface",
]
