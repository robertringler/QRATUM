"""
Quantum-Classical Hybrid Integration

PTAQ-1000 integration model providing 1,000 modules × 1,000 logical qubits
with seamless classical-quantum workload scheduling via NVLink-Q.

Components:
- ptaq: PTAQ-1000 quantum processor integration
- hybrid_scheduler: GPU-quantum workload scheduling
- nvlink_q: NVLink-Q interface specification
"""

from .hybrid_scheduler import (
    ClassicalTask,
    HybridScheduler,
    HybridSchedulerInterface,
    HybridWorkload,
    QuantumTask,
    SchedulingPolicy,
    WorkloadType,
)
from .nvlink_q import (
    NVLinkQFabric,
    NVLinkQInterface,
    NVLinkQInterfaceBuilder,
    NVLinkQPacket,
    NVLinkQProtocol,
    NVLinkQTransfer,
)
from .ptaq import (
    PTAQ1000,
    GateType,
    PTAQInterface,
    QuantumCircuit,
    QuantumJob,
    QuantumModule,
)

__all__ = [
    # PTAQ-1000
    "PTAQ1000",
    "QuantumModule",
    "QuantumCircuit",
    "QuantumJob",
    "PTAQInterface",
    "GateType",
    # Hybrid Scheduler
    "HybridScheduler",
    "HybridWorkload",
    "ClassicalTask",
    "QuantumTask",
    "WorkloadType",
    "SchedulingPolicy",
    "HybridSchedulerInterface",
    # NVLink-Q
    "NVLinkQInterface",
    "NVLinkQFabric",
    "NVLinkQPacket",
    "NVLinkQTransfer",
    "NVLinkQProtocol",
    "NVLinkQInterfaceBuilder",
]
