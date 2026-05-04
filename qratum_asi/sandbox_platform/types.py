"""Type definitions for the Sandboxed Autonomous Action Platform.

Defines core types for sandboxed execution including configurations,
proposals, results, and resource allocations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ExecutionMode(Enum):
    """Mode of sandbox execution."""

    SYNC = "sync"  # Synchronous execution
    ASYNC = "async"  # Asynchronous execution
    BATCH = "batch"  # Batch evaluation
    SPECULATIVE = "speculative"  # Speculative pre-execution
    LAZY = "lazy"  # Lazy evaluation


class IsolationLevel(Enum):
    """Level of sandbox isolation."""

    MINIMAL = "minimal"  # Thread isolation
    PROCESS = "process"  # Process isolation
    CONTAINER = "container"  # Container isolation
    FULL = "full"  # Full containerized isolation with network separation


class ProposalPriority(Enum):
    """Priority level for proposal evaluation."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ResourceType(Enum):
    """Types of computational resources."""

    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    QUANTUM = "quantum"


@dataclass
class ResourceAllocation:
    """Resource allocation for sandbox execution.

    Attributes:
        cpu_cores: Number of CPU cores
        memory_mb: Memory in megabytes
        gpu_units: GPU compute units (0 for CPU-only)
        quantum_qubits: Quantum qubits allocated (0 for classical-only)
        max_duration_ms: Maximum execution duration in milliseconds
        priority: Resource priority level
    """

    cpu_cores: int = 1
    memory_mb: int = 512
    gpu_units: float = 0.0
    quantum_qubits: int = 0
    max_duration_ms: int = 30000
    priority: ProposalPriority = ProposalPriority.NORMAL

    def to_dict(self) -> dict[str, Any]:
        """Serialize resource allocation."""
        return {
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "gpu_units": self.gpu_units,
            "quantum_qubits": self.quantum_qubits,
            "max_duration_ms": self.max_duration_ms,
            "priority": self.priority.value,
        }


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution.

    Attributes:
        sandbox_id: Unique sandbox identifier
        isolation_level: Level of isolation
        execution_mode: Mode of execution
        resources: Resource allocation
        enable_merkle_verification: Whether to enable Merkle verification
        enable_metrics: Whether to enable passive metrics
        auto_destroy: Whether to auto-destroy after evaluation
        deterministic_mode: Whether to use deterministic stubs
    """

    sandbox_id: str
    isolation_level: IsolationLevel = IsolationLevel.CONTAINER
    execution_mode: ExecutionMode = ExecutionMode.ASYNC
    resources: ResourceAllocation = field(default_factory=ResourceAllocation)
    enable_merkle_verification: bool = True
    enable_metrics: bool = True
    auto_destroy: bool = True
    deterministic_mode: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration."""
        return {
            "sandbox_id": self.sandbox_id,
            "isolation_level": self.isolation_level.value,
            "execution_mode": self.execution_mode.value,
            "resources": self.resources.to_dict(),
            "enable_merkle_verification": self.enable_merkle_verification,
            "enable_metrics": self.enable_metrics,
            "auto_destroy": self.auto_destroy,
            "deterministic_mode": self.deterministic_mode,
            "created_at": self.created_at,
        }


@dataclass
class SandboxProposal:
    """Proposal for sandboxed evaluation.

    Attributes:
        proposal_id: Unique proposal identifier
        source_id: Source system/workflow identifier
        proposal_type: Type of proposal
        payload: Proposal data payload
        priority: Evaluation priority
        target_subsystems: Target subsystems to affect
        estimated_impact: Estimated impact score (0-1)
        merkle_proof: Merkle proof from source
    """

    proposal_id: str
    source_id: str
    proposal_type: str
    payload: dict[str, Any]
    priority: ProposalPriority = ProposalPriority.NORMAL
    target_subsystems: list[str] = field(default_factory=list)
    estimated_impact: float = 0.5
    merkle_proof: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute deterministic hash of proposal content."""
        content = {
            "proposal_id": self.proposal_id,
            "source_id": self.source_id,
            "proposal_type": self.proposal_type,
            "payload": self.payload,
            "target_subsystems": self.target_subsystems,
        }
        return hashlib.sha3_256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize proposal."""
        return {
            "proposal_id": self.proposal_id,
            "source_id": self.source_id,
            "proposal_type": self.proposal_type,
            "payload": self.payload,
            "priority": self.priority.value,
            "target_subsystems": self.target_subsystems,
            "estimated_impact": self.estimated_impact,
            "merkle_proof": self.merkle_proof,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "content_hash": self.compute_hash(),
        }


@dataclass
class SandboxEvaluationResult:
    """Result of sandbox evaluation.

    Attributes:
        result_id: Unique result identifier
        proposal_id: Source proposal ID
        sandbox_id: Sandbox that performed evaluation
        success: Whether evaluation succeeded
        outcome: Evaluation outcome/recommendation
        fidelity_score: Fidelity of evaluation (0-1)
        side_effects: Detected side effects
        metrics: Collected metrics during evaluation
        execution_time_ms: Execution time in milliseconds
        merkle_proof: Merkle proof of result
        approved: Whether proposal was approved for production
        approval_status: Detailed approval status
    """

    result_id: str
    proposal_id: str
    sandbox_id: str
    success: bool
    outcome: str
    fidelity_score: float
    side_effects: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    merkle_proof: str = ""
    approved: bool = False
    approval_status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def compute_hash(self) -> str:
        """Compute deterministic hash of result content."""
        content = {
            "result_id": self.result_id,
            "proposal_id": self.proposal_id,
            "sandbox_id": self.sandbox_id,
            "success": self.success,
            "outcome": self.outcome,
            "fidelity_score": self.fidelity_score,
            "side_effects": self.side_effects,
        }
        return hashlib.sha3_256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize result."""
        return {
            "result_id": self.result_id,
            "proposal_id": self.proposal_id,
            "sandbox_id": self.sandbox_id,
            "success": self.success,
            "outcome": self.outcome,
            "fidelity_score": self.fidelity_score,
            "side_effects": self.side_effects,
            "metrics": self.metrics,
            "execution_time_ms": self.execution_time_ms,
            "merkle_proof": self.merkle_proof,
            "approved": self.approved,
            "approval_status": self.approval_status,
            "created_at": self.created_at,
            "result_hash": self.compute_hash(),
        }


@dataclass
class MetricsSnapshot:
    """Snapshot of sandbox metrics at a point in time.

    Attributes:
        snapshot_id: Unique snapshot identifier
        sandbox_id: Sandbox being measured
        latency_ms: Current latency in milliseconds
        throughput_ops: Operations per second
        entropy: Current entropy measure
        topological_index: Topological complexity index
        memory_usage_mb: Memory usage in MB
        cpu_utilization: CPU utilization (0-1)
        queue_depth: Current queue depth
    """

    snapshot_id: str
    sandbox_id: str
    latency_ms: float = 0.0
    throughput_ops: float = 0.0
    entropy: float = 0.0
    topological_index: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0
    queue_depth: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics snapshot."""
        return {
            "snapshot_id": self.snapshot_id,
            "sandbox_id": self.sandbox_id,
            "latency_ms": self.latency_ms,
            "throughput_ops": self.throughput_ops,
            "entropy": self.entropy,
            "topological_index": self.topological_index,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_utilization": self.cpu_utilization,
            "queue_depth": self.queue_depth,
            "timestamp": self.timestamp,
        }
