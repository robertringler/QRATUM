"""QRATUM Sandboxed Autonomous Action Platform.

Implements performance-optimized sandboxed execution for autonomous action
evaluation while preserving production stability, trust, and invariants.

Core Principles:
- Sandbox is a detached advisory layer
- Fully deterministic, auditable, and proposal-only
- Does not block, throttle, or interfere with production operations
- Trust, invariants, and throughput preserved simultaneously

Key Features:
1. Decoupled Execution - Isolated memory, Merkle-verified communication
2. Parallelism - Sharded workloads, batched evaluation
3. Lightweight Deterministic Evaluation - Ephemeral containers, deterministic stubs
4. Observability - Passive read-only metrics, offloaded diagnostics
5. Non-blocking Proposal Integration - Async queuing, dual-control approval
6. Optional Quantum/High-Performance Layer - Hybrid quantum sandboxing
7. Optimization Techniques - Lazy evaluation, speculative execution, throttling

Version: 1.0.0
Status: Production Ready
QuASIM: v2025.12.28
"""

# Core sandbox types
from qratum_asi.sandbox_platform.types import (
    SandboxConfig,
    SandboxProposal,
    SandboxEvaluationResult,
    ExecutionMode,
    IsolationLevel,
    ResourceAllocation,
    MetricsSnapshot,
    ProposalPriority,
)

# Isolated execution
from qratum_asi.sandbox_platform.isolated_sandbox import (
    IsolatedSandboxExecutor,
    SandboxContainer,
    MemoryIsolation,
)

# Merkle verification
from qratum_asi.sandbox_platform.merkle_verifier import (
    MerkleVerifiedChannel,
    VerificationResult,
    AuditChainLogger,
)

# Async pipeline
from qratum_asi.sandbox_platform.async_pipeline import (
    AsyncEvaluationPipeline,
    PipelineStage,
    NonBlockingQueue,
)

# Parallelism
from qratum_asi.sandbox_platform.sharded_executor import (
    ShardedSandboxExecutor,
    WorkloadShard,
    NodeAllocation,
)

from qratum_asi.sandbox_platform.batch_evaluator import (
    BatchProposalEvaluator,
    BatchResult,
    EvaluationBatch,
)

# Lightweight evaluation
from qratum_asi.sandbox_platform.ephemeral_container import (
    EphemeralContainer,
    ContainerLifecycle,
    AutoDestroyPolicy,
)

from qratum_asi.sandbox_platform.deterministic_stubs import (
    DeterministicStubRegistry,
    ComputationStub,
    StubFidelityLevel,
)

from qratum_asi.sandbox_platform.incremental_state import (
    IncrementalStateEvaluator,
    StateDelta,
    StateCheckpoint,
)

# Observability
from qratum_asi.sandbox_platform.passive_metrics import (
    PassiveMetricsCollector,
    LatencyMetrics,
    ThroughputMetrics,
    EntropyMetrics,
    TopologicalIndices,
)

from qratum_asi.sandbox_platform.diagnostics_offload import (
    DiagnosticsOffloader,
    DiagnosticJob,
    OffloadTarget,
)

# Non-blocking integration
from qratum_asi.sandbox_platform.proposal_queue import (
    ProposalQueue,
    QueuedProposal,
    QueuePriority,
)

from qratum_asi.sandbox_platform.dual_control_gateway import (
    DualControlGateway,
    ApprovalRequest,
    AuthorizationStatus,
)

# Optimization
from qratum_asi.sandbox_platform.lazy_evaluator import (
    LazyEvaluator,
    EvaluationPolicy,
    CriticalityAssessment,
)

from qratum_asi.sandbox_platform.speculative_executor import (
    SpeculativeExecutor,
    SpeculativeResult,
    LikelihoodEstimator,
)

from qratum_asi.sandbox_platform.resource_throttler import (
    ResourceThrottler,
    ThrottlePolicy,
    LoadMetrics,
)

# Quantum/High-performance (optional)
from qratum_asi.sandbox_platform.quantum_sandbox import (
    QuantumSandbox,
    HybridExecutor,
    QuantumIsolation,
)

from qratum_asi.sandbox_platform.tensor_network import (
    TensorNetworkPrecomputer,
    TensorCache,
    HybridComputation,
)

# Main orchestrator
from qratum_asi.sandbox_platform.orchestrator import (
    SandboxPlatformOrchestrator,
    PlatformConfig,
    PlatformStatus,
)

__version__ = "1.0.0"

__all__ = [
    # Types
    "SandboxConfig",
    "SandboxProposal",
    "SandboxEvaluationResult",
    "ExecutionMode",
    "IsolationLevel",
    "ResourceAllocation",
    "MetricsSnapshot",
    "ProposalPriority",
    # Isolated execution
    "IsolatedSandboxExecutor",
    "SandboxContainer",
    "MemoryIsolation",
    # Merkle verification
    "MerkleVerifiedChannel",
    "VerificationResult",
    "AuditChainLogger",
    # Async pipeline
    "AsyncEvaluationPipeline",
    "PipelineStage",
    "NonBlockingQueue",
    # Parallelism
    "ShardedSandboxExecutor",
    "WorkloadShard",
    "NodeAllocation",
    "BatchProposalEvaluator",
    "BatchResult",
    "EvaluationBatch",
    # Lightweight evaluation
    "EphemeralContainer",
    "ContainerLifecycle",
    "AutoDestroyPolicy",
    "DeterministicStubRegistry",
    "ComputationStub",
    "StubFidelityLevel",
    "IncrementalStateEvaluator",
    "StateDelta",
    "StateCheckpoint",
    # Observability
    "PassiveMetricsCollector",
    "LatencyMetrics",
    "ThroughputMetrics",
    "EntropyMetrics",
    "TopologicalIndices",
    "DiagnosticsOffloader",
    "DiagnosticJob",
    "OffloadTarget",
    # Non-blocking integration
    "ProposalQueue",
    "QueuedProposal",
    "QueuePriority",
    "DualControlGateway",
    "ApprovalRequest",
    "AuthorizationStatus",
    # Optimization
    "LazyEvaluator",
    "EvaluationPolicy",
    "CriticalityAssessment",
    "SpeculativeExecutor",
    "SpeculativeResult",
    "LikelihoodEstimator",
    "ResourceThrottler",
    "ThrottlePolicy",
    "LoadMetrics",
    # Quantum/High-performance
    "QuantumSandbox",
    "HybridExecutor",
    "QuantumIsolation",
    "TensorNetworkPrecomputer",
    "TensorCache",
    "HybridComputation",
    # Orchestrator
    "SandboxPlatformOrchestrator",
    "PlatformConfig",
    "PlatformStatus",
]
