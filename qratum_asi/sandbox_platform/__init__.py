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
# Async pipeline
from qratum_asi.sandbox_platform.async_pipeline import (
    AsyncEvaluationPipeline,
    NonBlockingQueue,
    PipelineStage,
)
from qratum_asi.sandbox_platform.batch_evaluator import (
    BatchProposalEvaluator,
    BatchResult,
    EvaluationBatch,
)
from qratum_asi.sandbox_platform.deterministic_stubs import (
    ComputationStub,
    DeterministicStubRegistry,
    StubFidelityLevel,
)
from qratum_asi.sandbox_platform.diagnostics_offload import (
    DiagnosticJob,
    DiagnosticsOffloader,
    OffloadTarget,
)
from qratum_asi.sandbox_platform.dual_control_gateway import (
    ApprovalRequest,
    AuthorizationStatus,
    DualControlGateway,
)

# Lightweight evaluation
from qratum_asi.sandbox_platform.ephemeral_container import (
    AutoDestroyPolicy,
    ContainerLifecycle,
    EphemeralContainer,
)
from qratum_asi.sandbox_platform.incremental_state import (
    IncrementalStateEvaluator,
    StateCheckpoint,
    StateDelta,
)

# Isolated execution
from qratum_asi.sandbox_platform.isolated_sandbox import (
    IsolatedSandboxExecutor,
    MemoryIsolation,
    SandboxContainer,
)

# Optimization
from qratum_asi.sandbox_platform.lazy_evaluator import (
    CriticalityAssessment,
    EvaluationPolicy,
    LazyEvaluator,
)

# Merkle verification
from qratum_asi.sandbox_platform.merkle_verifier import (
    AuditChainLogger,
    MerkleVerifiedChannel,
    VerificationResult,
)

# Main orchestrator
from qratum_asi.sandbox_platform.orchestrator import (
    PlatformConfig,
    PlatformStatus,
    SandboxPlatformOrchestrator,
)

# Observability
from qratum_asi.sandbox_platform.passive_metrics import (
    EntropyMetrics,
    LatencyMetrics,
    PassiveMetricsCollector,
    ThroughputMetrics,
    TopologicalIndices,
)

# Non-blocking integration
from qratum_asi.sandbox_platform.proposal_queue import (
    ProposalQueue,
    QueuedProposal,
    QueuePriority,
)

# Quantum/High-performance (optional)
from qratum_asi.sandbox_platform.quantum_sandbox import (
    HybridExecutor,
    QuantumIsolation,
    QuantumSandbox,
)
from qratum_asi.sandbox_platform.resource_throttler import (
    LoadMetrics,
    ResourceThrottler,
    ThrottlePolicy,
)

# Parallelism
from qratum_asi.sandbox_platform.sharded_executor import (
    NodeAllocation,
    ShardedSandboxExecutor,
    WorkloadShard,
)
from qratum_asi.sandbox_platform.speculative_executor import (
    LikelihoodEstimator,
    SpeculativeExecutor,
    SpeculativeResult,
)
from qratum_asi.sandbox_platform.tensor_network import (
    HybridComputation,
    TensorCache,
    TensorNetworkPrecomputer,
)
from qratum_asi.sandbox_platform.types import (
    ExecutionMode,
    IsolationLevel,
    MetricsSnapshot,
    ProposalPriority,
    ResourceAllocation,
    SandboxConfig,
    SandboxEvaluationResult,
    SandboxProposal,
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
