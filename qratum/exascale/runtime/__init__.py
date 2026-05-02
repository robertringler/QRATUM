"""
QDR - QRATUM Deterministic Runtime

Provides deterministic execution guarantees at exascale, ensuring bit-identical
reproducibility across all runs while maintaining high performance.

Components:
- qdr: Main runtime orchestration
- allreduce: Fixed-order binary tree AllReduce
- checkpoint: Deterministic checkpointing
- scheduler: Deterministic GPU kernel scheduling
- global_state: Global seed and execution ordering
"""

from .qdr import QDR, QDRConfig, ExecutionContext, RuntimeMode, ExecutionPhase
from .allreduce import DeterministicAllReduce, BinaryTreeTopology, ReductionOp, TreeTopology
from .checkpoint import DeterministicCheckpoint, CheckpointManager, CheckpointType
from .scheduler import DeterministicScheduler, KernelQueue, KernelPriority, StreamType
from .global_state import GlobalStateManager, SeedManager, ExecutionEpoch, SeedDerivationMethod

__all__ = [
    "QDR",
    "QDRConfig",
    "ExecutionContext",
    "RuntimeMode",
    "ExecutionPhase",
    "DeterministicAllReduce",
    "BinaryTreeTopology",
    "ReductionOp",
    "TreeTopology",
    "DeterministicCheckpoint",
    "CheckpointManager",
    "CheckpointType",
    "DeterministicScheduler",
    "KernelQueue",
    "KernelPriority",
    "StreamType",
    "GlobalStateManager",
    "SeedManager",
    "ExecutionEpoch",
    "SeedDerivationMethod",
]
