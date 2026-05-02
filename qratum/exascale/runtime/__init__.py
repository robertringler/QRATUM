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

from .allreduce import (BinaryTreeTopology, DeterministicAllReduce,
                        ReductionOp, TreeTopology)
from .checkpoint import (CheckpointManager, CheckpointType,
                         DeterministicCheckpoint)
from .global_state import (ExecutionEpoch, GlobalStateManager,
                           SeedDerivationMethod, SeedManager)
from .qdr import QDR, ExecutionContext, ExecutionPhase, QDRConfig, RuntimeMode
from .scheduler import (DeterministicScheduler, KernelPriority, KernelQueue,
                        StreamType)

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
