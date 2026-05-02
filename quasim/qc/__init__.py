"""Quantum Computing module for QuASIM.

This module provides quantum circuit simulation, gate operations,
quantum algorithm implementations, and distributed multi-GPU/multi-node
quantum simulation capabilities.
"""

from __future__ import annotations

from .circuit import QuantumCircuit
from .gates import GateSet
from .simulator import QCSimulator

# Multi-qubit and distributed simulation
try:
    from .quasim_dist import (DistContext, ShardedState, dist_apply_gate,
                              init_cluster, load_checkpoint, profile,
                              save_checkpoint, shard_state)
    from .quasim_multi import MultiQubitSimulator
    from .quasim_tn import TensorNetworkEngine

    _DISTRIBUTED_AVAILABLE = True
except ImportError:
    _DISTRIBUTED_AVAILABLE = False

__all__ = [
    "QuantumCircuit",
    "GateSet",
    "QCSimulator",
]

if _DISTRIBUTED_AVAILABLE:
    __all__.extend(
        [
            "MultiQubitSimulator",
            "TensorNetworkEngine",
            "DistContext",
            "ShardedState",
            "init_cluster",
            "shard_state",
            "dist_apply_gate",
            "save_checkpoint",
            "load_checkpoint",
            "profile",
        ]
    )
