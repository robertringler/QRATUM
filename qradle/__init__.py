"""
QRADLE - Quantum-Resilient Auditable Deterministic Ledger Engine

This is the foundational execution layer for QRATUM, providing:
- Deterministic execution with cryptographic proofs
- Merkle-chained event logs for complete auditability
- Contract-based operations with rollback capability
- 8 Fatal Invariants enforcement at runtime

Version: 1.0.0
Status: Production-Ready
"""

from qradle.contracts.system import ContractExecutor, ContractValidator
from qradle.core.engine import (DeterministicEngine, ExecutionContext,
                                ExecutionResult)
from qradle.core.invariants import FatalInvariants, InvariantViolation
from qradle.core.merkle import MerkleChain, MerkleProof
from qradle.core.rollback import Checkpoint, RollbackManager
from qradle.events.chain import Event, EventChain

__version__ = "1.0.0"
__all__ = [
    "DeterministicEngine",
    "ExecutionContext",
    "ExecutionResult",
    "FatalInvariants",
    "InvariantViolation",
    "MerkleChain",
    "MerkleProof",
    "RollbackManager",
    "Checkpoint",
    "ContractExecutor",
    "ContractValidator",
    "EventChain",
    "Event",
]
