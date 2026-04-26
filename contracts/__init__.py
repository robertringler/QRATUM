"""Contracts - Immutable Contract System for QRATUM.

This module provides immutable, hash-addressed contracts for intent authorization,
capability binding, temporal management, event sequencing, provenance tracking,
rollback operations with cryptographic proofs, and fatal invariant enforcement.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from contracts.base import (
    BaseContract,
    compute_contract_hash,
    generate_contract_id,
    get_current_timestamp,
)
from contracts.capability import (
    CapabilityContract,
    ClusterTopology,
    create_capability_contract,
)
from contracts.enforcement import (
    ContractEnforcer,
    EnforcedContract,
    EnforcementCheckpoint,
    EnforcementResult,
    create_enforced_contract,
)
from contracts.event import EventContract, create_event_contract
from contracts.intent import IntentContract, create_intent_contract
from contracts.provenance import (
    ComplianceArtifact,
    ComplianceStandard,
    ProvenanceChainBuilder,
    ProvenanceContract,
    ProvenanceEntry,
    ProvenanceType,
    create_21cfr11_provenance,
    create_do178c_provenance,
)
from contracts.rollback_proof import (
    RollbackContract,
    RollbackOrchestrator,
    RollbackProof,
    RollbackProofGenerator,
    RollbackReason,
    StateSnapshot,
    create_rollback_contract,
)
from contracts.temporal import TemporalContract, create_temporal_contract

__all__ = [
    # Base
    "BaseContract",
    "compute_contract_hash",
    "generate_contract_id",
    "get_current_timestamp",
    # Intent
    "IntentContract",
    "create_intent_contract",
    # Capability
    "CapabilityContract",
    "ClusterTopology",
    "create_capability_contract",
    # Temporal
    "TemporalContract",
    "create_temporal_contract",
    # Event
    "EventContract",
    "create_event_contract",
    # Provenance
    "ProvenanceContract",
    "ProvenanceChainBuilder",
    "ProvenanceEntry",
    "ProvenanceType",
    "ComplianceArtifact",
    "ComplianceStandard",
    "create_21cfr11_provenance",
    "create_do178c_provenance",
    # Rollback
    "RollbackContract",
    "RollbackProof",
    "RollbackProofGenerator",
    "RollbackOrchestrator",
    "RollbackReason",
    "StateSnapshot",
    "create_rollback_contract",
    # Enforcement
    "ContractEnforcer",
    "EnforcedContract",
    "EnforcementCheckpoint",
    "EnforcementResult",
    "create_enforced_contract",
]

__version__ = "1.0.0"
