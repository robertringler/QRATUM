"""Rollback Proof - Cryptographic Proof of Rollback Operations.

This module implements cryptographic proofs for rollback operations,
ensuring that all state rollbacks are verifiable and traceable.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from contracts.base import (
    BaseContract,
    compute_contract_hash,
    generate_contract_id,
    get_current_timestamp,
)


class RollbackReason(Enum):
    """Reasons for rollback operations."""

    INVARIANT_VIOLATION = "invariant_violation"
    EXECUTION_TIMEOUT = "execution_timeout"
    EXECUTION_FAILURE = "execution_failure"
    USER_INITIATED = "user_initiated"
    SYSTEM_RECOVERY = "system_recovery"
    SECURITY_INCIDENT = "security_incident"
    COMPLIANCE_REQUIREMENT = "compliance_requirement"


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable snapshot of system state.

    Attributes:
        snapshot_id: Unique snapshot identifier
        timestamp: When snapshot was taken
        state_hash: Hash of the state data
        state_data: The actual state data
        merkle_root: Merkle root of the state at this point
        zone: Security zone when snapshot was taken
    """

    snapshot_id: str
    timestamp: str
    state_hash: str
    state_data: dict[str, Any]
    merkle_root: str
    zone: str = "Z0"

    def verify(self) -> bool:
        """Verify snapshot integrity.

        Returns:
            True if snapshot is valid
        """
        serialized = json.dumps(self.state_data, sort_keys=True)
        computed_hash = hashlib.sha256(serialized.encode()).hexdigest()
        return computed_hash == self.state_hash

    def serialize(self) -> dict[str, Any]:
        """Serialize snapshot to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "state_hash": self.state_hash,
            "state_data": self.state_data,
            "merkle_root": self.merkle_root,
            "zone": self.zone,
        }


@dataclass(frozen=True)
class RollbackProof:
    """Cryptographic proof of a rollback operation.

    Attributes:
        proof_id: Unique proof identifier
        source_snapshot: Snapshot before rollback
        target_snapshot: Snapshot after rollback
        rollback_reason: Why rollback occurred
        authorized_by: Who authorized the rollback
        timestamp: When rollback occurred
        proof_signature: Cryptographic signature of proof
        chain_of_custody: List of custody transfers
    """

    proof_id: str
    source_snapshot_id: str
    source_state_hash: str
    target_snapshot_id: str
    target_state_hash: str
    rollback_reason: RollbackReason
    authorized_by: str
    timestamp: str
    proof_signature: str
    chain_of_custody: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def compute_proof_hash(self) -> str:
        """Compute hash of the proof content."""
        content = {
            "proof_id": self.proof_id,
            "source_snapshot_id": self.source_snapshot_id,
            "source_state_hash": self.source_state_hash,
            "target_snapshot_id": self.target_snapshot_id,
            "target_state_hash": self.target_state_hash,
            "rollback_reason": self.rollback_reason.value,
            "authorized_by": self.authorized_by,
            "timestamp": self.timestamp,
            "chain_of_custody": list(self.chain_of_custody),
        }
        json_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def verify(self) -> bool:
        """Verify proof integrity.

        Returns:
            True if proof signature matches computed hash
        """
        return self.proof_signature == self.compute_proof_hash()

    def serialize(self) -> dict[str, Any]:
        """Serialize proof to dictionary."""
        return {
            "proof_id": self.proof_id,
            "source_snapshot_id": self.source_snapshot_id,
            "source_state_hash": self.source_state_hash,
            "target_snapshot_id": self.target_snapshot_id,
            "target_state_hash": self.target_state_hash,
            "rollback_reason": self.rollback_reason.value,
            "authorized_by": self.authorized_by,
            "timestamp": self.timestamp,
            "proof_signature": self.proof_signature,
            "chain_of_custody": list(self.chain_of_custody),
            "verified": self.verify(),
        }


@dataclass(frozen=True)
class RollbackContract(BaseContract):
    """Immutable contract for rollback operations.

    Provides cryptographic proof of state rollbacks with full
    audit trail and chain of custody.

    Attributes:
        rollback_proof: The rollback proof
        source_contract_id: Contract that triggered rollback
        affected_contracts: List of affected contract IDs
        rollback_depth: Number of states rolled back
        zone_classification: Security zone
        rollback_authorization: Authorization proof
    """

    rollback_proof: dict[str, Any] = field(default_factory=dict)
    source_contract_id: str = ""
    affected_contracts: tuple[str, ...] = field(default_factory=tuple)
    rollback_depth: int = 0
    zone_classification: str = "Z0"
    rollback_authorization: str = ""

    def __post_init__(self) -> None:
        """Validate rollback contract after initialization."""
        super().__post_init__()
        if not self.rollback_proof:
            raise ValueError("rollback_proof cannot be empty")
        if not self.source_contract_id:
            raise ValueError("source_contract_id cannot be empty")
        if not self.rollback_authorization:
            raise ValueError("rollback_authorization cannot be empty")

    def serialize(self) -> dict[str, Any]:
        """Serialize rollback contract to dictionary."""
        base = super().serialize()
        base.update(
            {
                "rollback_proof": self.rollback_proof,
                "source_contract_id": self.source_contract_id,
                "affected_contracts": list(self.affected_contracts),
                "rollback_depth": self.rollback_depth,
                "zone_classification": self.zone_classification,
                "rollback_authorization": self.rollback_authorization,
            }
        )
        return base

    def verify_proof(self) -> bool:
        """Verify the embedded rollback proof.

        Returns:
            True if proof is valid
        """
        proof_data = self.rollback_proof

        # Recompute proof hash
        content = {
            "proof_id": proof_data.get("proof_id"),
            "source_snapshot_id": proof_data.get("source_snapshot_id"),
            "source_state_hash": proof_data.get("source_state_hash"),
            "target_snapshot_id": proof_data.get("target_snapshot_id"),
            "target_state_hash": proof_data.get("target_state_hash"),
            "rollback_reason": proof_data.get("rollback_reason"),
            "authorized_by": proof_data.get("authorized_by"),
            "timestamp": proof_data.get("timestamp"),
            "chain_of_custody": proof_data.get("chain_of_custody", []),
        }
        json_str = json.dumps(content, sort_keys=True)
        computed_hash = hashlib.sha256(json_str.encode("utf-8")).hexdigest()

        return proof_data.get("proof_signature") == computed_hash


class RollbackProofGenerator:
    """Generates cryptographic proofs for rollback operations."""

    def __init__(self):
        """Initialize the generator."""
        self._proof_count = 0
        self._custody_log: list[dict[str, Any]] = []

    def create_snapshot(
        self,
        state_data: dict[str, Any],
        merkle_root: str,
        zone: str = "Z0",
    ) -> StateSnapshot:
        """Create a state snapshot.

        Args:
            state_data: State data to snapshot
            merkle_root: Current Merkle root
            zone: Security zone

        Returns:
            StateSnapshot
        """
        timestamp = get_current_timestamp()
        serialized = json.dumps(state_data, sort_keys=True)
        state_hash = hashlib.sha256(serialized.encode()).hexdigest()
        snapshot_id = f"snap_{state_hash[:16]}_{int(datetime.now(timezone.utc).timestamp())}"

        return StateSnapshot(
            snapshot_id=snapshot_id,
            timestamp=timestamp,
            state_hash=state_hash,
            state_data=state_data,
            merkle_root=merkle_root,
            zone=zone,
        )

    def generate_proof(
        self,
        source_snapshot: StateSnapshot,
        target_snapshot: StateSnapshot,
        reason: RollbackReason,
        authorized_by: str,
    ) -> RollbackProof:
        """Generate a rollback proof.

        Args:
            source_snapshot: Snapshot before rollback
            target_snapshot: Snapshot to rollback to
            reason: Reason for rollback
            authorized_by: Authorizer

        Returns:
            RollbackProof
        """
        self._proof_count += 1
        timestamp = get_current_timestamp()
        proof_id = f"proof_{self._proof_count:06d}_{compute_contract_hash({'ts': timestamp})[:8]}"

        # Record custody transfer
        custody_entry = {
            "timestamp": timestamp,
            "action": "rollback_initiated",
            "from_state": source_snapshot.snapshot_id,
            "to_state": target_snapshot.snapshot_id,
            "authorized_by": authorized_by,
        }
        self._custody_log.append(custody_entry)

        # Build chain of custody
        chain_of_custody = tuple(self._custody_log[-10:])  # Last 10 entries

        # Create proof content for signing
        content = {
            "proof_id": proof_id,
            "source_snapshot_id": source_snapshot.snapshot_id,
            "source_state_hash": source_snapshot.state_hash,
            "target_snapshot_id": target_snapshot.snapshot_id,
            "target_state_hash": target_snapshot.state_hash,
            "rollback_reason": reason.value,
            "authorized_by": authorized_by,
            "timestamp": timestamp,
            "chain_of_custody": list(chain_of_custody),
        }
        json_str = json.dumps(content, sort_keys=True)
        proof_signature = hashlib.sha256(json_str.encode("utf-8")).hexdigest()

        return RollbackProof(
            proof_id=proof_id,
            source_snapshot_id=source_snapshot.snapshot_id,
            source_state_hash=source_snapshot.state_hash,
            target_snapshot_id=target_snapshot.snapshot_id,
            target_state_hash=target_snapshot.state_hash,
            rollback_reason=reason,
            authorized_by=authorized_by,
            timestamp=timestamp,
            proof_signature=proof_signature,
            chain_of_custody=chain_of_custody,
        )

    def get_custody_log(self) -> list[dict[str, Any]]:
        """Get the chain of custody log.

        Returns:
            List of custody log entries
        """
        return self._custody_log.copy()


def create_rollback_contract(
    rollback_proof: RollbackProof,
    source_contract_id: str,
    affected_contracts: list[str],
    rollback_depth: int = 1,
    zone: str = "Z0",
    authorization: str = "rollback_authorized",
) -> RollbackContract:
    """Create a RollbackContract.

    Args:
        rollback_proof: The rollback proof
        source_contract_id: Contract that triggered rollback
        affected_contracts: List of affected contract IDs
        rollback_depth: Number of states rolled back
        zone: Security zone
        authorization: Authorization proof

    Returns:
        RollbackContract
    """
    content = {
        "rollback_proof": rollback_proof.serialize(),
        "source_contract_id": source_contract_id,
        "affected_contracts": affected_contracts,
        "rollback_depth": rollback_depth,
        "zone_classification": zone,
        "rollback_authorization": authorization,
        "created_at": get_current_timestamp(),
        "version": "1.0.0",
    }

    contract_id = generate_contract_id("RollbackContract", content)

    return RollbackContract(
        contract_id=contract_id,
        contract_type="RollbackContract",
        created_at=content["created_at"],
        version=content["version"],
        rollback_proof=content["rollback_proof"],
        source_contract_id=content["source_contract_id"],
        affected_contracts=tuple(content["affected_contracts"]),
        rollback_depth=content["rollback_depth"],
        zone_classification=content["zone_classification"],
        rollback_authorization=content["rollback_authorization"],
    )


class RollbackOrchestrator:
    """Orchestrates rollback operations with cryptographic proof generation."""

    def __init__(self):
        """Initialize the orchestrator."""
        self.proof_generator = RollbackProofGenerator()
        self._snapshots: dict[str, StateSnapshot] = {}
        self._rollback_contracts: list[RollbackContract] = []

    def create_checkpoint(
        self,
        state_data: dict[str, Any],
        merkle_root: str,
        zone: str = "Z0",
    ) -> StateSnapshot:
        """Create a state checkpoint.

        Args:
            state_data: State data to checkpoint
            merkle_root: Current Merkle root
            zone: Security zone

        Returns:
            StateSnapshot
        """
        snapshot = self.proof_generator.create_snapshot(state_data, merkle_root, zone)
        self._snapshots[snapshot.snapshot_id] = snapshot
        return snapshot

    def execute_rollback(
        self,
        target_snapshot_id: str,
        reason: RollbackReason,
        authorized_by: str,
        source_contract_id: str,
        affected_contracts: list[str] | None = None,
    ) -> RollbackContract:
        """Execute a rollback operation with proof generation.

        Args:
            target_snapshot_id: Snapshot to rollback to
            reason: Reason for rollback
            authorized_by: Authorizer
            source_contract_id: Contract triggering rollback
            affected_contracts: List of affected contracts

        Returns:
            RollbackContract documenting the rollback

        Raises:
            ValueError: If target snapshot not found
        """
        if target_snapshot_id not in self._snapshots:
            raise ValueError(f"Target snapshot not found: {target_snapshot_id}")

        target_snapshot = self._snapshots[target_snapshot_id]

        # Get most recent snapshot as source
        if self._snapshots:
            source_snapshot = max(
                self._snapshots.values(),
                key=lambda s: s.timestamp,
            )
        else:
            # No source - create empty snapshot
            source_snapshot = self.proof_generator.create_snapshot({"state": "empty"}, "0" * 64)

        # Generate rollback proof
        proof = self.proof_generator.generate_proof(
            source_snapshot=source_snapshot,
            target_snapshot=target_snapshot,
            reason=reason,
            authorized_by=authorized_by,
        )

        # Calculate rollback depth more efficiently
        snapshot_ids = list(self._snapshots.keys())
        try:
            source_idx = snapshot_ids.index(source_snapshot.snapshot_id)
            target_idx = snapshot_ids.index(target_snapshot_id)
            rollback_depth = abs(source_idx - target_idx)
        except ValueError:
            rollback_depth = 1  # Default if calculation fails

        # Create rollback contract
        contract = create_rollback_contract(
            rollback_proof=proof,
            source_contract_id=source_contract_id,
            affected_contracts=affected_contracts or [],
            rollback_depth=rollback_depth,
            zone=target_snapshot.zone,
        )

        self._rollback_contracts.append(contract)
        return contract

    def get_snapshot(self, snapshot_id: str) -> StateSnapshot | None:
        """Get snapshot by ID.

        Args:
            snapshot_id: Snapshot ID

        Returns:
            StateSnapshot or None
        """
        return self._snapshots.get(snapshot_id)

    def list_snapshots(self) -> list[dict[str, Any]]:
        """List all snapshots.

        Returns:
            List of snapshot summaries
        """
        return [
            {
                "snapshot_id": s.snapshot_id,
                "timestamp": s.timestamp,
                "state_hash": s.state_hash,
                "zone": s.zone,
            }
            for s in self._snapshots.values()
        ]

    def get_rollback_history(self) -> list[dict[str, Any]]:
        """Get rollback history.

        Returns:
            List of rollback contract summaries
        """
        return [
            {
                "contract_id": c.contract_id,
                "source_contract_id": c.source_contract_id,
                "rollback_depth": c.rollback_depth,
                "created_at": c.created_at,
                "zone": c.zone_classification,
            }
            for c in self._rollback_contracts
        ]

    def verify_all_proofs(self) -> dict[str, bool]:
        """Verify all rollback proofs.

        Returns:
            Dictionary mapping contract IDs to verification results
        """
        return {c.contract_id: c.verify_proof() for c in self._rollback_contracts}
