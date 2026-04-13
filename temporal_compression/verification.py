"""Cryptographic Verification Module for Temporal Compression Experiments.

Provides tamper-evident audit trails and cryptographic verification
of state transitions for deterministic simulations.

Features:
- SHA3-256 hash-based state verification
- Merkle-chain audit trails
- Tamper detection
- Independent verification support
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class EventType(str, Enum):
    """Types of auditable events."""

    GENESIS = "genesis"
    STATE_TRANSITION = "state_transition"
    CHECKPOINT = "checkpoint"
    VERIFICATION = "verification"
    BRANCH_CREATE = "branch_create"
    BRANCH_EVOLVE = "branch_evolve"
    EXPERIMENT_START = "experiment_start"
    EXPERIMENT_END = "experiment_end"
    REPRODUCIBILITY_CHECK = "reproducibility_check"


@dataclass
class AuditEntry:
    """Single entry in the audit trail.

    Attributes:
        event_type: Type of event
        step: Simulation step when event occurred
        timestamp: Wall-clock time
        state_hash: Hash of state at this point
        data: Additional event data
        previous_hash: Hash of previous audit entry
        entry_hash: Hash of this entry
    """

    event_type: EventType
    step: int
    timestamp: float
    state_hash: str
    data: dict[str, Any]
    previous_hash: str = "0" * 64
    entry_hash: str = ""

    def __post_init__(self) -> None:
        """Compute entry hash after initialization."""
        if not self.entry_hash:
            self.entry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA3-256 hash of this entry."""
        payload = {
            "event_type": self.event_type.value,
            "step": self.step,
            "timestamp": self.timestamp,
            "state_hash": self.state_hash,
            "data": self.data,
            "previous_hash": self.previous_hash,
        }
        json_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha3_256(json_str.encode()).hexdigest()

    def verify_hash(self) -> bool:
        """Verify that stored hash matches computed hash."""
        return self.entry_hash == self._compute_hash()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "step": self.step,
            "timestamp": self.timestamp,
            "state_hash": self.state_hash,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEntry:
        """Create entry from dictionary."""
        return cls(
            event_type=EventType(data["event_type"]),
            step=data["step"],
            timestamp=data["timestamp"],
            state_hash=data["state_hash"],
            data=data.get("data", {}),
            previous_hash=data["previous_hash"],
            entry_hash=data.get("entry_hash", ""),
        )


class VerificationChain:
    """Merkle-chained audit trail for verification.

    Maintains tamper-evident record of all state transitions
    and significant events during simulation.

    Example:
        chain = VerificationChain()
        chain.add_state_transition(step=0, state_hash="abc123", data={})
        chain.add_state_transition(step=1, state_hash="def456", data={})

        assert chain.verify_integrity()
        proof = chain.get_chain_proof()
    """

    def __init__(self) -> None:
        """Initialize verification chain."""
        self.entries: list[AuditEntry] = []
        self._add_genesis()

    def _add_genesis(self) -> None:
        """Add genesis entry to chain."""
        genesis = AuditEntry(
            event_type=EventType.GENESIS,
            step=0,
            timestamp=time.time(),
            state_hash="0" * 64,
            data={"message": "Temporal compression verification chain initialized"},
            previous_hash="0" * 64,
        )
        self.entries.append(genesis)

    def add_event(
        self,
        event_type: EventType,
        step: int,
        state_hash: str,
        data: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Add event to the chain.

        Args:
            event_type: Type of event
            step: Current simulation step
            state_hash: Hash of current state
            data: Additional event data

        Returns:
            Created AuditEntry
        """
        previous_hash = self.entries[-1].entry_hash if self.entries else "0" * 64

        entry = AuditEntry(
            event_type=event_type,
            step=step,
            timestamp=time.time(),
            state_hash=state_hash,
            data=data or {},
            previous_hash=previous_hash,
        )

        self.entries.append(entry)
        return entry

    def add_state_transition(
        self,
        step: int,
        state_hash: str,
        data: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Add state transition to chain.

        Args:
            step: Current simulation step
            state_hash: Hash of current state
            data: Additional transition data

        Returns:
            Created AuditEntry
        """
        return self.add_event(EventType.STATE_TRANSITION, step, state_hash, data)

    def add_checkpoint(
        self,
        step: int,
        state_hash: str,
        checkpoint_id: str,
        data: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Add checkpoint to chain.

        Args:
            step: Current simulation step
            state_hash: Hash of current state
            checkpoint_id: Unique checkpoint identifier
            data: Additional checkpoint data

        Returns:
            Created AuditEntry
        """
        checkpoint_data = {"checkpoint_id": checkpoint_id, **(data or {})}
        return self.add_event(EventType.CHECKPOINT, step, state_hash, checkpoint_data)

    def verify_integrity(self) -> bool:
        """Verify integrity of the entire chain.

        Returns:
            True if chain is valid, False if tampered
        """
        if not self.entries:
            return False

        # Verify genesis
        if self.entries[0].previous_hash != "0" * 64:
            return False

        # Verify chain links and hashes
        for i in range(1, len(self.entries)):
            current = self.entries[i]
            previous = self.entries[i - 1]

            # Verify previous hash linkage
            if current.previous_hash != previous.entry_hash:
                return False

            # Verify entry hash integrity
            if not current.verify_hash():
                return False

        return True

    def get_chain_proof(self) -> str:
        """Get cryptographic proof of chain state.

        Returns:
            Hash representing the entire chain
        """
        if not self.entries:
            return "0" * 64
        return self.entries[-1].entry_hash

    def get_entries(
        self,
        event_type: EventType | None = None,
    ) -> list[AuditEntry]:
        """Get entries, optionally filtered by type.

        Args:
            event_type: Optional filter by event type

        Returns:
            List of matching entries
        """
        if event_type is None:
            return self.entries.copy()
        return [e for e in self.entries if e.event_type == event_type]

    def get_state_transitions(self) -> list[AuditEntry]:
        """Get all state transition entries."""
        return self.get_entries(EventType.STATE_TRANSITION)

    def export_for_verification(self) -> dict[str, Any]:
        """Export chain for independent verification.

        Returns:
            Dictionary with all data needed for verification
        """
        return {
            "version": "1.0",
            "algorithm": "SHA3-256",
            "chain_proof": self.get_chain_proof(),
            "chain_length": len(self.entries),
            "integrity_verified": self.verify_integrity(),
            "entries": [e.to_dict() for e in self.entries],
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert chain to dictionary."""
        return {
            "chain_proof": self.get_chain_proof(),
            "chain_length": len(self.entries),
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerificationChain:
        """Create chain from dictionary."""
        chain = cls.__new__(cls)
        chain.entries = [AuditEntry.from_dict(e) for e in data["entries"]]
        return chain


class StateVerifier:
    """Verifier for state transitions and reproducibility.

    Provides tools for verifying that simulation runs produce
    identical results and detecting any deviations.
    """

    def __init__(self, tolerance: float = 0.0) -> None:
        """Initialize state verifier.

        Args:
            tolerance: Tolerance for floating-point comparisons (0 = exact)
        """
        self.tolerance = tolerance
        self.verification_results: list[dict[str, Any]] = []

    def compute_state_hash(self, state: np.ndarray) -> str:
        """Compute hash of state array.

        Args:
            state: State array to hash

        Returns:
            SHA3-256 hash of state
        """
        hasher = hashlib.sha3_256()
        hasher.update(state.tobytes())
        return hasher.hexdigest()

    def verify_identical_states(
        self,
        state1: np.ndarray,
        state2: np.ndarray,
    ) -> tuple[bool, dict[str, Any]]:
        """Verify two states are identical.

        Args:
            state1: First state
            state2: Second state

        Returns:
            Tuple of (is_identical, details)
        """
        if state1.shape != state2.shape:
            return False, {
                "reason": "shape_mismatch",
                "shape1": state1.shape,
                "shape2": state2.shape,
            }

        hash1 = self.compute_state_hash(state1)
        hash2 = self.compute_state_hash(state2)

        if hash1 != hash2:
            # Find where they differ
            diff_mask = state1 != state2
            diff_count = int(np.sum(diff_mask))
            diff_indices = np.where(diff_mask)[0].tolist()[:10]  # First 10

            return False, {
                "reason": "content_mismatch",
                "hash1": hash1,
                "hash2": hash2,
                "diff_count": diff_count,
                "sample_diff_indices": diff_indices,
            }

        return True, {
            "reason": "identical",
            "hash": hash1,
        }

    def verify_reproducibility(
        self,
        run_hashes: list[str],
    ) -> tuple[bool, dict[str, Any]]:
        """Verify reproducibility across multiple runs.

        Args:
            run_hashes: List of final state hashes from runs

        Returns:
            Tuple of (is_reproducible, details)
        """
        if not run_hashes:
            return False, {"reason": "no_data"}

        first_hash = run_hashes[0]
        all_match = all(h == first_hash for h in run_hashes)

        if all_match:
            return True, {
                "reason": "all_identical",
                "hash": first_hash,
                "num_runs": len(run_hashes),
            }
        else:
            # Find mismatches
            unique_hashes = list(set(run_hashes))
            return False, {
                "reason": "hash_mismatch",
                "num_runs": len(run_hashes),
                "unique_hashes": unique_hashes,
                "num_unique": len(unique_hashes),
            }

    def verify_chain(self, chain: VerificationChain) -> tuple[bool, dict[str, Any]]:
        """Verify a verification chain.

        Args:
            chain: Chain to verify

        Returns:
            Tuple of (is_valid, details)
        """
        is_valid = chain.verify_integrity()

        details = {
            "is_valid": is_valid,
            "chain_length": len(chain.entries),
            "chain_proof": chain.get_chain_proof(),
        }

        if not is_valid:
            # Find where chain is broken
            for i in range(1, len(chain.entries)):
                current = chain.entries[i]
                previous = chain.entries[i - 1]

                if current.previous_hash != previous.entry_hash:
                    details["break_at"] = i
                    details["reason"] = "link_broken"
                    break

                if not current.verify_hash():
                    details["break_at"] = i
                    details["reason"] = "hash_invalid"
                    break

        self.verification_results.append(details)
        return is_valid, details

    def generate_verification_report(
        self,
        chain: VerificationChain,
        run_hashes: list[str],
    ) -> dict[str, Any]:
        """Generate comprehensive verification report.

        Args:
            chain: Verification chain to report on
            run_hashes: Hashes from multiple runs

        Returns:
            Verification report dictionary
        """
        chain_valid, chain_details = self.verify_chain(chain)
        reproducible, repro_details = self.verify_reproducibility(run_hashes)

        return {
            "timestamp": time.time(),
            "chain_verification": {
                "is_valid": chain_valid,
                **chain_details,
            },
            "reproducibility": {
                "is_reproducible": reproducible,
                **repro_details,
            },
            "audit_summary": {
                "total_events": len(chain.entries),
                "state_transitions": len(chain.get_state_transitions()),
                "checkpoints": len(chain.get_entries(EventType.CHECKPOINT)),
            },
            "verification_metadata": {
                "algorithm": "SHA3-256",
                "tolerance": self.tolerance,
            },
        }
