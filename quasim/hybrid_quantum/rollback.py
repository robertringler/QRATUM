"""Rollback Management for Quantum Executions.

This module implements deterministic rollback capabilities for quantum
executions, enabling recovery from failed or rejected results.

Key features:
- State checkpointing before quantum execution
- Deterministic state recovery
- Chain-based rollback for multi-step executions
- Integration with provenance tracking
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable


class CheckpointStatus(Enum):
    """Status of a checkpoint."""

    ACTIVE = "active"
    RESTORED = "restored"
    SUPERSEDED = "superseded"
    INVALID = "invalid"


@dataclass
class Checkpoint:
    """Checkpoint of system state for rollback.

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint
        state_snapshot: Serialized system state
        state_hash: Hash of state for integrity verification
        timestamp: When checkpoint was created
        execution_id: ID of quantum execution this checkpoint precedes
        parent_checkpoint_id: ID of parent checkpoint (for chained rollbacks)
        status: Current status of checkpoint
        metadata: Additional metadata
    """

    checkpoint_id: str
    state_snapshot: dict[str, Any]
    state_hash: str = ""
    timestamp: str = ""
    execution_id: str = ""
    parent_checkpoint_id: str | None = None
    status: CheckpointStatus = CheckpointStatus.ACTIVE
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set defaults and compute hash."""
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        if not self.state_hash:
            self.state_hash = self._compute_state_hash()

    def _compute_state_hash(self) -> str:
        """Compute hash of state snapshot."""
        state_str = json.dumps(self.state_snapshot, sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify checkpoint integrity.

        Returns:
            True if state hash matches computed hash
        """
        return self.state_hash == self._compute_state_hash()


class RollbackManager:
    """Manager for deterministic rollback of quantum executions.

    This manager enables recovery from failed quantum executions by:
    1. Creating checkpoints before each execution
    2. Storing execution chain relationships
    3. Restoring state on rollback request

    Example:
        >>> manager = RollbackManager()
        >>> # Create checkpoint before quantum execution
        >>> checkpoint_id = manager.create_checkpoint(current_state, execution_id)
        >>> # Execute quantum operation
        >>> result = backend.execute_circuit(circuit)
        >>> # If result fails verification, rollback
        >>> if not verification.is_approved:
        ...     restored_state = manager.rollback(checkpoint_id)
    """

    def __init__(self, max_checkpoints: int = 100):
        """Initialize rollback manager.

        Args:
            max_checkpoints: Maximum checkpoints to retain
        """
        self.max_checkpoints = max_checkpoints
        self._checkpoints: dict[str, Checkpoint] = {}
        self._execution_to_checkpoint: dict[str, str] = {}
        self._checkpoint_chain: list[str] = []

    def create_checkpoint(
        self,
        state: dict[str, Any],
        execution_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create checkpoint before quantum execution.

        Args:
            state: Current system state to checkpoint
            execution_id: ID of upcoming quantum execution
            metadata: Additional metadata

        Returns:
            Checkpoint ID
        """
        checkpoint_id = str(uuid.uuid4())

        # Deep copy state to prevent mutations
        state_snapshot = copy.deepcopy(state)

        # Get parent checkpoint
        parent_id = self._checkpoint_chain[-1] if self._checkpoint_chain else None

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            state_snapshot=state_snapshot,
            execution_id=execution_id or "",
            parent_checkpoint_id=parent_id,
            metadata=metadata or {},
        )

        self._checkpoints[checkpoint_id] = checkpoint
        self._checkpoint_chain.append(checkpoint_id)

        if execution_id:
            self._execution_to_checkpoint[execution_id] = checkpoint_id

        # Prune old checkpoints if needed
        self._prune_checkpoints()

        return checkpoint_id

    def rollback(self, checkpoint_id: str) -> dict[str, Any] | None:
        """Rollback to checkpoint state.

        Args:
            checkpoint_id: ID of checkpoint to restore

        Returns:
            Restored state or None if checkpoint not found
        """
        if checkpoint_id not in self._checkpoints:
            return None

        checkpoint = self._checkpoints[checkpoint_id]

        if not checkpoint.verify_integrity():
            checkpoint.status = CheckpointStatus.INVALID
            return None

        # Mark subsequent checkpoints as superseded
        try:
            checkpoint_idx = self._checkpoint_chain.index(checkpoint_id)
            for later_id in self._checkpoint_chain[checkpoint_idx + 1 :]:
                if later_id in self._checkpoints:
                    self._checkpoints[later_id].status = CheckpointStatus.SUPERSEDED
            # Truncate chain
            self._checkpoint_chain = self._checkpoint_chain[: checkpoint_idx + 1]
        except ValueError:
            pass

        # Mark checkpoint as restored
        checkpoint.status = CheckpointStatus.RESTORED

        # Return deep copy of state
        return copy.deepcopy(checkpoint.state_snapshot)

    def rollback_execution(self, execution_id: str) -> dict[str, Any] | None:
        """Rollback to state before specific execution.

        Args:
            execution_id: ID of quantum execution to rollback

        Returns:
            Restored state or None if execution not found
        """
        checkpoint_id = self._execution_to_checkpoint.get(execution_id)
        if checkpoint_id:
            return self.rollback(checkpoint_id)
        return None

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Get checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint or None if not found
        """
        return self._checkpoints.get(checkpoint_id)

    def get_checkpoint_chain(self) -> list[Checkpoint]:
        """Get full checkpoint chain.

        Returns:
            List of checkpoints in creation order
        """
        return [self._checkpoints[cid] for cid in self._checkpoint_chain if cid in self._checkpoints]

    def verify_all_checkpoints(self) -> dict[str, bool]:
        """Verify integrity of all checkpoints.

        Returns:
            Dictionary mapping checkpoint_id to integrity status
        """
        return {cid: cp.verify_integrity() for cid, cp in self._checkpoints.items()}

    def _prune_checkpoints(self) -> None:
        """Remove oldest checkpoints if over limit."""
        while len(self._checkpoint_chain) > self.max_checkpoints:
            oldest_id = self._checkpoint_chain.pop(0)
            if oldest_id in self._checkpoints:
                del self._checkpoints[oldest_id]
            # Clean up execution mapping
            self._execution_to_checkpoint = {
                eid: cid for eid, cid in self._execution_to_checkpoint.items() if cid in self._checkpoints
            }

    def export_checkpoint_log(self) -> str:
        """Export checkpoint log as JSON.

        Returns:
            JSON string of checkpoint metadata (without full state snapshots)
        """
        log = []
        for cid in self._checkpoint_chain:
            if cid in self._checkpoints:
                cp = self._checkpoints[cid]
                log.append(
                    {
                        "checkpoint_id": cp.checkpoint_id,
                        "state_hash": cp.state_hash,
                        "timestamp": cp.timestamp,
                        "execution_id": cp.execution_id,
                        "parent_checkpoint_id": cp.parent_checkpoint_id,
                        "status": cp.status.value,
                    }
                )
        return json.dumps({"checkpoints": log}, indent=2)


class DualApprovalGate:
    """Gate requiring dual approval for quantum execution.

    This gate implements the dual-control invariant for high-stakes
    quantum operations, requiring two independent approvals before
    execution proceeds.

    Example:
        >>> gate = DualApprovalGate()
        >>> request_id = gate.request_approval(execution_id, circuit)
        >>> gate.approve(request_id, "approver_1")
        >>> gate.approve(request_id, "approver_2")
        >>> if gate.is_approved(request_id):
        ...     # Proceed with execution
        ...     pass
    """

    def __init__(self, required_approvals: int = 2):
        """Initialize dual approval gate.

        Args:
            required_approvals: Number of approvals required
        """
        self.required_approvals = required_approvals
        self._pending_requests: dict[str, dict[str, Any]] = {}

    def request_approval(
        self,
        execution_id: str,
        circuit_info: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Request approval for quantum execution.

        Args:
            execution_id: ID of proposed execution
            circuit_info: Information about circuit to execute
            metadata: Additional context

        Returns:
            Request ID for tracking approvals
        """
        request_id = str(uuid.uuid4())

        self._pending_requests[request_id] = {
            "request_id": request_id,
            "execution_id": execution_id,
            "circuit_info": circuit_info,
            "metadata": metadata or {},
            "approvals": [],
            "rejections": [],
            "timestamp": datetime.utcnow().isoformat(),
            "status": "pending",
        }

        return request_id

    def approve(self, request_id: str, approver_id: str, notes: str = "") -> bool:
        """Approve quantum execution request.

        Args:
            request_id: Request ID to approve
            approver_id: ID of approving entity
            notes: Optional approval notes

        Returns:
            True if approval recorded
        """
        if request_id not in self._pending_requests:
            return False

        request = self._pending_requests[request_id]

        if approver_id in [a["approver_id"] for a in request["approvals"]]:
            return False  # Already approved by this entity

        request["approvals"].append(
            {
                "approver_id": approver_id,
                "timestamp": datetime.utcnow().isoformat(),
                "notes": notes,
            }
        )

        if len(request["approvals"]) >= self.required_approvals:
            request["status"] = "approved"

        return True

    def reject(self, request_id: str, rejector_id: str, reason: str = "") -> bool:
        """Reject quantum execution request.

        Args:
            request_id: Request ID to reject
            rejector_id: ID of rejecting entity
            reason: Rejection reason

        Returns:
            True if rejection recorded
        """
        if request_id not in self._pending_requests:
            return False

        request = self._pending_requests[request_id]
        request["rejections"].append(
            {
                "rejector_id": rejector_id,
                "timestamp": datetime.utcnow().isoformat(),
                "reason": reason,
            }
        )
        request["status"] = "rejected"

        return True

    def is_approved(self, request_id: str) -> bool:
        """Check if request has required approvals.

        Args:
            request_id: Request ID to check

        Returns:
            True if approved
        """
        if request_id not in self._pending_requests:
            return False

        return self._pending_requests[request_id]["status"] == "approved"

    def get_request_status(self, request_id: str) -> dict[str, Any] | None:
        """Get request status details.

        Args:
            request_id: Request ID

        Returns:
            Request details or None if not found
        """
        return self._pending_requests.get(request_id)

    def get_pending_requests(self) -> list[dict[str, Any]]:
        """Get all pending requests.

        Returns:
            List of pending request details
        """
        return [r for r in self._pending_requests.values() if r["status"] == "pending"]
