"""QIL Discovery Contracts for Reinjection.

Implements QIL (QRATUM Intent Language) contracts for discovery reinjection
with dual-control authorization and compliance mapping.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from qradle.merkle import MerkleChain

from qratum_asi.reinjection.types import (
    DiscoveryDomain,
    ReinjectionCandidate,
    ReinjectionStatus,
    ValidationLevel,
)


class ReinjectionContractStatus(Enum):
    """Status of a reinjection contract."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    Z1_SANDBOX = "z1_sandbox"
    AWAITING_APPROVAL = "awaiting_approval"
    Z2_COMMITTED = "z2_committed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


@dataclass
class ApprovalRecord:
    """Record of an approval decision.

    Attributes:
        approver_id: ID of approver
        decision: approval decision (approve/reject)
        reason: Reason for decision
        scope: Scope of approval
        timestamp: Decision timestamp
    """

    approver_id: str
    decision: str  # "approve" or "reject"
    reason: str
    scope: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize approval record."""
        return {
            "approver_id": self.approver_id,
            "decision": self.decision,
            "reason": self.reason,
            "scope": self.scope,
            "timestamp": self.timestamp,
        }


@dataclass
class ReinjectionContract:
    """QIL Contract for discovery reinjection.

    Implements contract-bound reinjection with:
    - Dual-control authorization
    - Zone-enforced execution (Z1 sandbox -> Z2 commit)
    - Full provenance tracking
    - Rollback capability

    Attributes:
        contract_id: Unique contract identifier
        candidate: Reinjection candidate
        required_approvers: List of required approvers
        approvals: Collected approvals
        execution_log: Execution audit log
        merkle_chain: Provenance chain
        status: Contract status
    """

    contract_id: str
    candidate: ReinjectionCandidate
    required_approvers: list[str]
    approvals: list[ApprovalRecord] = field(default_factory=list)
    execution_log: list[dict[str, Any]] = field(default_factory=list)
    merkle_chain: MerkleChain = field(default_factory=MerkleChain)
    status: ReinjectionContractStatus = ReinjectionContractStatus.DRAFT
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    committed_at: str | None = None
    rollback_id: str | None = None

    def __post_init__(self):
        """Initialize contract."""
        self.merkle_chain.add_event(
            "contract_created",
            {
                "contract_id": self.contract_id,
                "candidate_id": self.candidate.candidate_id,
                "required_approvers": self.required_approvers,
            },
        )
        self._log_execution("contract_initialized")

    def submit(self) -> bool:
        """Submit contract for sandbox testing.

        Returns:
            True if submission succeeded
        """
        if self.status != ReinjectionContractStatus.DRAFT:
            return False

        self.status = ReinjectionContractStatus.SUBMITTED
        self._log_execution("contract_submitted")
        self.merkle_chain.add_event(
            "contract_submitted",
            {"contract_id": self.contract_id},
        )
        return True

    def enter_z1_sandbox(self) -> bool:
        """Move contract to Z1 sandbox testing.

        Returns:
            True if transition succeeded
        """
        if self.status != ReinjectionContractStatus.SUBMITTED:
            return False

        self.status = ReinjectionContractStatus.Z1_SANDBOX
        self._log_execution("entered_z1_sandbox")
        self.merkle_chain.add_event(
            "z1_sandbox_entered",
            {"contract_id": self.contract_id},
        )
        return True

    def request_approval(self) -> bool:
        """Move to approval-pending state after sandbox success.

        Returns:
            True if transition succeeded
        """
        if self.status != ReinjectionContractStatus.Z1_SANDBOX:
            return False

        self.status = ReinjectionContractStatus.AWAITING_APPROVAL
        self._log_execution("approval_requested")
        self.merkle_chain.add_event(
            "approval_requested",
            {
                "contract_id": self.contract_id,
                "required_approvers": self.required_approvers,
            },
        )
        return True

    def add_approval(
        self,
        approver_id: str,
        decision: str,
        reason: str,
        scope: list[str] | None = None,
    ) -> ApprovalRecord:
        """Add an approval decision.

        Args:
            approver_id: ID of approver
            decision: "approve" or "reject"
            reason: Reason for decision
            scope: Scope of approval (defaults to full)

        Returns:
            Created ApprovalRecord
        """
        if approver_id not in self.required_approvers:
            raise ValueError(f"Approver {approver_id} not in required approvers")

        if any(a.approver_id == approver_id for a in self.approvals):
            raise ValueError(f"Approver {approver_id} has already submitted decision")

        approval = ApprovalRecord(
            approver_id=approver_id,
            decision=decision,
            reason=reason,
            scope=scope or ["full"],
        )

        self.approvals.append(approval)
        self._log_execution(f"approval_added_{decision}", {"approver": approver_id})
        self.merkle_chain.add_event(
            "approval_added",
            {
                "contract_id": self.contract_id,
                "approver_id": approver_id,
                "decision": decision,
            },
        )

        # Check if all approvals received
        self._check_approval_status()

        return approval

    def _check_approval_status(self) -> None:
        """Check if all required approvals are received."""
        if self.status != ReinjectionContractStatus.AWAITING_APPROVAL:
            return

        # Check if all required approvers have submitted
        approver_ids = {a.approver_id for a in self.approvals}
        if not set(self.required_approvers).issubset(approver_ids):
            return  # Still waiting for approvals

        # Check if any rejections
        rejections = [a for a in self.approvals if a.decision == "reject"]
        if rejections:
            self.status = ReinjectionContractStatus.REJECTED
            self._log_execution("contract_rejected")
            self.merkle_chain.add_event(
                "contract_rejected",
                {
                    "contract_id": self.contract_id,
                    "rejection_count": len(rejections),
                },
            )

    def commit_z2(self) -> bool:
        """Commit contract to Z2 (production).

        Returns:
            True if commit succeeded
        """
        if self.status != ReinjectionContractStatus.AWAITING_APPROVAL:
            return False

        # Verify all approvals are positive
        for approver_id in self.required_approvers:
            approval = next(
                (a for a in self.approvals if a.approver_id == approver_id), None
            )
            if not approval or approval.decision != "approve":
                return False

        self.status = ReinjectionContractStatus.Z2_COMMITTED
        self.committed_at = datetime.now(timezone.utc).isoformat()
        self.rollback_id = f"rollback_{self.contract_id}"

        self._log_execution("z2_committed")
        self.merkle_chain.add_event(
            "z2_committed",
            {
                "contract_id": self.contract_id,
                "rollback_id": self.rollback_id,
                "committed_at": self.committed_at,
            },
        )

        # Update candidate status
        self.candidate.status = ReinjectionStatus.COMMITTED

        return True

    def rollback(self, reason: str, actor_id: str) -> bool:
        """Rollback a committed contract.

        Args:
            reason: Reason for rollback
            actor_id: Actor initiating rollback

        Returns:
            True if rollback succeeded
        """
        if self.status != ReinjectionContractStatus.Z2_COMMITTED:
            return False

        self.status = ReinjectionContractStatus.ROLLED_BACK
        self._log_execution("rolled_back", {"reason": reason, "actor": actor_id})
        self.merkle_chain.add_event(
            "contract_rolled_back",
            {
                "contract_id": self.contract_id,
                "reason": reason,
                "actor_id": actor_id,
            },
        )

        # Update candidate status
        self.candidate.status = ReinjectionStatus.ROLLED_BACK

        return True

    def _log_execution(self, event: str, details: dict[str, Any] | None = None) -> None:
        """Log an execution event."""
        self.execution_log.append(
            {
                "event": event,
                "details": details or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def compute_hash(self) -> str:
        """Compute hash of contract."""
        content = {
            "contract_id": self.contract_id,
            "candidate_hash": self.candidate.compute_hash(),
            "required_approvers": self.required_approvers,
            "status": self.status.value,
        }
        return hashlib.sha3_256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def is_approved(self) -> bool:
        """Check if contract has all required approvals."""
        if not self.required_approvers:
            return True

        approved_by = {
            a.approver_id for a in self.approvals if a.decision == "approve"
        }
        return set(self.required_approvers).issubset(approved_by)

    def to_dict(self) -> dict[str, Any]:
        """Serialize contract."""
        return {
            "contract_id": self.contract_id,
            "candidate": self.candidate.to_dict(),
            "required_approvers": self.required_approvers,
            "approvals": [a.to_dict() for a in self.approvals],
            "execution_log": self.execution_log,
            "status": self.status.value,
            "created_at": self.created_at,
            "committed_at": self.committed_at,
            "rollback_id": self.rollback_id,
            "contract_hash": self.compute_hash(),
            "merkle_proof": self.merkle_chain.get_chain_proof(),
        }


def create_reinjection_contract(
    candidate: ReinjectionCandidate,
    required_approvers: list[str] | None = None,
) -> ReinjectionContract:
    """Factory function to create a reinjection contract.

    Args:
        candidate: Reinjection candidate
        required_approvers: List of required approvers for dual-control

    Returns:
        Configured ReinjectionContract
    """
    contract_id = f"rc_{candidate.candidate_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # Determine required approvers based on validation level
    if required_approvers is None:
        if candidate.validation_level == ValidationLevel.ENHANCED:
            required_approvers = ["primary_reviewer", "secondary_reviewer"]
        elif candidate.validation_level == ValidationLevel.CRITICAL:
            required_approvers = ["primary_reviewer", "secondary_reviewer", "board_member"]
        else:
            required_approvers = ["primary_reviewer"]

    return ReinjectionContract(
        contract_id=contract_id,
        candidate=candidate,
        required_approvers=required_approvers,
    )
