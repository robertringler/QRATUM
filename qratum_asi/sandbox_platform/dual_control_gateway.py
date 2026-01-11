"""Dual-Control Gateway for Human Approval.

Implements dual-control approval gateway ensuring production is
unaffected until explicit human authorization.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from qradle.merkle import MerkleChain


class AuthorizationStatus(Enum):
    """Status of authorization request."""

    PENDING = "pending"
    AWAITING_SECOND = "awaiting_second"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalLevel(Enum):
    """Level of approval required."""

    SINGLE = "single"  # Single approver
    DUAL = "dual"  # Two independent approvers
    BOARD = "board"  # Board-level review


@dataclass
class ApprovalRecord:
    """Record of an approval decision.

    Attributes:
        record_id: Unique record identifier
        approver_id: Approver's identifier
        decision: Approval decision
        reason: Reason for decision
        timestamp: When decision was made
    """

    record_id: str
    approver_id: str
    decision: str  # "approve" or "reject"
    reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    signature: str = ""

    def compute_hash(self) -> str:
        """Compute hash of approval record."""
        content = {
            "record_id": self.record_id,
            "approver_id": self.approver_id,
            "decision": self.decision,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }
        return hashlib.sha3_256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize approval record."""
        return {
            "record_id": self.record_id,
            "approver_id": self.approver_id,
            "decision": self.decision,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "record_hash": self.compute_hash(),
        }


@dataclass
class ApprovalRequest:
    """Request for approval.

    Attributes:
        request_id: Unique request identifier
        resource_id: ID of resource requiring approval
        resource_type: Type of resource
        description: Description of what's being approved
        approval_level: Level of approval required
        required_approvers: List of required approver IDs
        status: Current authorization status
        approvals: List of approval records
        expires_at: Request expiration time
    """

    request_id: str
    resource_id: str
    resource_type: str
    description: str
    approval_level: ApprovalLevel = ApprovalLevel.DUAL
    required_approvers: list[str] = field(default_factory=list)
    status: AuthorizationStatus = AuthorizationStatus.PENDING
    approvals: list[ApprovalRecord] = field(default_factory=list)
    rejections: list[ApprovalRecord] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: str | None = None
    completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_approved(self) -> bool:
        """Check if request is fully approved."""
        if self.status != AuthorizationStatus.APPROVED:
            return False

        if self.approval_level == ApprovalLevel.SINGLE:
            return len(self.approvals) >= 1
        elif self.approval_level == ApprovalLevel.DUAL:
            return len(self.approvals) >= 2
        elif self.approval_level == ApprovalLevel.BOARD:
            return len(self.approvals) >= 3

        return False

    @property
    def is_rejected(self) -> bool:
        """Check if request is rejected."""
        return len(self.rejections) > 0

    def compute_hash(self) -> str:
        """Compute hash of request content."""
        content = {
            "request_id": self.request_id,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "description": self.description,
            "approval_level": self.approval_level.value,
        }
        return hashlib.sha3_256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize approval request."""
        return {
            "request_id": self.request_id,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "description": self.description,
            "approval_level": self.approval_level.value,
            "required_approvers": self.required_approvers,
            "status": self.status.value,
            "approval_count": len(self.approvals),
            "rejection_count": len(self.rejections),
            "is_approved": self.is_approved,
            "is_rejected": self.is_rejected,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "completed_at": self.completed_at,
            "request_hash": self.compute_hash(),
        }


class DualControlGateway:
    """Gateway for dual-control human approval.

    Ensures:
    - Production is unaffected until explicit authorization
    - Multiple independent approvers for sensitive operations
    - Complete audit trail of all decisions
    - Merkle-chained verification
    """

    def __init__(
        self,
        gateway_id: str = "dual_control",
        default_level: ApprovalLevel = ApprovalLevel.DUAL,
        request_timeout_hours: int = 24,
        merkle_chain: MerkleChain | None = None,
    ):
        """Initialize dual-control gateway.

        Args:
            gateway_id: Unique gateway identifier
            default_level: Default approval level
            request_timeout_hours: Request timeout in hours
            merkle_chain: Merkle chain for audit trail
        """
        self.gateway_id = gateway_id
        self.default_level = default_level
        self.request_timeout_hours = request_timeout_hours
        self.merkle_chain = merkle_chain or MerkleChain()

        # Request management
        self.requests: dict[str, ApprovalRequest] = {}
        self._request_counter = 0
        self._record_counter = 0
        self._lock = threading.RLock()

        # Callbacks
        self._on_approved_callbacks: list[Callable[[ApprovalRequest], None]] = []
        self._on_rejected_callbacks: list[Callable[[ApprovalRequest], None]] = []

        # Statistics
        self._total_requests = 0
        self._approved_requests = 0
        self._rejected_requests = 0
        self._expired_requests = 0

        # Log initialization
        self.merkle_chain.add_event(
            "dual_control_gateway_initialized",
            {
                "gateway_id": gateway_id,
                "default_level": default_level.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def create_request(
        self,
        resource_id: str,
        resource_type: str,
        description: str,
        approval_level: ApprovalLevel | None = None,
        required_approvers: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalRequest:
        """Create an approval request.

        Args:
            resource_id: ID of resource requiring approval
            resource_type: Type of resource
            description: What's being approved
            approval_level: Level of approval required
            required_approvers: Specific approvers required
            metadata: Additional metadata

        Returns:
            Created ApprovalRequest
        """
        with self._lock:
            self._request_counter += 1
            request_id = f"req_{self.gateway_id}_{self._request_counter:08d}"

            level = approval_level or self.default_level

            # Determine required approvers based on level
            if required_approvers is None:
                if level == ApprovalLevel.SINGLE:
                    required_approvers = ["approver_1"]
                elif level == ApprovalLevel.DUAL:
                    required_approvers = ["approver_1", "approver_2"]
                else:
                    required_approvers = ["approver_1", "approver_2", "board_member"]

            request = ApprovalRequest(
                request_id=request_id,
                resource_id=resource_id,
                resource_type=resource_type,
                description=description,
                approval_level=level,
                required_approvers=required_approvers,
                metadata=metadata or {},
            )

            self.requests[request_id] = request
            self._total_requests += 1

            # Log request creation
            self.merkle_chain.add_event(
                "approval_request_created",
                {
                    "request_id": request_id,
                    "resource_id": resource_id,
                    "resource_type": resource_type,
                    "approval_level": level.value,
                },
            )

            return request

    def add_approval(
        self,
        request_id: str,
        approver_id: str,
        decision: str,
        reason: str = "",
    ) -> ApprovalRecord | None:
        """Add an approval decision to a request.

        Args:
            request_id: Request to approve
            approver_id: Approver's identifier
            decision: "approve" or "reject"
            reason: Reason for decision

        Returns:
            ApprovalRecord if added, None if invalid
        """
        with self._lock:
            if request_id not in self.requests:
                return None

            request = self.requests[request_id]

            # Check if request is still pending
            if request.status in (
                AuthorizationStatus.APPROVED,
                AuthorizationStatus.REJECTED,
                AuthorizationStatus.EXPIRED,
            ):
                return None

            # Check if approver already voted
            existing_approvers = {a.approver_id for a in request.approvals}
            existing_rejectors = {r.approver_id for r in request.rejections}
            if approver_id in existing_approvers or approver_id in existing_rejectors:
                return None

            # Create approval record
            self._record_counter += 1
            record_id = f"record_{request_id}_{self._record_counter:06d}"

            record = ApprovalRecord(
                record_id=record_id,
                approver_id=approver_id,
                decision=decision,
                reason=reason,
            )
            record.signature = record.compute_hash()

            if decision == "approve":
                request.approvals.append(record)
                self._update_approval_status(request)
            else:
                request.rejections.append(record)
                request.status = AuthorizationStatus.REJECTED
                request.completed_at = datetime.now(timezone.utc).isoformat()
                self._rejected_requests += 1

                # Notify rejection callbacks
                for callback in self._on_rejected_callbacks:
                    try:
                        callback(request)
                    except Exception:
                        pass

            # Log decision
            self.merkle_chain.add_event(
                "approval_decision",
                {
                    "request_id": request_id,
                    "record_id": record_id,
                    "approver_id": approver_id,
                    "decision": decision,
                },
            )

            return record

    def _update_approval_status(self, request: ApprovalRequest) -> None:
        """Update request status based on approvals."""
        approval_count = len(request.approvals)

        if request.approval_level == ApprovalLevel.SINGLE:
            required = 1
        elif request.approval_level == ApprovalLevel.DUAL:
            required = 2
        else:
            required = 3

        if approval_count >= required:
            request.status = AuthorizationStatus.APPROVED
            request.completed_at = datetime.now(timezone.utc).isoformat()
            self._approved_requests += 1

            # Notify approval callbacks
            for callback in self._on_approved_callbacks:
                try:
                    callback(request)
                except Exception:
                    pass
        elif approval_count == 1 and required > 1:
            request.status = AuthorizationStatus.AWAITING_SECOND

    def check_authorization(self, resource_id: str) -> bool:
        """Check if a resource is authorized.

        Args:
            resource_id: Resource to check

        Returns:
            True if authorized
        """
        for request in self.requests.values():
            if request.resource_id == resource_id and request.is_approved:
                return True
        return False

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        """Get request by ID."""
        return self.requests.get(request_id)

    def get_pending_requests(self) -> list[ApprovalRequest]:
        """Get all pending requests."""
        return [
            r for r in self.requests.values()
            if r.status in (AuthorizationStatus.PENDING, AuthorizationStatus.AWAITING_SECOND)
        ]

    def get_requests_for_approver(self, approver_id: str) -> list[ApprovalRequest]:
        """Get pending requests for a specific approver.

        Args:
            approver_id: Approver to get requests for

        Returns:
            List of pending requests
        """
        return [
            r for r in self.get_pending_requests()
            if approver_id in r.required_approvers
            and approver_id not in {a.approver_id for a in r.approvals}
            and approver_id not in {a.approver_id for a in r.rejections}
        ]

    def register_on_approved(
        self,
        callback: Callable[[ApprovalRequest], None],
    ) -> None:
        """Register callback for approval events."""
        self._on_approved_callbacks.append(callback)

    def register_on_rejected(
        self,
        callback: Callable[[ApprovalRequest], None],
    ) -> None:
        """Register callback for rejection events."""
        self._on_rejected_callbacks.append(callback)

    def get_gateway_stats(self) -> dict[str, Any]:
        """Get gateway statistics."""
        pending_count = len(self.get_pending_requests())
        status_counts: dict[str, int] = {}
        for r in self.requests.values():
            status_counts[r.status.value] = status_counts.get(r.status.value, 0) + 1

        approval_rate = (
            self._approved_requests / self._total_requests
            if self._total_requests > 0
            else 0
        )

        return {
            "gateway_id": self.gateway_id,
            "default_level": self.default_level.value,
            "total_requests": self._total_requests,
            "approved_requests": self._approved_requests,
            "rejected_requests": self._rejected_requests,
            "expired_requests": self._expired_requests,
            "pending_requests": pending_count,
            "approval_rate": approval_rate,
            "status_counts": status_counts,
            "merkle_chain_valid": self.merkle_chain.verify_integrity(),
        }
