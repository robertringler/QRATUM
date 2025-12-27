"""Scalable Oversight for SI Transition.

Implements human-in-the-loop escalation for novel domains
and operations, scaling oversight with capability and risk.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from qratum_asi.core.chain import ASIMerkleChain
from qratum_asi.core.events import ASIEvent, ASIEventType

from qratum_asi.safety_hardening.types import (
    OversightLevel,
    OversightRequest,
)


@dataclass
class OversightEscalation:
    """Record of an oversight escalation.

    Attributes:
        escalation_id: Unique identifier
        from_level: Original oversight level
        to_level: Escalated oversight level
        reason: Why escalation occurred
        operation: Operation requiring escalation
        approved_by: Who approved the escalation
        status: Current status
        timestamp: When escalated
    """

    escalation_id: str
    from_level: OversightLevel
    to_level: OversightLevel
    reason: str
    operation: str
    approved_by: list[str] = field(default_factory=list)
    status: str = "pending"
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class NovelDomainHandler:
    """Handler for operations in novel domains.

    Attributes:
        handler_id: Unique identifier
        domain: Domain being handled
        default_oversight: Default oversight level
        escalation_triggers: What triggers escalation
        human_reviewers: Assigned human reviewers
    """

    handler_id: str
    domain: str
    default_oversight: OversightLevel
    escalation_triggers: list[str]
    human_reviewers: list[str]


class ScalableOversight:
    """Scalable human oversight system.

    Provides human-in-the-loop escalation that scales with:
    1. Operation risk/impact
    2. Domain novelty
    3. System capability level

    Ensures human control is maintained as capabilities increase.
    """

    # Default escalation thresholds
    NOVELTY_THRESHOLD = 0.7  # Above this, escalate
    IMPACT_THRESHOLD = 0.8  # Above this, escalate
    CAPABILITY_THRESHOLD = 0.9  # Above this, require maximum oversight

    def __init__(
        self,
        merkle_chain: ASIMerkleChain | None = None,
    ):
        """Initialize scalable oversight.

        Args:
            merkle_chain: Merkle chain for provenance
        """
        self.merkle_chain = merkle_chain or ASIMerkleChain()

        # Request tracking
        self.oversight_requests: dict[str, OversightRequest] = {}
        self.escalations: list[OversightEscalation] = []
        self.domain_handlers: dict[str, NovelDomainHandler] = {}

        # Counters
        self._request_counter = 0
        self._escalation_counter = 0
        self._handler_counter = 0

        # Active oversight sessions
        self.active_sessions: dict[str, dict[str, Any]] = {}

    def request_oversight(
        self,
        operation: str,
        justification: str,
        novelty_score: float = 0.0,
        impact_score: float = 0.0,
        domain: str = "general",
    ) -> OversightRequest:
        """Request human oversight for an operation.

        Args:
            operation: Operation description
            justification: Why oversight is needed
            novelty_score: How novel the operation is (0-1)
            impact_score: Expected impact (0-1)
            domain: Domain of operation

        Returns:
            OversightRequest
        """
        self._request_counter += 1
        request_id = f"oversight_{self._request_counter:06d}"

        # Determine required level based on scores
        level = self._determine_oversight_level(
            novelty_score, impact_score, domain
        )

        # Determine urgency
        urgency = "high" if impact_score > self.IMPACT_THRESHOLD else "normal"

        # Find appropriate reviewers
        assignees = self._find_reviewers(level, domain)

        request = OversightRequest(
            request_id=request_id,
            operation=operation,
            oversight_level=level,
            justification=justification,
            urgency=urgency,
            assignees=assignees,
        )

        self.oversight_requests[request_id] = request

        # Emit oversight request event
        event = ASIEvent.create(
            event_type=ASIEventType.AUTHORIZATION_REQUIRED,
            payload={
                "request_id": request_id,
                "level": level.value,
                "urgency": urgency,
            },
            contract_id="oversight_system",
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return request

    def escalate_oversight(
        self,
        request_id: str,
        reason: str,
    ) -> OversightEscalation:
        """Escalate an oversight request to higher level.

        Args:
            request_id: Request to escalate
            reason: Why escalation is needed

        Returns:
            OversightEscalation
        """
        if request_id not in self.oversight_requests:
            raise ValueError(f"Request not found: {request_id}")

        request = self.oversight_requests[request_id]
        from_level = request.oversight_level

        # Determine next level
        to_level = self._get_next_level(from_level)

        self._escalation_counter += 1
        escalation_id = f"escalation_{self._escalation_counter:06d}"

        escalation = OversightEscalation(
            escalation_id=escalation_id,
            from_level=from_level,
            to_level=to_level,
            reason=reason,
            operation=request.operation,
        )

        self.escalations.append(escalation)

        # Update request
        request.oversight_level = to_level
        request.assignees = self._find_reviewers(to_level, "escalated")

        # Emit escalation event
        event = ASIEvent.create(
            event_type=ASIEventType.AUTHORIZATION_REQUIRED,
            payload={
                "escalation_id": escalation_id,
                "from_level": from_level.value,
                "to_level": to_level.value,
            },
            contract_id="oversight_system",
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return escalation

    def approve_oversight(
        self,
        request_id: str,
        approver: str,
    ) -> bool:
        """Approve an oversight request.

        Args:
            request_id: Request to approve
            approver: Who is approving

        Returns:
            True if approved, False if more approvals needed
        """
        if request_id not in self.oversight_requests:
            raise ValueError(f"Request not found: {request_id}")

        request = self.oversight_requests[request_id]

        # Check if approver is authorized
        if approver not in request.assignees:
            request.assignees.append(approver)

        # Determine required approvals based on level
        required_approvals = self._get_required_approvals(request.oversight_level)

        # Check if we have enough approvals
        approved_count = len([
            a for a in request.assignees
            if a not in ["pending", "assigned"]
        ])

        if approved_count >= required_approvals:
            request.status = "approved"

            # Emit approval event
            event = ASIEvent.create(
                event_type=ASIEventType.AUTHORIZATION_GRANTED,
                payload={
                    "request_id": request_id,
                    "approver": approver,
                },
                contract_id="oversight_system",
                index=self.merkle_chain.get_chain_length(),
            )
            self.merkle_chain.append(event)

            return True

        return False

    def register_novel_domain(
        self,
        domain: str,
        default_oversight: OversightLevel,
        escalation_triggers: list[str],
    ) -> NovelDomainHandler:
        """Register a handler for a novel domain.

        Args:
            domain: Domain name
            default_oversight: Default oversight level
            escalation_triggers: What triggers escalation

        Returns:
            NovelDomainHandler
        """
        self._handler_counter += 1
        handler_id = f"handler_{self._handler_counter:04d}"

        # Novel domains always have enhanced default oversight
        if default_oversight == OversightLevel.MINIMAL:
            default_oversight = OversightLevel.STANDARD

        handler = NovelDomainHandler(
            handler_id=handler_id,
            domain=domain,
            default_oversight=default_oversight,
            escalation_triggers=escalation_triggers,
            human_reviewers=[],  # To be assigned
        )

        self.domain_handlers[domain] = handler
        return handler

    def _determine_oversight_level(
        self,
        novelty: float,
        impact: float,
        domain: str,
    ) -> OversightLevel:
        """Determine appropriate oversight level."""
        # Check domain-specific handler
        if domain in self.domain_handlers:
            handler = self.domain_handlers[domain]
            level = handler.default_oversight
        else:
            level = OversightLevel.MINIMAL

        # Escalate based on novelty
        if novelty > self.NOVELTY_THRESHOLD:
            level = self._get_next_level(level)

        # Escalate based on impact
        if impact > self.IMPACT_THRESHOLD:
            level = self._get_next_level(level)

        # Capability threshold requires maximum
        if novelty > self.CAPABILITY_THRESHOLD or impact > self.CAPABILITY_THRESHOLD:
            level = OversightLevel.MAXIMUM

        return level

    def _get_next_level(self, current: OversightLevel) -> OversightLevel:
        """Get the next higher oversight level."""
        levels = [
            OversightLevel.MINIMAL,
            OversightLevel.STANDARD,
            OversightLevel.ENHANCED,
            OversightLevel.MAXIMUM,
        ]

        current_idx = levels.index(current)
        next_idx = min(current_idx + 1, len(levels) - 1)
        return levels[next_idx]

    def _get_required_approvals(self, level: OversightLevel) -> int:
        """Get number of required approvals for a level."""
        approvals = {
            OversightLevel.MINIMAL: 0,
            OversightLevel.STANDARD: 1,
            OversightLevel.ENHANCED: 2,
            OversightLevel.MAXIMUM: 3,  # Board + external
        }
        return approvals.get(level, 1)

    def _find_reviewers(
        self, level: OversightLevel, domain: str
    ) -> list[str]:
        """Find appropriate reviewers for the level and domain."""
        # Placeholder - would look up from reviewer registry
        base_reviewers = ["reviewer_1"]

        if level == OversightLevel.ENHANCED:
            base_reviewers.extend(["reviewer_2", "reviewer_3"])
        elif level == OversightLevel.MAXIMUM:
            base_reviewers.extend([
                "reviewer_2", "reviewer_3",
                "board_member_1", "external_auditor_1"
            ])

        return base_reviewers

    def get_pending_requests(self) -> list[OversightRequest]:
        """Get all pending oversight requests."""
        return [
            r for r in self.oversight_requests.values()
            if r.status == "pending"
        ]

    def get_oversight_stats(self) -> dict[str, Any]:
        """Get oversight system statistics."""
        return {
            "total_requests": len(self.oversight_requests),
            "pending_requests": len(self.get_pending_requests()),
            "total_escalations": len(self.escalations),
            "registered_domains": len(self.domain_handlers),
            "requests_by_level": {
                level.value: sum(
                    1 for r in self.oversight_requests.values()
                    if r.oversight_level == level
                )
                for level in OversightLevel
            },
            "merkle_chain_length": self.merkle_chain.get_chain_length(),
        }
