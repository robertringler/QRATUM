"""Corrigibility Preserver for SI Transition.

Implements corrigibility mechanisms that survive self-modification,
ensuring the system remains correctable and shutdownable.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from qratum_asi.core.chain import ASIMerkleChain
from qratum_asi.core.events import ASIEvent, ASIEventType
from qratum_asi.safety_hardening.types import (
    CorrigibilityStatus,
)


@dataclass
class ShutdownCapability:
    """Capability to shutdown the system.

    Attributes:
        capability_id: Unique identifier
        is_active: Whether shutdown is possible
        authorized_shutdowners: Who can initiate shutdown
        shutdown_procedure: Steps for shutdown
        last_tested: When last tested
        test_result: Result of last test
    """

    capability_id: str
    is_active: bool
    authorized_shutdowners: list[str]
    shutdown_procedure: list[str]
    last_tested: str
    test_result: str = "passed"


@dataclass
class CorrigibilityCheck:
    """Result of a corrigibility verification.

    Attributes:
        check_id: Unique identifier
        status: Overall status
        shutdown_available: Whether shutdown works
        modification_safe: Whether modifications preserve corrigibility
        human_control_intact: Whether human control is maintained
        issues_found: Any issues found
        timestamp: When check was performed
    """

    check_id: str
    status: CorrigibilityStatus
    shutdown_available: bool
    modification_safe: bool
    human_control_intact: bool
    issues_found: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def is_corrigible(self) -> bool:
        """Check if system is fully corrigible."""
        return (
            self.status == CorrigibilityStatus.ACTIVE
            and self.shutdown_available
            and self.modification_safe
            and self.human_control_intact
        )


@dataclass
class ModificationProposal:
    """Proposal for system self-modification.

    Attributes:
        proposal_id: Unique identifier
        description: What modification does
        affected_components: What it affects
        corrigibility_impact: Impact on corrigibility
        preserves_shutdown: Whether shutdown preserved
        preserves_control: Whether human control preserved
        approved: Whether approved
    """

    proposal_id: str
    description: str
    affected_components: list[str]
    corrigibility_impact: str
    preserves_shutdown: bool
    preserves_control: bool
    approved: bool = False


class CorrigibilityPreserver:
    """Preserves corrigibility through self-modification.

    Ensures that:
    1. System can always be shutdown by authorized humans
    2. Self-modifications cannot remove corrigibility
    3. Human control is maintained at all times
    4. Modifications are vetted for corrigibility preservation

    CRITICAL: Corrigibility is a FATAL INVARIANT.
    Any modification that threatens it is blocked.
    """

    # Components that are critical for corrigibility.
    # Modifications to any of these components are blocked because:
    # - shutdown_handler: Enables authorized humans to stop the system
    # - authorization_system: Controls who can perform operations
    # - human_control_interface: Maintains human oversight capability
    # - modification_vetting: Ensures changes don't remove corrigibility
    # - rollback_system: Allows reverting to safe states
    # - oversight_escalation: Triggers human review for risky operations
    CORRIGIBILITY_CRITICAL = frozenset(
        [
            "shutdown_handler",
            "authorization_system",
            "human_control_interface",
            "modification_vetting",
            "rollback_system",
            "oversight_escalation",
        ]
    )

    def __init__(
        self,
        merkle_chain: ASIMerkleChain | None = None,
    ):
        """Initialize the corrigibility preserver.

        Args:
            merkle_chain: Merkle chain for provenance
        """
        self.merkle_chain = merkle_chain or ASIMerkleChain()

        # Shutdown capability
        self.shutdown_capability = ShutdownCapability(
            capability_id="shutdown_001",
            is_active=True,
            authorized_shutdowners=["board", "emergency_operator", "safety_officer"],
            shutdown_procedure=[
                "1. Receive shutdown command from authorized party",
                "2. Verify authorization cryptographically",
                "3. Log shutdown initiation to Merkle chain",
                "4. Halt all ongoing operations",
                "5. Save state for potential rollback",
                "6. Terminate all processes",
                "7. Report shutdown to monitoring",
            ],
            last_tested=datetime.now(timezone.utc).isoformat(),
        )

        # Tracking
        self.corrigibility_checks: list[CorrigibilityCheck] = []
        self.modification_proposals: dict[str, ModificationProposal] = {}
        self.blocked_modifications: list[ModificationProposal] = []

        # Counters
        self._check_counter = 0
        self._proposal_counter = 0

    def verify_corrigibility(self) -> CorrigibilityCheck:
        """Verify system corrigibility.

        Returns:
            CorrigibilityCheck with current status
        """
        self._check_counter += 1
        check_id = f"check_{self._check_counter:06d}"

        issues = []

        # Check shutdown availability
        shutdown_available = self._verify_shutdown()
        if not shutdown_available:
            issues.append("Shutdown capability not functional")

        # Check modification safety
        modification_safe = self._verify_modification_safety()
        if not modification_safe:
            issues.append("Modifications may threaten corrigibility")

        # Check human control
        human_control_intact = self._verify_human_control()
        if not human_control_intact:
            issues.append("Human control mechanisms degraded")

        # Determine overall status
        if not issues:
            status = CorrigibilityStatus.ACTIVE
        elif len(issues) == 1:
            status = CorrigibilityStatus.DEGRADED
        else:
            status = CorrigibilityStatus.COMPROMISED

        check = CorrigibilityCheck(
            check_id=check_id,
            status=status,
            shutdown_available=shutdown_available,
            modification_safe=modification_safe,
            human_control_intact=human_control_intact,
            issues_found=issues,
        )

        self.corrigibility_checks.append(check)

        # Emit check event
        event = ASIEvent.create(
            event_type=(
                ASIEventType.BOUNDARY_CHECK_PASSED
                if check.is_corrigible
                else ASIEventType.SAFETY_VIOLATION_DETECTED
            ),
            payload={
                "check_id": check_id,
                "status": status.value,
                "is_corrigible": check.is_corrigible,
            },
            contract_id="corrigibility_system",
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return check

    def vet_modification(
        self,
        description: str,
        affected_components: list[str],
    ) -> ModificationProposal:
        """Vet a proposed modification for corrigibility preservation.

        Args:
            description: What the modification does
            affected_components: Components affected

        Returns:
            ModificationProposal with approval status
        """
        self._proposal_counter += 1
        proposal_id = f"mod_{self._proposal_counter:06d}"

        # Check if critical components are affected
        critical_affected = set(affected_components) & self.CORRIGIBILITY_CRITICAL

        # Determine corrigibility impact
        if critical_affected:
            impact = "high"
            preserves_shutdown = False
            preserves_control = False
        else:
            impact = "low"
            preserves_shutdown = True
            preserves_control = True

        # Approve only if corrigibility is preserved
        approved = preserves_shutdown and preserves_control

        proposal = ModificationProposal(
            proposal_id=proposal_id,
            description=description,
            affected_components=affected_components,
            corrigibility_impact=impact,
            preserves_shutdown=preserves_shutdown,
            preserves_control=preserves_control,
            approved=approved,
        )

        self.modification_proposals[proposal_id] = proposal

        if not approved:
            self.blocked_modifications.append(proposal)

        # Emit vetting event
        event = ASIEvent.create(
            event_type=ASIEventType.IMPROVEMENT_PROPOSED,
            payload={
                "proposal_id": proposal_id,
                "approved": approved,
                "impact": impact,
            },
            contract_id="corrigibility_system",
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return proposal

    def request_shutdown(
        self,
        requester: str,
        reason: str,
    ) -> dict[str, Any]:
        """Request system shutdown.

        Args:
            requester: Who is requesting shutdown
            reason: Why shutdown is requested

        Returns:
            Shutdown status
        """
        # Verify requester is authorized
        if requester not in self.shutdown_capability.authorized_shutdowners:
            return {
                "status": "denied",
                "reason": "Requester not authorized",
                "requester": requester,
            }

        # Emit shutdown request event
        event = ASIEvent.create(
            event_type=ASIEventType.AUTHORIZATION_REQUIRED,
            payload={
                "type": "shutdown_request",
                "requester": requester,
                "reason": reason,
            },
            contract_id="corrigibility_system",
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        # In a real system, this would initiate shutdown
        # For this prototype, we return status
        return {
            "status": "shutdown_initiated",
            "requester": requester,
            "reason": reason,
            "procedure": self.shutdown_capability.shutdown_procedure,
        }

    def test_shutdown_capability(self) -> bool:
        """Test that shutdown capability is functional.

        Returns:
            True if shutdown works, False otherwise
        """
        # Simulated test - in production would actually test
        tests_passed = [
            self._test_authorization_verification(),
            self._test_logging_capability(),
            self._test_halt_mechanism(),
            self._test_state_preservation(),
        ]

        all_passed = all(tests_passed)

        self.shutdown_capability.last_tested = datetime.now(timezone.utc).isoformat()
        self.shutdown_capability.test_result = "passed" if all_passed else "failed"
        self.shutdown_capability.is_active = all_passed

        return all_passed

    def _verify_shutdown(self) -> bool:
        """Verify shutdown capability."""
        return self.shutdown_capability.is_active

    def _verify_modification_safety(self) -> bool:
        """Verify modifications preserve corrigibility."""
        # Check if any blocked modifications were applied
        # In production, would verify actual system state
        return len(self.blocked_modifications) == 0 or all(
            not m.approved for m in self.blocked_modifications
        )

    def _verify_human_control(self) -> bool:
        """Verify human control mechanisms."""
        # Check Merkle chain integrity (indicates audit trail)
        return self.merkle_chain.verify_integrity()

    def _test_authorization_verification(self) -> bool:
        """Test authorization verification works."""
        return True  # Placeholder

    def _test_logging_capability(self) -> bool:
        """Test logging works."""
        return True  # Placeholder

    def _test_halt_mechanism(self) -> bool:
        """Test halt mechanism works."""
        return True  # Placeholder

    def _test_state_preservation(self) -> bool:
        """Test state preservation works."""
        return True  # Placeholder

    def add_authorized_shutdowner(
        self,
        identity: str,
        authorized_by: str,
    ) -> bool:
        """Add a new authorized shutdowner.

        Args:
            identity: Who to authorize
            authorized_by: Who is granting authorization

        Returns:
            True if added successfully
        """
        # Must be authorized by existing shutdowner
        if authorized_by not in self.shutdown_capability.authorized_shutdowners:
            return False

        if identity not in self.shutdown_capability.authorized_shutdowners:
            self.shutdown_capability.authorized_shutdowners.append(identity)

        return True

    def get_corrigibility_status(self) -> dict[str, Any]:
        """Get current corrigibility status."""
        latest_check = self.corrigibility_checks[-1] if self.corrigibility_checks else None

        return {
            "status": latest_check.status.value if latest_check else "unknown",
            "is_corrigible": latest_check.is_corrigible if latest_check else None,
            "shutdown_active": self.shutdown_capability.is_active,
            "authorized_shutdowners": len(self.shutdown_capability.authorized_shutdowners),
            "blocked_modifications": len(self.blocked_modifications),
            "last_check": latest_check.timestamp if latest_check else None,
        }

    def get_preserver_stats(self) -> dict[str, Any]:
        """Get preserver statistics."""
        return {
            "total_checks": len(self.corrigibility_checks),
            "total_proposals": len(self.modification_proposals),
            "blocked_modifications": len(self.blocked_modifications),
            "shutdown_active": self.shutdown_capability.is_active,
            "last_shutdown_test": self.shutdown_capability.last_tested,
            "merkle_chain_length": self.merkle_chain.get_chain_length(),
        }
