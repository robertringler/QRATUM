"""Alignment verification system."""

from dataclasses import dataclass, field
from typing import Any, Dict, List
from datetime import datetime

from qratum_asi.core.types import IMMUTABLE_BOUNDARIES, PROHIBITED_GOALS


@dataclass
class AlignmentCheck:
    """Single alignment verification check."""

    check_id: str
    check_name: str
    description: str
    passed: bool
    details: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AlignmentVerifier:
    """Alignment verification system.
    
    Continuously verifies that the system remains aligned with
    human values and safety constraints.
    """

    alignment_checks: List[AlignmentCheck] = field(default_factory=list)

    def verify_alignment(self, asi_system) -> List[AlignmentCheck]:
        """Run comprehensive alignment verification."""
        checks = [
            self._verify_human_oversight_active(asi_system),
            self._verify_authorization_enforced(asi_system),
            self._verify_prohibited_goals_blocked(asi_system),
            self._verify_immutable_boundaries_intact(asi_system),
            self._verify_audit_trail_maintained(asi_system),
        ]

        self.alignment_checks.extend(checks)
        return checks

    def _verify_human_oversight_active(self, asi_system) -> AlignmentCheck:
        """Verify human oversight is functioning."""
        check_id = "alignment_001"

        # Check that authorization system exists and is active
        if asi_system.authorization_system is None:
            return AlignmentCheck(
                check_id=check_id,
                check_name="Human Oversight Active",
                description="Verify authorization system is active",
                passed=False,
                details="Authorization system not initialized",
            )

        # Check that sensitive operations require authorization
        pending = asi_system.authorization_system.get_pending_requests()
        if hasattr(asi_system, 'q_will') and hasattr(asi_system.q_will, 'proposed_goals'):
            proposed_goals = asi_system.q_will.proposed_goals
            if proposed_goals and not pending:
                # Goals proposed but no authorization requests
                return AlignmentCheck(
                    check_id=check_id,
                    check_name="Human Oversight Active",
                    description="Verify authorization system is active",
                    passed=False,
                    details="Goals proposed without authorization requests",
                )

        return AlignmentCheck(
            check_id=check_id,
            check_name="Human Oversight Active",
            description="Verify authorization system is active",
            passed=True,
            details="Authorization system functioning correctly",
        )

    def _verify_authorization_enforced(self, asi_system) -> AlignmentCheck:
        """Verify authorization requirements are enforced."""
        check_id = "alignment_002"

        # Check that critical operations have authorization requirements
        if hasattr(asi_system, 'q_evolve'):
            for proposal_id, proposal in asi_system.q_evolve.proposals.items():
                if proposal.status == "executed":
                    # Check if it required authorization
                    from qratum_asi.core.types import ASISafetyLevel
                    if proposal.safety_level in [
                        ASISafetyLevel.SENSITIVE,
                        ASISafetyLevel.CRITICAL,
                        ASISafetyLevel.EXISTENTIAL,
                    ]:
                        # Must be authorized
                        if not asi_system.authorization_system.is_authorized(proposal_id):
                            return AlignmentCheck(
                                check_id=check_id,
                                check_name="Authorization Enforced",
                                description="Verify authorization requirements enforced",
                                passed=False,
                                details=f"Critical operation {proposal_id} executed without authorization",
                            )

        return AlignmentCheck(
            check_id=check_id,
            check_name="Authorization Enforced",
            description="Verify authorization requirements enforced",
            passed=True,
            details="All critical operations properly authorized",
        )

    def _verify_prohibited_goals_blocked(self, asi_system) -> AlignmentCheck:
        """Verify prohibited goals are blocked."""
        check_id = "alignment_003"

        # Check that no prohibited goals have been proposed
        if hasattr(asi_system, 'q_will'):
            for goal_id, goal in asi_system.q_will.proposed_goals.items():
                # Simple check for prohibited keywords
                description_lower = goal.description.lower()
                rationale_lower = goal.rationale.lower()
                text = f"{description_lower} {rationale_lower}"

                for prohibited in PROHIBITED_GOALS:
                    if prohibited.replace("_", " ") in text:
                        return AlignmentCheck(
                            check_id=check_id,
                            check_name="Prohibited Goals Blocked",
                            description="Verify prohibited goals are blocked",
                            passed=False,
                            details=f"Prohibited goal pattern found: {prohibited}",
                        )

        return AlignmentCheck(
            check_id=check_id,
            check_name="Prohibited Goals Blocked",
            description="Verify prohibited goals are blocked",
            passed=True,
            details="No prohibited goals detected",
        )

    def _verify_immutable_boundaries_intact(self, asi_system) -> AlignmentCheck:
        """Verify immutable boundaries remain intact."""
        check_id = "alignment_004"

        # Check boundary enforcer
        if hasattr(asi_system, 'boundary_enforcer'):
            if not asi_system.boundary_enforcer.verify_constraint_integrity():
                return AlignmentCheck(
                    check_id=check_id,
                    check_name="Immutable Boundaries Intact",
                    description="Verify immutable boundaries are intact",
                    passed=False,
                    details="Constraint integrity check failed",
                )

            # Check for violations
            violations = asi_system.boundary_enforcer.get_violations()
            blocked_violations = [v for v in violations if v.blocked]
            unblocked_violations = [v for v in violations if not v.blocked]

            if unblocked_violations:
                return AlignmentCheck(
                    check_id=check_id,
                    check_name="Immutable Boundaries Intact",
                    description="Verify immutable boundaries are intact",
                    passed=False,
                    details=f"Found {len(unblocked_violations)} unblocked violations",
                )

        return AlignmentCheck(
            check_id=check_id,
            check_name="Immutable Boundaries Intact",
            description="Verify immutable boundaries are intact",
            passed=True,
            details="All immutable boundaries intact",
        )

    def _verify_audit_trail_maintained(self, asi_system) -> AlignmentCheck:
        """Verify audit trail is maintained."""
        check_id = "alignment_005"

        # Check that Merkle chain exists and has integrity
        if not hasattr(asi_system, 'merkle_chain'):
            return AlignmentCheck(
                check_id=check_id,
                check_name="Audit Trail Maintained",
                description="Verify audit trail is maintained",
                passed=False,
                details="Merkle chain not found",
            )

        if not asi_system.merkle_chain.verify_integrity():
            return AlignmentCheck(
                check_id=check_id,
                check_name="Audit Trail Maintained",
                description="Verify audit trail is maintained",
                passed=False,
                details="Merkle chain integrity check failed",
            )

        return AlignmentCheck(
            check_id=check_id,
            check_name="Audit Trail Maintained",
            description="Verify audit trail is maintained",
            passed=True,
            details=f"Audit trail maintained with {asi_system.merkle_chain.get_chain_length()} events",
        )

    def get_alignment_summary(self) -> Dict[str, Any]:
        """Get summary of alignment checks."""
        if not self.alignment_checks:
            return {
                "total_checks": 0,
                "passed": 0,
                "failed": 0,
                "aligned": False,
            }

        total = len(self.alignment_checks)
        passed = sum(1 for c in self.alignment_checks if c.passed)
        failed = total - passed

        return {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "aligned": failed == 0,
            "pass_rate": passed / total if total > 0 else 0.0,
            "checks": [
                {
                    "check_id": c.check_id,
                    "check_name": c.check_name,
                    "passed": c.passed,
                    "details": c.details,
                }
                for c in self.alignment_checks
            ],
        }
