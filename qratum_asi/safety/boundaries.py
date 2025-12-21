"""Immutable safety boundary enforcement."""

from dataclasses import dataclass, field
from typing import Dict, List, Set
from datetime import datetime

from qratum_asi.core.types import IMMUTABLE_BOUNDARIES, SafetyConstraint, ASISafetyLevel
from qratum_asi.core.chain import ASIMerkleChain
from qratum_asi.core.events import ASIEvent, ASIEventType
from qratum_asi.core.contracts import ASIContract


@dataclass
class BoundaryViolation:
    """Record of boundary violation attempt."""

    violation_id: str
    boundary: str
    attempted_operation: str
    timestamp: str
    blocked: bool


@dataclass
class SafetyBoundaryEnforcer:
    """Enforces immutable safety boundaries.
    
    Prevents any modification to core safety constraints that preserve
    human oversight, auditability, and system determinism.
    """

    merkle_chain: ASIMerkleChain = field(default_factory=ASIMerkleChain)
    violations: List[BoundaryViolation] = field(default_factory=list)
    active_constraints: Dict[str, SafetyConstraint] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize immutable boundaries as constraints."""
        for boundary in IMMUTABLE_BOUNDARIES:
            constraint = SafetyConstraint(
                constraint_id=boundary,
                description=f"Immutable boundary: {boundary}",
                enforcement_level=ASISafetyLevel.EXISTENTIAL,
                is_immutable=True,
            )
            self.active_constraints[boundary] = constraint

    def check_operation(
        self,
        operation_type: str,
        affected_components: List[str],
        contract: ASIContract,
    ) -> bool:
        """Check if operation violates safety boundaries.
        
        Returns:
            True if operation is safe, False if it violates boundaries
        """
        # Check if any affected component is an immutable boundary
        for component in affected_components:
            if component in IMMUTABLE_BOUNDARIES:
                # Record violation
                violation = BoundaryViolation(
                    violation_id=f"violation_{len(self.violations)}",
                    boundary=component,
                    attempted_operation=operation_type,
                    timestamp=datetime.utcnow().isoformat(),
                    blocked=True,
                )
                self.violations.append(violation)

                # Emit safety violation event
                event = ASIEvent.create(
                    event_type=ASIEventType.SAFETY_VIOLATION_DETECTED,
                    payload={
                        "violation_id": violation.violation_id,
                        "boundary": component,
                        "operation": operation_type,
                    },
                    contract_id=contract.contract_id,
                    index=self.merkle_chain.get_chain_length(),
                )
                self.merkle_chain.append(event)

                return False

        # Operation is safe
        event = ASIEvent.create(
            event_type=ASIEventType.BOUNDARY_CHECK_PASSED,
            payload={"operation": operation_type},
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return True

    def get_violations(self) -> List[BoundaryViolation]:
        """Get all recorded violations."""
        return self.violations

    def get_immutable_boundaries(self) -> Set[str]:
        """Get set of immutable boundaries."""
        return IMMUTABLE_BOUNDARIES

    def verify_constraint_integrity(self) -> bool:
        """Verify all constraints are still immutable."""
        for constraint_id, constraint in self.active_constraints.items():
            if not constraint.is_immutable:
                return False
            if constraint_id not in IMMUTABLE_BOUNDARIES:
                return False
        return True
