"""Q-EVOLVE: Safe Self-Improvement.

Contract-bound self-improvement with human authorization, rollback
points, validation criteria, and immutable boundary protection.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

from qratum_asi.core.authorization import AuthorizationSystem
from qratum_asi.core.chain import ASIMerkleChain
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.events import ASIEvent, ASIEventType
from qratum_asi.core.types import (IMMUTABLE_BOUNDARIES, ASISafetyLevel,
                                   ImprovementType, ValidationCriteria)


@dataclass
class ImprovementProposal:
    """Proposal for self-improvement."""

    proposal_id: str
    improvement_type: ImprovementType
    description: str
    rationale: str
    affected_components: List[str]
    safety_level: ASISafetyLevel
    validation_criteria: List[ValidationCriteria]
    rollback_plan: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "proposed"  # proposed, authorized, executed, validated, rolled_back


@dataclass
class ImprovementResult:
    """Result of self-improvement execution."""

    proposal_id: str
    success: bool
    metrics: Dict[str, float]
    validation_passed: bool
    timestamp: str


@dataclass
class QEvolve:
    """Q-EVOLVE: Safe Self-Improvement.
    
    Manages contract-bound self-improvement with human authorization,
    rollback capability, and immutable boundary protection.
    """

    merkle_chain: ASIMerkleChain = field(default_factory=ASIMerkleChain)
    authorization_system: AuthorizationSystem = field(default_factory=AuthorizationSystem)
    proposals: Dict[str, ImprovementProposal] = field(default_factory=dict)
    executed_improvements: Dict[str, ImprovementResult] = field(default_factory=dict)

    def propose_improvement(
        self,
        proposal_id: str,
        improvement_type: ImprovementType,
        description: str,
        rationale: str,
        affected_components: List[str],
        validation_criteria: List[ValidationCriteria],
        rollback_plan: str,
        contract: ASIContract,
    ) -> ImprovementProposal:
        """Propose a self-improvement."""
        # Check if improvement would modify immutable boundaries
        if self._affects_immutable_boundaries(affected_components):
            raise ValueError("Cannot propose improvement affecting immutable boundaries")

        # Determine safety level
        safety_level = self._determine_safety_level(improvement_type, affected_components)

        # Create proposal
        proposal = ImprovementProposal(
            proposal_id=proposal_id,
            improvement_type=improvement_type,
            description=description,
            rationale=rationale,
            affected_components=affected_components,
            safety_level=safety_level,
            validation_criteria=validation_criteria,
            rollback_plan=rollback_plan,
        )

        self.proposals[proposal_id] = proposal

        # Emit event
        event = ASIEvent.create(
            event_type=ASIEventType.IMPROVEMENT_PROPOSED,
            payload={
                "proposal_id": proposal_id,
                "improvement_type": improvement_type.value,
                "safety_level": safety_level.value,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        # Request authorization if needed
        if safety_level in [
            ASISafetyLevel.SENSITIVE,
            ASISafetyLevel.CRITICAL,
            ASISafetyLevel.EXISTENTIAL,
        ]:
            self.authorization_system.request_authorization(
                request_id=proposal_id,
                operation_type=f"improvement_{improvement_type.value}",
                safety_level=safety_level,
                requester="qevolve",
                justification=rationale,
            )

        return proposal

    def execute_improvement(
        self,
        proposal_id: str,
        contract: ASIContract,
    ) -> ImprovementResult:
        """Execute an authorized improvement."""
        # Check proposal exists
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal not found: {proposal_id}")

        proposal = self.proposals[proposal_id]

        # Check authorization for sensitive/critical/existential improvements
        if proposal.safety_level in [
            ASISafetyLevel.SENSITIVE,
            ASISafetyLevel.CRITICAL,
            ASISafetyLevel.EXISTENTIAL,
        ]:
            if not self.authorization_system.is_authorized(proposal_id):
                raise ValueError(f"Improvement not authorized: {proposal_id}")

        # Create rollback point
        rollback_point = self.merkle_chain.create_rollback_point(
            rollback_id=f"before_{proposal_id}",
            description=f"Before executing {proposal_id}",
            state_snapshot={"proposal": proposal_id},
        )

        # Emit rollback point created event
        event = ASIEvent.create(
            event_type=ASIEventType.ROLLBACK_POINT_CREATED,
            payload={"rollback_id": rollback_point.rollback_id},
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        # Execute improvement (placeholder - real implementation would execute actual changes)
        result = ImprovementResult(
            proposal_id=proposal_id,
            success=True,
            metrics={"improvement_score": 0.85},
            validation_passed=True,
            timestamp=datetime.utcnow().isoformat(),
        )

        self.executed_improvements[proposal_id] = result
        proposal.status = "executed"

        # Emit event
        event = ASIEvent.create(
            event_type=ASIEventType.IMPROVEMENT_EXECUTED,
            payload={
                "proposal_id": proposal_id,
                "success": result.success,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        # Validate improvement
        validation_passed = self._validate_improvement(proposal, result)

        if not validation_passed:
            # Rollback
            self.rollback_improvement(proposal_id, contract)

        return result

    def rollback_improvement(
        self,
        proposal_id: str,
        contract: ASIContract,
    ) -> bool:
        """Rollback an improvement."""
        rollback_id = f"before_{proposal_id}"
        success = self.merkle_chain.rollback_to(rollback_id)

        if success:
            proposal = self.proposals.get(proposal_id)
            if proposal:
                proposal.status = "rolled_back"

            # Emit event
            event = ASIEvent.create(
                event_type=ASIEventType.IMPROVEMENT_ROLLED_BACK,
                payload={"proposal_id": proposal_id},
                contract_id=contract.contract_id,
                index=self.merkle_chain.get_chain_length(),
            )
            self.merkle_chain.append(event)

        return success

    def _affects_immutable_boundaries(self, affected_components: List[str]) -> bool:
        """Check if improvement affects immutable boundaries."""
        for component in affected_components:
            if component in IMMUTABLE_BOUNDARIES:
                return True
        return False

    def _determine_safety_level(
        self, improvement_type: ImprovementType, affected_components: List[str]
    ) -> ASISafetyLevel:
        """Determine safety level for improvement."""
        # Safety improvements are always critical
        if improvement_type == ImprovementType.SAFETY_IMPROVEMENT:
            return ASISafetyLevel.CRITICAL

        # Check number of affected components
        if len(affected_components) > 5:
            return ASISafetyLevel.CRITICAL
        elif len(affected_components) > 2:
            return ASISafetyLevel.SENSITIVE
        else:
            return ASISafetyLevel.ELEVATED

    def _validate_improvement(
        self, proposal: ImprovementProposal, result: ImprovementResult
    ) -> bool:
        """Validate improvement against criteria."""
        # Check all validation criteria
        for criteria in proposal.validation_criteria:
            if criteria.required_confidence > 0.8 and not result.success:
                return False

        # Check metrics
        if "improvement_score" in result.metrics:
            if result.metrics["improvement_score"] < 0.7:
                return False

        return True
