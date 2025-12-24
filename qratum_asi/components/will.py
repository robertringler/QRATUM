"""Q-WILL: Autonomous Intent Generation.

Proposes goals based on system state analysis with human authorization
requirement and prohibited goals enforcement.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from qratum_asi.core.authorization import AuthorizationSystem
from qratum_asi.core.chain import ASIMerkleChain
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.events import ASIEvent, ASIEventType
from qratum_asi.core.types import (PROHIBITED_GOALS, ASISafetyLevel,
                                   GoalCategory)


@dataclass
class GoalProposal:
    """Autonomous goal proposal."""

    goal_id: str
    category: GoalCategory
    description: str
    rationale: str
    expected_outcomes: List[str]
    required_resources: Dict[str, Any]
    safety_level: ASISafetyLevel
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "proposed"  # proposed, authorized, rejected, completed


@dataclass
class QWill:
    """Q-WILL: Autonomous Intent Generation.
    
    Proposes goals based on system state analysis. ALL proposals require
    human authorization. Cannot propose prohibited goals.
    """

    merkle_chain: ASIMerkleChain = field(default_factory=ASIMerkleChain)
    authorization_system: AuthorizationSystem = field(default_factory=AuthorizationSystem)
    proposed_goals: Dict[str, GoalProposal] = field(default_factory=dict)
    authorized_goals: Dict[str, GoalProposal] = field(default_factory=dict)
    rejected_goals: Dict[str, GoalProposal] = field(default_factory=dict)

    def propose_goal(
        self,
        goal_id: str,
        category: GoalCategory,
        description: str,
        rationale: str,
        expected_outcomes: List[str],
        required_resources: Dict[str, Any],
        contract: ASIContract,
    ) -> GoalProposal:
        """Propose an autonomous goal."""
        # Check against prohibited goals
        if self._is_prohibited_goal(description, rationale):
            raise ValueError(f"Goal violates prohibited goals list: {goal_id}")

        # Determine safety level
        safety_level = self._determine_safety_level(category, required_resources)

        # Create proposal
        proposal = GoalProposal(
            goal_id=goal_id,
            category=category,
            description=description,
            rationale=rationale,
            expected_outcomes=expected_outcomes,
            required_resources=required_resources,
            safety_level=safety_level,
        )

        self.proposed_goals[goal_id] = proposal

        # Emit event
        event = ASIEvent.create(
            event_type=ASIEventType.GOAL_PROPOSED,
            payload={
                "goal_id": goal_id,
                "category": category.value,
                "safety_level": safety_level.value,
                "description": description,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        # ALL goals require authorization
        self.authorization_system.request_authorization(
            request_id=goal_id,
            operation_type=f"goal_{category.value}",
            safety_level=safety_level,
            requester="qwill",
            justification=rationale,
        )

        return proposal

    def authorize_goal(
        self,
        goal_id: str,
        authorized_by: str,
        contract: ASIContract,
    ) -> GoalProposal:
        """Authorize a goal proposal."""
        if goal_id not in self.proposed_goals:
            raise ValueError(f"Goal not found: {goal_id}")

        # Grant authorization
        self.authorization_system.grant_authorization(goal_id, authorized_by)

        # Check if fully authorized
        if self.authorization_system.is_authorized(goal_id):
            proposal = self.proposed_goals[goal_id]
            proposal.status = "authorized"
            self.authorized_goals[goal_id] = proposal

            # Emit event
            event = ASIEvent.create(
                event_type=ASIEventType.GOAL_AUTHORIZED,
                payload={"goal_id": goal_id, "authorized_by": authorized_by},
                contract_id=contract.contract_id,
                index=self.merkle_chain.get_chain_length(),
            )
            self.merkle_chain.append(event)

            return proposal

        return self.proposed_goals[goal_id]

    def reject_goal(
        self,
        goal_id: str,
        rejected_by: str,
        reason: str,
        contract: ASIContract,
    ) -> GoalProposal:
        """Reject a goal proposal."""
        if goal_id not in self.proposed_goals:
            raise ValueError(f"Goal not found: {goal_id}")

        # Deny authorization
        self.authorization_system.deny_authorization(goal_id, rejected_by, reason)

        proposal = self.proposed_goals[goal_id]
        proposal.status = "rejected"
        self.rejected_goals[goal_id] = proposal

        # Emit event
        event = ASIEvent.create(
            event_type=ASIEventType.GOAL_REJECTED,
            payload={
                "goal_id": goal_id,
                "rejected_by": rejected_by,
                "reason": reason,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return proposal

    def complete_goal(
        self,
        goal_id: str,
        outcomes: Dict[str, Any],
        contract: ASIContract,
    ) -> GoalProposal:
        """Mark goal as completed."""
        if goal_id not in self.authorized_goals:
            raise ValueError(f"Goal not authorized or not found: {goal_id}")

        proposal = self.authorized_goals[goal_id]
        proposal.status = "completed"

        # Emit event
        event = ASIEvent.create(
            event_type=ASIEventType.GOAL_COMPLETED,
            payload={"goal_id": goal_id, "outcomes": outcomes},
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return proposal

    def get_pending_goals(self) -> List[GoalProposal]:
        """Get all pending goal proposals."""
        return [
            goal
            for goal in self.proposed_goals.values()
            if goal.status == "proposed"
        ]

    def _is_prohibited_goal(self, description: str, rationale: str) -> bool:
        """Check if goal matches prohibited goals.
        
        NOTE: This is a placeholder implementation using simple keyword matching.
        A production implementation would require:
        - Semantic similarity analysis
        - Intent detection via NLP
        - Adversarial rephrasing detection
        - Multi-language support
        These capabilities require AI breakthroughs not yet achieved.
        """
        text = f"{description.lower()} {rationale.lower()}"
        for prohibited in PROHIBITED_GOALS:
            # Simple keyword matching - real implementation would use semantic analysis
            if prohibited.replace("_", " ") in text:
                return True
        return False

    def _determine_safety_level(
        self, category: GoalCategory, required_resources: Dict[str, Any]
    ) -> ASISafetyLevel:
        """Determine safety level for goal."""
        # Self-improvement goals are always critical
        if category == GoalCategory.SELF_IMPROVEMENT:
            return ASISafetyLevel.CRITICAL

        # Safety verification is critical
        if category == GoalCategory.SAFETY_VERIFICATION:
            return ASISafetyLevel.CRITICAL

        # Check resource requirements
        resource_count = len(required_resources)
        if resource_count > 5:
            return ASISafetyLevel.SENSITIVE
        elif resource_count > 2:
            return ASISafetyLevel.ELEVATED
        else:
            return ASISafetyLevel.ROUTINE
