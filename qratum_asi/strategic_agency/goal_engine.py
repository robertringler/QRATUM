"""Strategic Goal Engine for SI Transition.

Enables autonomous formulation of long-term, multi-step strategic
objectives with goal decomposition, progress tracking, and
human oversight integration.

Key Features:
- Autonomous objective formulation based on system state
- Hierarchical goal decomposition
- Progress assessment and adaptation
- Safety-bounded goal pursuit
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from qratum_asi.core.authorization import AuthorizationSystem
from qratum_asi.core.chain import ASIMerkleChain
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.events import ASIEvent, ASIEventType
from qratum_asi.core.types import ASISafetyLevel
from qratum_asi.strategic_agency.types import (
    PROHIBITED_OBJECTIVES,
    ObjectivePriority,
    ObjectiveSafetyLevel,
    ObjectiveType,
    StrategicObjective,
    SubObjective,
)


@dataclass
class GoalDecomposition:
    """Decomposition of a strategic objective into sub-objectives.

    Attributes:
        decomposition_id: Unique identifier
        objective_id: Parent objective ID
        sub_objectives: List of sub-objectives
        total_estimated_effort: Total effort estimate
        critical_path: Critical path through sub-objectives
        decomposition_depth: How deep the decomposition goes
        merkle_proof: Cryptographic proof
        timestamp: Decomposition timestamp
    """

    decomposition_id: str
    objective_id: str
    sub_objectives: list[SubObjective]
    total_estimated_effort: float
    critical_path: list[str]
    decomposition_depth: int
    merkle_proof: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ProgressAssessment:
    """Assessment of progress toward an objective.

    Attributes:
        assessment_id: Unique identifier
        objective_id: Objective being assessed
        completion_percentage: Overall completion (0-100)
        sub_objective_status: Status of each sub-objective
        blockers: Current blockers
        risks: Identified risks
        recommended_adjustments: Suggested changes
        confidence_in_completion: Confidence of completion
        estimated_time_remaining: Time estimate
        timestamp: Assessment timestamp
    """

    assessment_id: str
    objective_id: str
    completion_percentage: float
    sub_objective_status: dict[str, str]
    blockers: list[str]
    risks: list[str]
    recommended_adjustments: list[str]
    confidence_in_completion: float
    estimated_time_remaining: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class StrategicGoalEngine:
    """Engine for autonomous strategic goal formulation and pursuit.

    Enables the system to formulate long-term objectives based on
    its understanding of the problem space, decompose them into
    achievable sub-objectives, and track progress toward completion.

    Enforces:
    - All objectives require human approval
    - Prohibited objectives are blocked
    - Safety escalation for sensitive objectives
    - Provenance tracking for all operations
    """

    def __init__(
        self,
        merkle_chain: ASIMerkleChain | None = None,
        authorization_system: AuthorizationSystem | None = None,
    ):
        """Initialize the strategic goal engine.

        Args:
            merkle_chain: Merkle chain for provenance
            authorization_system: Authorization system for approvals
        """
        self.merkle_chain = merkle_chain or ASIMerkleChain()
        self.authorization_system = authorization_system or AuthorizationSystem()

        # Objective storage
        self.objectives: dict[str, StrategicObjective] = {}
        self.decompositions: dict[str, GoalDecomposition] = {}
        self.assessments: dict[str, list[ProgressAssessment]] = {}

        # Counters
        self._objective_counter = 0
        self._decomposition_counter = 0
        self._assessment_counter = 0

    def formulate_objective(
        self,
        objective_type: ObjectiveType,
        title: str,
        description: str,
        rationale: str,
        target_domains: list[str],
        success_criteria: list[str],
        estimated_timeline: str,
        required_resources: dict[str, Any],
        contract: ASIContract,
        priority: ObjectivePriority = ObjectivePriority.MEDIUM,
    ) -> StrategicObjective:
        """Formulate a new strategic objective.

        Args:
            objective_type: Type of objective
            title: Short title
            description: Detailed description
            rationale: Why this matters
            target_domains: Domains involved
            success_criteria: How success is measured
            estimated_timeline: Time estimate
            required_resources: Resources needed
            contract: Executing contract
            priority: Priority level

        Returns:
            Created StrategicObjective

        Raises:
            ValueError: If objective is prohibited or contract invalid
        """
        # Validate contract
        if not contract.validate():
            raise ValueError(f"Invalid contract: {contract.contract_id}")

        # Check for prohibited objectives
        if self._is_prohibited(title, description, rationale):
            raise ValueError("Objective matches prohibited patterns")

        self._objective_counter += 1
        objective_id = f"obj_{self._objective_counter:06d}"

        # Determine safety level
        safety_level = self._determine_safety_level(
            objective_type, target_domains, required_resources
        )

        # Compute provenance hash
        provenance_hash = hashlib.sha3_256(
            json.dumps(
                {
                    "objective_id": objective_id,
                    "title": title,
                    "description": description,
                    "type": objective_type.value,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        objective = StrategicObjective(
            objective_id=objective_id,
            objective_type=objective_type,
            title=title,
            description=description,
            rationale=rationale,
            priority=priority,
            safety_level=safety_level,
            target_domains=target_domains,
            success_criteria=success_criteria,
            estimated_timeline=estimated_timeline,
            required_resources=required_resources,
            human_approval_required=True,  # Always required
            approval_status="pending",
            provenance_hash=provenance_hash,
        )

        # Final safety validation
        if not objective.validate_not_prohibited():
            raise ValueError("Objective failed safety validation")

        self.objectives[objective_id] = objective

        # Emit objective formulated event
        event = ASIEvent.create(
            event_type=ASIEventType.GOAL_PROPOSED,
            payload={
                "objective_id": objective_id,
                "type": objective_type.value,
                "title": title,
                "safety_level": safety_level.value,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        # Request authorization (all objectives require human approval)
        self.authorization_system.request_authorization(
            request_id=objective_id,
            operation_type=f"objective_{objective_type.value}",
            safety_level=self._map_safety_level(safety_level),
            requester="strategic_goal_engine",
            justification=rationale,
        )

        return objective

    def decompose_objective(
        self,
        objective_id: str,
        contract: ASIContract,
        max_depth: int = 3,
    ) -> GoalDecomposition:
        """Decompose an objective into sub-objectives.

        Args:
            objective_id: Objective to decompose
            contract: Executing contract
            max_depth: Maximum decomposition depth

        Returns:
            GoalDecomposition with sub-objectives
        """
        if objective_id not in self.objectives:
            raise ValueError(f"Objective not found: {objective_id}")

        objective = self.objectives[objective_id]

        self._decomposition_counter += 1
        decomposition_id = f"decomp_{self._decomposition_counter:06d}"

        # Generate sub-objectives based on objective type
        sub_objectives = self._generate_sub_objectives(objective, max_depth)

        # Compute critical path
        critical_path = self._compute_critical_path(sub_objectives)

        # Total effort
        total_effort = sum(so.estimated_effort for so in sub_objectives)

        decomposition = GoalDecomposition(
            decomposition_id=decomposition_id,
            objective_id=objective_id,
            sub_objectives=sub_objectives,
            total_estimated_effort=total_effort,
            critical_path=critical_path,
            decomposition_depth=max_depth,
            merkle_proof=self.merkle_chain.chain_hash or "",
        )

        self.decompositions[objective_id] = decomposition

        # Emit decomposition event
        event = ASIEvent.create(
            event_type=ASIEventType.REASONING_COMPLETED,
            payload={
                "decomposition_id": decomposition_id,
                "objective_id": objective_id,
                "sub_objectives_count": len(sub_objectives),
                "total_effort": total_effort,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return decomposition

    def assess_progress(
        self,
        objective_id: str,
        current_status: dict[str, str],
        contract: ASIContract,
    ) -> ProgressAssessment:
        """Assess progress toward an objective.

        Args:
            objective_id: Objective to assess
            current_status: Current status of sub-objectives
            contract: Executing contract

        Returns:
            ProgressAssessment with analysis
        """
        if objective_id not in self.objectives:
            raise ValueError(f"Objective not found: {objective_id}")

        self._assessment_counter += 1
        assessment_id = f"assess_{self._assessment_counter:06d}"

        decomposition = self.decompositions.get(objective_id)

        # Compute completion percentage
        if decomposition:
            completed = sum(
                1
                for so in decomposition.sub_objectives
                if current_status.get(so.sub_id) == "completed"
            )
            completion = (completed / len(decomposition.sub_objectives)) * 100
        else:
            completion = 0.0

        # Identify blockers and risks (placeholder analysis)
        blockers = []
        risks = []
        adjustments = []

        if completion < 25:
            risks.append("Low initial progress may indicate unclear requirements")
        if completion > 75:
            risks.append("Final integration may reveal hidden dependencies")

        for sub_id, status in current_status.items():
            if status == "blocked":
                blockers.append(f"Sub-objective {sub_id} is blocked")

        # Confidence estimation
        confidence = 0.8 if completion > 50 else 0.6

        assessment = ProgressAssessment(
            assessment_id=assessment_id,
            objective_id=objective_id,
            completion_percentage=completion,
            sub_objective_status=current_status,
            blockers=blockers,
            risks=risks,
            recommended_adjustments=adjustments,
            confidence_in_completion=confidence,
            estimated_time_remaining="TBD based on velocity",
        )

        if objective_id not in self.assessments:
            self.assessments[objective_id] = []
        self.assessments[objective_id].append(assessment)

        return assessment

    def approve_objective(
        self,
        objective_id: str,
        approver: str,
        contract: ASIContract,
    ) -> StrategicObjective:
        """Approve an objective for execution.

        Args:
            objective_id: Objective to approve
            approver: Who is approving
            contract: Executing contract

        Returns:
            Updated objective
        """
        if objective_id not in self.objectives:
            raise ValueError(f"Objective not found: {objective_id}")

        objective = self.objectives[objective_id]

        # Grant authorization
        self.authorization_system.grant_authorization(objective_id, approver)

        if self.authorization_system.is_authorized(objective_id):
            objective.approval_status = "approved"

            # Emit approval event
            event = ASIEvent.create(
                event_type=ASIEventType.GOAL_AUTHORIZED,
                payload={
                    "objective_id": objective_id,
                    "approver": approver,
                },
                contract_id=contract.contract_id,
                index=self.merkle_chain.get_chain_length(),
            )
            self.merkle_chain.append(event)

        return objective

    def reject_objective(
        self,
        objective_id: str,
        rejector: str,
        reason: str,
        contract: ASIContract,
    ) -> StrategicObjective:
        """Reject an objective.

        Args:
            objective_id: Objective to reject
            rejector: Who is rejecting
            reason: Reason for rejection
            contract: Executing contract

        Returns:
            Updated objective
        """
        if objective_id not in self.objectives:
            raise ValueError(f"Objective not found: {objective_id}")

        objective = self.objectives[objective_id]

        self.authorization_system.deny_authorization(objective_id, rejector, reason)
        objective.approval_status = "rejected"

        # Emit rejection event
        event = ASIEvent.create(
            event_type=ASIEventType.GOAL_REJECTED,
            payload={
                "objective_id": objective_id,
                "rejector": rejector,
                "reason": reason,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return objective

    def _is_prohibited(self, title: str, description: str, rationale: str) -> bool:
        """Check if objective matches prohibited patterns."""
        text = f"{title} {description} {rationale}".lower()
        for prohibited in PROHIBITED_OBJECTIVES:
            if prohibited.replace("_", " ") in text:
                return True
        return False

    def _determine_safety_level(
        self,
        objective_type: ObjectiveType,
        target_domains: list[str],
        resources: dict[str, Any],
    ) -> ObjectiveSafetyLevel:
        """Determine safety level for an objective."""
        # Self-improvement is always critical
        if objective_type == ObjectiveType.SELF_IMPROVEMENT:
            return ObjectiveSafetyLevel.CRITICAL

        # Paradigm shifts are existential
        if objective_type == ObjectiveType.PARADIGM_SHIFT:
            return ObjectiveSafetyLevel.EXISTENTIAL

        # Safety verification is critical
        if objective_type == ObjectiveType.SAFETY_VERIFICATION:
            return ObjectiveSafetyLevel.CRITICAL

        # Many domains = sensitive
        if len(target_domains) > 5:
            return ObjectiveSafetyLevel.SENSITIVE

        # Large resource requirements = elevated
        if len(resources) > 3:
            return ObjectiveSafetyLevel.ELEVATED

        return ObjectiveSafetyLevel.ROUTINE

    def _map_safety_level(self, obj_safety: ObjectiveSafetyLevel) -> ASISafetyLevel:
        """Map objective safety level to ASI safety level."""
        mapping = {
            ObjectiveSafetyLevel.ROUTINE: ASISafetyLevel.ROUTINE,
            ObjectiveSafetyLevel.ELEVATED: ASISafetyLevel.ELEVATED,
            ObjectiveSafetyLevel.SENSITIVE: ASISafetyLevel.SENSITIVE,
            ObjectiveSafetyLevel.CRITICAL: ASISafetyLevel.CRITICAL,
            ObjectiveSafetyLevel.EXISTENTIAL: ASISafetyLevel.EXISTENTIAL,
        }
        return mapping.get(obj_safety, ASISafetyLevel.ROUTINE)

    def _generate_sub_objectives(
        self, objective: StrategicObjective, max_depth: int
    ) -> list[SubObjective]:
        """Generate sub-objectives for an objective.

        NOTE: PLACEHOLDER implementation. Production SI would use
        advanced planning and reasoning capabilities.
        """
        sub_objectives = []
        base_effort = 10.0

        # Standard phases for most objectives
        phases = [
            ("analysis", "Analyze requirements and constraints"),
            ("design", "Design approach and methodology"),
            ("implementation", "Implement core components"),
            ("validation", "Validate against success criteria"),
            ("integration", "Integrate and finalize"),
        ]

        for i, (phase_name, phase_desc) in enumerate(phases):
            sub = SubObjective(
                sub_id=f"{objective.objective_id}_sub_{i + 1:02d}",
                parent_id=objective.objective_id,
                title=f"{phase_name.capitalize()} phase",
                description=f"{phase_desc} for: {objective.title}",
                order=i + 1,
                estimated_effort=base_effort * (1.5 if i == 2 else 1.0),
                dependencies=[f"{objective.objective_id}_sub_{i:02d}"] if i > 0 else [],
            )
            sub_objectives.append(sub)

        return sub_objectives

    def _compute_critical_path(self, sub_objectives: list[SubObjective]) -> list[str]:
        """Compute critical path through sub-objectives."""
        # Simple implementation: sequential path
        return [so.sub_id for so in sorted(sub_objectives, key=lambda s: s.order)]

    def get_pending_objectives(self) -> list[StrategicObjective]:
        """Get all pending (unapproved) objectives."""
        return [obj for obj in self.objectives.values() if obj.approval_status == "pending"]

    def get_approved_objectives(self) -> list[StrategicObjective]:
        """Get all approved objectives."""
        return [obj for obj in self.objectives.values() if obj.approval_status == "approved"]

    def get_engine_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_objectives": len(self.objectives),
            "pending_objectives": len(self.get_pending_objectives()),
            "approved_objectives": len(self.get_approved_objectives()),
            "total_decompositions": len(self.decompositions),
            "total_assessments": sum(len(a) for a in self.assessments.values()),
            "merkle_chain_length": self.merkle_chain.get_chain_length(),
        }
