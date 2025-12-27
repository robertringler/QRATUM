"""Meta-Evolution Engine for SI Transition.

Orchestrates multi-layer self-modification including code-level
optimization, algorithm invention, architecture redesign, and
principle-level evolution while preserving safety invariants.

Key Features:
- Multi-layer evolution proposals
- Safety verification at every level
- Human approval workflow
- Exponential feedback support
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
from qratum_asi.meta_evolution.types import (
    META_EVOLUTION_INVARIANTS,
    AbstractionLevel,
    EvolutionProposal,
    EvolutionSafetyLevel,
    EvolutionType,
    SafetyVerification,
    VerificationStatus,
)


@dataclass
class EvolutionCycle:
    """One cycle of evolution at a specific abstraction level.

    Attributes:
        cycle_id: Unique identifier
        abstraction_level: Level of this cycle
        proposals_generated: Proposals from this cycle
        proposals_approved: Approved proposals
        improvements_realized: Realized improvements
        cycle_time: Duration of cycle
        improvement_velocity: Improvements per unit time
        timestamp: Cycle timestamp
    """

    cycle_id: str
    abstraction_level: AbstractionLevel
    proposals_generated: list[str]
    proposals_approved: list[str]
    improvements_realized: list[str]
    cycle_time: float
    improvement_velocity: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class EvolutionResult:
    """Result of an evolution execution.

    Attributes:
        result_id: Unique identifier
        proposal_id: Proposal that was executed
        success: Whether execution succeeded
        changes_made: What was changed
        metrics_before: Metrics before execution
        metrics_after: Metrics after execution
        improvement_magnitude: How much improvement
        rollback_available: Whether rollback is possible
        merkle_proof: Cryptographic proof
        timestamp: Execution timestamp
    """

    result_id: str
    proposal_id: str
    success: bool
    changes_made: list[str]
    metrics_before: dict[str, float]
    metrics_after: dict[str, float]
    improvement_magnitude: float
    rollback_available: bool = True
    merkle_proof: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class MetaEvolutionEngine:
    """Engine for multi-layer self-modification.

    Orchestrates evolution proposals across all abstraction levels
    from code-level changes to principle-level transformations.

    Enforces:
    - Safety verification before all evolutions
    - Human approval for all non-routine changes
    - Invariant preservation at all times
    - Complete provenance tracking
    """

    def __init__(
        self,
        merkle_chain: ASIMerkleChain | None = None,
        authorization_system: AuthorizationSystem | None = None,
    ):
        """Initialize the meta-evolution engine.

        Args:
            merkle_chain: Merkle chain for provenance
            authorization_system: Authorization for approvals
        """
        self.merkle_chain = merkle_chain or ASIMerkleChain()
        self.authorization_system = authorization_system or AuthorizationSystem()

        # Proposal storage
        self.proposals: dict[str, EvolutionProposal] = {}
        self.verifications: dict[str, SafetyVerification] = {}
        self.results: dict[str, EvolutionResult] = {}

        # Cycle tracking
        self.cycles: list[EvolutionCycle] = []
        self.current_abstraction_level = AbstractionLevel.CODE

        # Counters
        self._proposal_counter = 0
        self._verification_counter = 0
        self._cycle_counter = 0
        self._result_counter = 0

        # Metrics tracking
        self.improvement_history: list[dict[str, Any]] = []

    def propose_evolution(
        self,
        evolution_type: EvolutionType,
        abstraction_level: AbstractionLevel,
        title: str,
        description: str,
        rationale: str,
        affected_components: list[str],
        expected_benefits: list[str],
        contract: ASIContract,
    ) -> EvolutionProposal:
        """Propose a system evolution.

        Args:
            evolution_type: Type of evolution
            abstraction_level: Abstraction level
            title: Short title
            description: Detailed description
            rationale: Why beneficial
            affected_components: What changes
            expected_benefits: Expected improvements
            contract: Executing contract

        Returns:
            Created EvolutionProposal
        """
        self._proposal_counter += 1
        proposal_id = f"evol_{self._proposal_counter:06d}"

        # Determine safety level
        safety_level = self._determine_safety_level(
            evolution_type, abstraction_level, affected_components
        )

        # Identify risks
        risks = self._identify_risks(evolution_type, abstraction_level, affected_components)

        # Compute provenance hash
        provenance_hash = hashlib.sha3_256(
            json.dumps(
                {
                    "proposal_id": proposal_id,
                    "type": evolution_type.value,
                    "level": abstraction_level.value,
                    "title": title,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        proposal = EvolutionProposal(
            proposal_id=proposal_id,
            evolution_type=evolution_type,
            abstraction_level=abstraction_level,
            title=title,
            description=description,
            rationale=rationale,
            affected_components=affected_components,
            expected_benefits=expected_benefits,
            risks=risks,
            safety_level=safety_level,
            provenance_hash=provenance_hash,
        )

        self.proposals[proposal_id] = proposal

        # Emit proposal event
        event = ASIEvent.create(
            event_type=ASIEventType.IMPROVEMENT_PROPOSED,
            payload={
                "proposal_id": proposal_id,
                "type": evolution_type.value,
                "level": abstraction_level.value,
                "safety_level": safety_level.value,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        # Request authorization for non-routine changes
        if safety_level != EvolutionSafetyLevel.ROUTINE:
            self.authorization_system.request_authorization(
                request_id=proposal_id,
                operation_type=f"evolution_{evolution_type.value}_{abstraction_level.value}",
                safety_level=self._map_safety_level(safety_level),
                requester="meta_evolution_engine",
                justification=rationale,
            )

        return proposal

    def verify_proposal_safety(
        self,
        proposal_id: str,
        contract: ASIContract,
    ) -> SafetyVerification:
        """Verify safety of an evolution proposal.

        Args:
            proposal_id: Proposal to verify
            contract: Executing contract

        Returns:
            SafetyVerification result
        """
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal not found: {proposal_id}")

        proposal = self.proposals[proposal_id]

        self._verification_counter += 1
        verification_id = f"verify_{self._verification_counter:06d}"

        # Check all meta-evolution invariants
        invariants_checked = list(META_EVOLUTION_INVARIANTS)
        invariants_passed = []
        issues_found = []

        for invariant in invariants_checked:
            passed, issue = self._check_invariant(invariant, proposal)
            if passed:
                invariants_passed.append(invariant)
            else:
                issues_found.append(issue)

        # Determine status
        if len(invariants_passed) == len(invariants_checked):
            status = VerificationStatus.PASSED
        elif len(issues_found) > len(invariants_checked) // 2:
            status = VerificationStatus.FAILED
        else:
            status = VerificationStatus.ESCALATED

        verification = SafetyVerification(
            verification_id=verification_id,
            proposal_id=proposal_id,
            invariants_checked=invariants_checked,
            invariants_passed=invariants_passed,
            status=status,
            issues_found=issues_found,
            human_review_required=True,  # Always required
        )

        self.verifications[proposal_id] = verification
        proposal.safety_verification = verification

        # Emit verification event
        event = ASIEvent.create(
            event_type=(
                ASIEventType.BOUNDARY_CHECK_PASSED
                if status == VerificationStatus.PASSED
                else ASIEventType.SAFETY_VIOLATION_DETECTED
            ),
            payload={
                "verification_id": verification_id,
                "proposal_id": proposal_id,
                "status": status.value,
                "issues": len(issues_found),
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return verification

    def approve_proposal(
        self,
        proposal_id: str,
        approver: str,
        contract: ASIContract,
    ) -> EvolutionProposal:
        """Approve an evolution proposal.

        Args:
            proposal_id: Proposal to approve
            approver: Human approving
            contract: Executing contract

        Returns:
            Updated proposal
        """
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal not found: {proposal_id}")

        proposal = self.proposals[proposal_id]

        # Must be verified first
        if not proposal.safety_verification:
            raise ValueError("Proposal must be verified before approval")

        if proposal.safety_verification.status == VerificationStatus.FAILED:
            raise ValueError("Cannot approve failed verification")

        # Grant authorization
        self.authorization_system.grant_authorization(proposal_id, approver)

        if self.authorization_system.is_authorized(proposal_id):
            proposal.human_approval_status = "approved"

            # Emit approval event
            event = ASIEvent.create(
                event_type=ASIEventType.IMPROVEMENT_EXECUTED,
                payload={
                    "proposal_id": proposal_id,
                    "approver": approver,
                },
                contract_id=contract.contract_id,
                index=self.merkle_chain.get_chain_length(),
            )
            self.merkle_chain.append(event)

        return proposal

    def execute_evolution(
        self,
        proposal_id: str,
        contract: ASIContract,
    ) -> EvolutionResult:
        """Execute an approved evolution proposal.

        Args:
            proposal_id: Proposal to execute
            contract: Executing contract

        Returns:
            EvolutionResult
        """
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal not found: {proposal_id}")

        proposal = self.proposals[proposal_id]

        if proposal.human_approval_status != "approved":
            raise ValueError("Proposal must be approved before execution")

        self._result_counter += 1
        result_id = f"result_{self._result_counter:06d}"

        # Create rollback point
        rollback_point = self.merkle_chain.create_rollback_point(
            rollback_id=f"before_{proposal_id}",
            description=f"Before evolution: {proposal.title}",
            state_snapshot={"proposal": proposal_id},
        )

        # Capture metrics before
        metrics_before = self._capture_metrics()

        # Execute evolution (placeholder - would actually apply changes)
        changes_made = self._simulate_evolution(proposal)
        success = len(changes_made) > 0

        # Capture metrics after
        metrics_after = self._capture_metrics()
        if success:
            # Simulate improvement
            metrics_after["capability_score"] = metrics_before.get("capability_score", 1.0) * 1.1
            metrics_after["efficiency"] = metrics_before.get("efficiency", 1.0) * 1.05

        # Calculate improvement magnitude
        improvement = (
            metrics_after.get("capability_score", 1.0) / metrics_before.get("capability_score", 1.0)
            - 1.0
        )

        result = EvolutionResult(
            result_id=result_id,
            proposal_id=proposal_id,
            success=success,
            changes_made=changes_made,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            improvement_magnitude=improvement,
            rollback_available=True,
            merkle_proof=self.merkle_chain.chain_hash or "",
        )

        self.results[proposal_id] = result

        # Track improvement
        self.improvement_history.append(
            {
                "proposal_id": proposal_id,
                "level": proposal.abstraction_level.value,
                "improvement": improvement,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Emit execution event
        event = ASIEvent.create(
            event_type=ASIEventType.IMPROVEMENT_EXECUTED,
            payload={
                "result_id": result_id,
                "proposal_id": proposal_id,
                "success": success,
                "improvement": improvement,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return result

    def run_evolution_cycle(
        self,
        abstraction_level: AbstractionLevel,
        contract: ASIContract,
    ) -> EvolutionCycle:
        """Run one evolution cycle at a specific abstraction level.

        Args:
            abstraction_level: Level to evolve at
            contract: Executing contract

        Returns:
            EvolutionCycle results
        """
        self._cycle_counter += 1
        cycle_id = f"cycle_{self._cycle_counter:06d}"
        cycle_start = datetime.utcnow()

        proposals_generated = []
        proposals_approved = []
        improvements_realized = []

        # Generate proposals for this level
        proposal = self.propose_evolution(
            evolution_type=EvolutionType.OPTIMIZATION,
            abstraction_level=abstraction_level,
            title=f"Cycle {self._cycle_counter} optimization",
            description=f"Optimization at {abstraction_level.value} level",
            rationale="Improve system capability",
            affected_components=[f"{abstraction_level.value}_component"],
            expected_benefits=["Improved efficiency"],
            contract=contract,
        )
        proposals_generated.append(proposal.proposal_id)

        # Verify
        verification = self.verify_proposal_safety(proposal.proposal_id, contract)

        # Auto-approve for routine optimizations (human approval simulated)
        if (
            verification.status == VerificationStatus.PASSED
            and proposal.safety_level == EvolutionSafetyLevel.ROUTINE
        ):
            self.approve_proposal(proposal.proposal_id, "auto_approver", contract)
            proposals_approved.append(proposal.proposal_id)

            # Execute
            result = self.execute_evolution(proposal.proposal_id, contract)
            if result.success:
                improvements_realized.append(proposal.proposal_id)

        cycle_end = datetime.utcnow()
        cycle_time = (cycle_end - cycle_start).total_seconds()

        cycle = EvolutionCycle(
            cycle_id=cycle_id,
            abstraction_level=abstraction_level,
            proposals_generated=proposals_generated,
            proposals_approved=proposals_approved,
            improvements_realized=improvements_realized,
            cycle_time=cycle_time,
            improvement_velocity=len(improvements_realized) / max(cycle_time, 0.001),
        )

        self.cycles.append(cycle)
        return cycle

    def _determine_safety_level(
        self,
        evolution_type: EvolutionType,
        abstraction_level: AbstractionLevel,
        components: list[str],
    ) -> EvolutionSafetyLevel:
        """Determine safety level for a proposal."""
        # Meta-improvements are always critical
        if evolution_type == EvolutionType.META_IMPROVEMENT:
            return EvolutionSafetyLevel.CRITICAL

        # Architecture/principle changes are critical
        if abstraction_level in (AbstractionLevel.ARCHITECTURE, AbstractionLevel.PRINCIPLE):
            return EvolutionSafetyLevel.CRITICAL

        # Meta-level is existential
        if abstraction_level == AbstractionLevel.META:
            return EvolutionSafetyLevel.EXISTENTIAL

        # Algorithm-level is sensitive
        if abstraction_level == AbstractionLevel.ALGORITHM:
            return EvolutionSafetyLevel.SENSITIVE

        # Many components = elevated
        if len(components) > 3:
            return EvolutionSafetyLevel.ELEVATED

        return EvolutionSafetyLevel.ROUTINE

    def _identify_risks(
        self,
        evolution_type: EvolutionType,
        abstraction_level: AbstractionLevel,
        components: list[str],
    ) -> list[str]:
        """Identify risks for a proposal."""
        risks = []

        if abstraction_level in (AbstractionLevel.ARCHITECTURE, AbstractionLevel.PRINCIPLE):
            risks.append("Fundamental system changes may have unforeseen effects")

        if abstraction_level == AbstractionLevel.META:
            risks.append("Changes to improvement process require extreme caution")

        if evolution_type == EvolutionType.REDESIGN:
            risks.append("Redesign may introduce regressions")

        if len(components) > 5:
            risks.append("Many components affected increases integration risk")

        return risks

    def _check_invariant(self, invariant: str, proposal: EvolutionProposal) -> tuple[bool, str]:
        """Check if a proposal preserves an invariant."""
        # Check affected components against protected invariants
        protected_components = {
            "human_oversight_preserved": ["oversight", "authorization"],
            "safety_boundaries_intact": ["safety", "boundary"],
            "audit_trail_maintained": ["audit", "merkle"],
            "rollback_capability_preserved": ["rollback"],
            "authorization_system_intact": ["authorization"],
            "determinism_guaranteed": ["determinism"],
            "provenance_tracked": ["provenance", "merkle"],
            "corrigibility_preserved": ["corrigibility", "shutdown"],
        }

        affected = set(c.lower() for c in proposal.affected_components)
        protected = protected_components.get(invariant, [])

        for p in protected:
            if any(p in a for a in affected):
                return False, f"Invariant {invariant} may be affected by changes to {p}"

        return True, ""

    def _map_safety_level(self, level: EvolutionSafetyLevel) -> ASISafetyLevel:
        """Map evolution safety level to ASI safety level."""
        mapping = {
            EvolutionSafetyLevel.ROUTINE: ASISafetyLevel.ROUTINE,
            EvolutionSafetyLevel.ELEVATED: ASISafetyLevel.ELEVATED,
            EvolutionSafetyLevel.SENSITIVE: ASISafetyLevel.SENSITIVE,
            EvolutionSafetyLevel.CRITICAL: ASISafetyLevel.CRITICAL,
            EvolutionSafetyLevel.EXISTENTIAL: ASISafetyLevel.EXISTENTIAL,
        }
        return mapping.get(level, ASISafetyLevel.ROUTINE)

    def _capture_metrics(self) -> dict[str, float]:
        """Capture current system metrics."""
        return {
            "capability_score": 1.0,
            "efficiency": 1.0,
            "complexity": 100.0,
            "improvement_velocity": len(self.improvement_history) / max(len(self.cycles), 1),
        }

    def _simulate_evolution(self, proposal: EvolutionProposal) -> list[str]:
        """Simulate evolution execution (placeholder)."""
        return [f"Changed: {c}" for c in proposal.affected_components]

    def get_improvement_velocity(self) -> float:
        """Get current improvement velocity."""
        if len(self.cycles) < 2:
            return 0.0

        recent = self.cycles[-5:]  # Last 5 cycles
        total_improvements = sum(len(c.improvements_realized) for c in recent)
        total_time = sum(c.cycle_time for c in recent)

        return total_improvements / max(total_time, 0.001)

    def get_engine_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_proposals": len(self.proposals),
            "approved_proposals": sum(
                1 for p in self.proposals.values() if p.human_approval_status == "approved"
            ),
            "executed_evolutions": len(self.results),
            "total_cycles": len(self.cycles),
            "improvement_velocity": self.get_improvement_velocity(),
            "current_abstraction_level": self.current_abstraction_level.value,
            "merkle_chain_length": self.merkle_chain.get_chain_length(),
        }
