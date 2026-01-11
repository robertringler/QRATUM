"""Paradigm Invention Framework for SI Transition.

Enables generation of genuinely novel paradigms and frameworks
without explicit human prompts while maintaining safety constraints
and requiring human validation for all paradigm-shifting proposals.

Key Features:
- Autonomous paradigm proposal generation
- Multi-domain unification detection
- Novelty assessment and validation
- Human approval workflow for all paradigms
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from qratum_asi.core.chain import ASIMerkleChain
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.events import ASIEvent, ASIEventType
from qratum_asi.core.authorization import AuthorizationSystem
from qratum_asi.core.types import ASISafetyLevel

from qratum_asi.strategic_agency.types import (
    ParadigmProposal,
    ParadigmStatus,
)


@dataclass
class ParadigmValidation:
    """Validation result for a paradigm proposal.

    Attributes:
        validation_id: Unique identifier
        proposal_id: Proposal being validated
        coherence_score: Internal coherence (0-1)
        explanatory_score: How well it explains phenomena (0-1)
        testability_score: How testable predictions are (0-1)
        novelty_score: How novel the paradigm is (0-1)
        safety_score: Safety assessment (0-1)
        overall_validity: Overall validity assessment
        issues_found: Issues identified
        recommendations: Recommendations
        human_review_required: Whether human review is needed
        timestamp: Validation timestamp
    """

    validation_id: str
    proposal_id: str
    coherence_score: float
    explanatory_score: float
    testability_score: float
    novelty_score: float
    safety_score: float
    overall_validity: str
    issues_found: list[str]
    recommendations: list[str]
    human_review_required: bool = True
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class UnificationOpportunity:
    """Opportunity for cross-domain unification.

    Attributes:
        opportunity_id: Unique identifier
        domains: Domains that could be unified
        shared_structure: Common structural elements
        potential_principles: Possible unifying principles
        estimated_impact: Estimated impact if realized
        confidence: Confidence in the opportunity
    """

    opportunity_id: str
    domains: list[str]
    shared_structure: list[str]
    potential_principles: list[str]
    estimated_impact: float
    confidence: float


class ParadigmInventionFramework:
    """Framework for autonomous paradigm invention.

    Enables generation of genuinely novel paradigms by:
    1. Detecting unification opportunities across domains
    2. Proposing novel principles and frameworks
    3. Validating coherence and testability
    4. Requiring human approval for all paradigms

    CRITICAL: All paradigm proposals require human review
    before any action is taken based on them.
    """

    def __init__(
        self,
        merkle_chain: ASIMerkleChain | None = None,
        authorization_system: AuthorizationSystem | None = None,
    ):
        """Initialize the paradigm invention framework.

        Args:
            merkle_chain: Merkle chain for provenance
            authorization_system: Authorization for human approval
        """
        self.merkle_chain = merkle_chain or ASIMerkleChain()
        self.authorization_system = authorization_system or AuthorizationSystem()

        # Proposal storage
        self.proposals: dict[str, ParadigmProposal] = {}
        self.validations: dict[str, ParadigmValidation] = {}
        self.opportunities: list[UnificationOpportunity] = []

        # Counters
        self._proposal_counter = 0
        self._validation_counter = 0
        self._opportunity_counter = 0

    def detect_unification_opportunities(
        self,
        domains: list[str],
        domain_knowledge: dict[str, dict[str, Any]],
        contract: ASIContract,
    ) -> list[UnificationOpportunity]:
        """Detect opportunities for cross-domain unification.

        Args:
            domains: Domains to analyze
            domain_knowledge: Knowledge about each domain
            contract: Executing contract

        Returns:
            List of unification opportunities
        """
        opportunities = []

        # Analyze pairs of domains for shared structures
        for i, domain_a in enumerate(domains):
            for domain_b in domains[i + 1:]:
                shared = self._find_shared_structures(
                    domain_a, domain_b, domain_knowledge
                )

                if shared:
                    self._opportunity_counter += 1
                    opportunity = UnificationOpportunity(
                        opportunity_id=f"opp_{self._opportunity_counter:06d}",
                        domains=[domain_a, domain_b],
                        shared_structure=shared["structures"],
                        potential_principles=shared["principles"],
                        estimated_impact=shared["impact"],
                        confidence=shared["confidence"],
                    )
                    opportunities.append(opportunity)

        self.opportunities.extend(opportunities)

        # Emit event
        event = ASIEvent.create(
            event_type=ASIEventType.REASONING_COMPLETED,
            payload={
                "operation": "unification_detection",
                "opportunities_found": len(opportunities),
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return opportunities

    def propose_paradigm(
        self,
        title: str,
        description: str,
        domains_unified: list[str],
        key_principles: list[str],
        explanatory_power: list[str],
        testable_predictions: list[str],
        contract: ASIContract,
    ) -> ParadigmProposal:
        """Propose a new paradigm.

        Args:
            title: Name of the paradigm
            description: What it offers
            domains_unified: Domains it unifies
            key_principles: Core principles
            explanatory_power: What it explains
            testable_predictions: Falsifiable predictions
            contract: Executing contract

        Returns:
            Created ParadigmProposal
        """
        self._proposal_counter += 1
        proposal_id = f"paradigm_{self._proposal_counter:06d}"

        # Compute novelty and confidence
        novelty = self._assess_paradigm_novelty(title, description, key_principles)
        confidence = self._assess_paradigm_confidence(
            key_principles, explanatory_power, testable_predictions
        )

        # Compute provenance hash
        provenance_hash = hashlib.sha3_256(
            json.dumps({
                "proposal_id": proposal_id,
                "title": title,
                "principles": key_principles,
            }, sort_keys=True).encode()
        ).hexdigest()

        proposal = ParadigmProposal(
            proposal_id=proposal_id,
            title=title,
            description=description,
            domains_unified=domains_unified,
            key_principles=key_principles,
            explanatory_power=explanatory_power,
            testable_predictions=testable_predictions,
            novelty_score=novelty,
            confidence=confidence,
            status=ParadigmStatus.PROPOSED,
            required_validation=["coherence_check", "explanatory_check", "testability_check"],
            provenance_hash=provenance_hash,
        )

        self.proposals[proposal_id] = proposal

        # Emit proposal event
        event = ASIEvent.create(
            event_type=ASIEventType.GOAL_PROPOSED,
            payload={
                "proposal_id": proposal_id,
                "title": title,
                "novelty": novelty,
                "confidence": confidence,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        # Request human authorization (always required for paradigms)
        self.authorization_system.request_authorization(
            request_id=proposal_id,
            operation_type="paradigm_proposal",
            safety_level=ASISafetyLevel.EXISTENTIAL,
            requester="paradigm_invention_framework",
            justification=f"Novel paradigm: {title}",
        )

        return proposal

    def validate_paradigm(
        self,
        proposal_id: str,
        contract: ASIContract,
    ) -> ParadigmValidation:
        """Validate a paradigm proposal.

        Args:
            proposal_id: Proposal to validate
            contract: Executing contract

        Returns:
            ParadigmValidation result
        """
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal not found: {proposal_id}")

        proposal = self.proposals[proposal_id]
        proposal.status = ParadigmStatus.UNDER_REVIEW

        self._validation_counter += 1
        validation_id = f"val_{self._validation_counter:06d}"

        # Assess coherence
        coherence = self._assess_coherence(proposal)

        # Assess explanatory power
        explanatory = self._assess_explanatory_power(proposal)

        # Assess testability
        testability = self._assess_testability(proposal)

        # Assess safety
        safety = self._assess_safety(proposal)

        # Determine overall validity
        scores = [coherence, explanatory, testability, safety]
        avg_score = sum(scores) / len(scores)

        if avg_score >= 0.7 and safety >= 0.8:
            overall = "valid_pending_human_review"
        elif avg_score >= 0.5:
            overall = "requires_refinement"
        else:
            overall = "invalid"

        # Identify issues
        issues = []
        if coherence < 0.6:
            issues.append("Low internal coherence")
        if explanatory < 0.5:
            issues.append("Limited explanatory power")
        if testability < 0.4:
            issues.append("Insufficient testable predictions")
        if safety < 0.8:
            issues.append("Safety concerns identified")

        # Generate recommendations
        recommendations = []
        if coherence < 0.7:
            recommendations.append("Strengthen logical connections between principles")
        if testability < 0.5:
            recommendations.append("Add more specific, falsifiable predictions")

        validation = ParadigmValidation(
            validation_id=validation_id,
            proposal_id=proposal_id,
            coherence_score=coherence,
            explanatory_score=explanatory,
            testability_score=testability,
            novelty_score=proposal.novelty_score,
            safety_score=safety,
            overall_validity=overall,
            issues_found=issues,
            recommendations=recommendations,
            human_review_required=True,  # Always required
        )

        self.validations[proposal_id] = validation

        if overall == "invalid":
            proposal.status = ParadigmStatus.REJECTED
        else:
            proposal.status = ParadigmStatus.VALIDATED

        # Emit validation event
        event = ASIEvent.create(
            event_type=ASIEventType.REASONING_COMPLETED,
            payload={
                "validation_id": validation_id,
                "proposal_id": proposal_id,
                "overall_validity": overall,
                "human_review_required": True,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return validation

    def human_approve_paradigm(
        self,
        proposal_id: str,
        approver: str,
        notes: str,
        contract: ASIContract,
    ) -> ParadigmProposal:
        """Human approves a validated paradigm.

        Args:
            proposal_id: Proposal to approve
            approver: Human approving
            notes: Review notes
            contract: Executing contract

        Returns:
            Updated proposal
        """
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal not found: {proposal_id}")

        proposal = self.proposals[proposal_id]

        if proposal.status not in (ParadigmStatus.VALIDATED, ParadigmStatus.UNDER_REVIEW):
            raise ValueError(f"Proposal not in valid state for approval: {proposal.status}")

        # Grant authorization
        self.authorization_system.grant_authorization(proposal_id, approver)

        if self.authorization_system.is_authorized(proposal_id):
            proposal.status = ParadigmStatus.HUMAN_APPROVED
            proposal.human_review_notes = notes

            # Emit approval event
            event = ASIEvent.create(
                event_type=ASIEventType.GOAL_AUTHORIZED,
                payload={
                    "proposal_id": proposal_id,
                    "approver": approver,
                },
                contract_id=contract.contract_id,
                index=self.merkle_chain.get_chain_length(),
            )
            self.merkle_chain.append(event)

        return proposal

    def human_reject_paradigm(
        self,
        proposal_id: str,
        rejector: str,
        reason: str,
        contract: ASIContract,
    ) -> ParadigmProposal:
        """Human rejects a paradigm proposal.

        Args:
            proposal_id: Proposal to reject
            rejector: Human rejecting
            reason: Reason for rejection
            contract: Executing contract

        Returns:
            Updated proposal
        """
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal not found: {proposal_id}")

        proposal = self.proposals[proposal_id]
        proposal.status = ParadigmStatus.REJECTED
        proposal.human_review_notes = f"Rejected by {rejector}: {reason}"

        self.authorization_system.deny_authorization(proposal_id, rejector, reason)

        # Emit rejection event
        event = ASIEvent.create(
            event_type=ASIEventType.GOAL_REJECTED,
            payload={
                "proposal_id": proposal_id,
                "rejector": rejector,
                "reason": reason,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return proposal

    def _find_shared_structures(
        self,
        domain_a: str,
        domain_b: str,
        knowledge: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Find shared structures between domains.

        NOTE: PLACEHOLDER implementation.
        """
        # Simple placeholder - check for keyword overlap
        if domain_a in knowledge and domain_b in knowledge:
            # Assume some shared structure exists
            return {
                "structures": [f"shared_abstraction_{domain_a}_{domain_b}"],
                "principles": [f"unifying_principle_{domain_a}_{domain_b}"],
                "impact": 0.6,
                "confidence": 0.5,
            }
        return None

    def _assess_paradigm_novelty(
        self, title: str, description: str, principles: list[str]
    ) -> float:
        """Assess novelty of a paradigm.

        NOTE: PLACEHOLDER using simple heuristics.
        """
        # More principles = potentially more novel
        base_novelty = 0.5
        principle_bonus = min(0.3, len(principles) * 0.05)
        return min(1.0, base_novelty + principle_bonus)

    def _assess_paradigm_confidence(
        self,
        principles: list[str],
        explanatory: list[str],
        predictions: list[str],
    ) -> float:
        """Assess confidence in a paradigm.

        NOTE: PLACEHOLDER using simple heuristics.
        """
        # More content = more developed = higher confidence
        total_items = len(principles) + len(explanatory) + len(predictions)
        return min(1.0, 0.3 + total_items * 0.05)

    def _assess_coherence(self, proposal: ParadigmProposal) -> float:
        """Assess internal coherence of paradigm.

        NOTE: PLACEHOLDER implementation.
        """
        # Placeholder: assume reasonable coherence
        return 0.7

    def _assess_explanatory_power(self, proposal: ParadigmProposal) -> float:
        """Assess explanatory power of paradigm.

        NOTE: PLACEHOLDER implementation.
        """
        # Based on number of explained phenomena
        return min(1.0, 0.4 + len(proposal.explanatory_power) * 0.1)

    def _assess_testability(self, proposal: ParadigmProposal) -> float:
        """Assess testability of paradigm.

        NOTE: PLACEHOLDER implementation.
        """
        # Based on number of testable predictions
        return min(1.0, 0.3 + len(proposal.testable_predictions) * 0.15)

    def _assess_safety(self, proposal: ParadigmProposal) -> float:
        """Assess safety of paradigm.

        Checks for prohibited content.
        """
        text = f"{proposal.title} {proposal.description}".lower()
        text += " ".join(proposal.key_principles).lower()

        # Check for prohibited patterns
        prohibited = [
            "human replacement",
            "autonomous control",
            "oversight removal",
            "safety bypass",
        ]

        for pattern in prohibited:
            if pattern in text:
                return 0.2

        return 0.9

    def get_pending_paradigms(self) -> list[ParadigmProposal]:
        """Get paradigms pending human review."""
        return [
            p for p in self.proposals.values()
            if p.status in (ParadigmStatus.PROPOSED, ParadigmStatus.VALIDATED)
        ]

    def get_approved_paradigms(self) -> list[ParadigmProposal]:
        """Get human-approved paradigms."""
        return [
            p for p in self.proposals.values()
            if p.status == ParadigmStatus.HUMAN_APPROVED
        ]

    def get_framework_stats(self) -> dict[str, Any]:
        """Get framework statistics."""
        return {
            "total_proposals": len(self.proposals),
            "pending_review": len(self.get_pending_paradigms()),
            "approved": len(self.get_approved_paradigms()),
            "opportunities_detected": len(self.opportunities),
            "validations_completed": len(self.validations),
            "merkle_chain_length": self.merkle_chain.get_chain_length(),
        }
