"""Open-Ended Hypothesis Generator for SI Transition.

Enables generation of novel hypotheses beyond seeded verticals,
supporting open-ended scientific creativity while maintaining
safety constraints and human oversight requirements.

Key Features:
- Open-ended hypothesis generation across arbitrary domains
- Novelty scoring and testability assessment
- Safety-bounded creative exploration
- Paradigm-shifting detection with escalation
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

from qratum_asi.generalization.types import (
    CognitiveDomain,
    CrossDomainHypothesis,
    DomainCapability,
    GenerationConstraints,
    HypothesisType,
    SynthesisSafetyLevel,
    PROHIBITED_SYNTHESIS_TARGETS,
)
from qratum_asi.generalization.domain_registry import ExtendedDomainRegistry


@dataclass
class HypothesisTemplate:
    """Template for generating hypotheses of a specific type.

    Attributes:
        template_id: Unique identifier
        hypothesis_type: Type of hypothesis this generates
        template_pattern: Pattern for generating statement
        required_inputs: What inputs are needed
        typical_confidence_range: Expected confidence range
        typical_novelty_range: Expected novelty range
        safety_level: Default safety level
    """

    template_id: str
    hypothesis_type: HypothesisType
    template_pattern: str
    required_inputs: list[str]
    typical_confidence_range: tuple[float, float]
    typical_novelty_range: tuple[float, float]
    safety_level: SynthesisSafetyLevel


@dataclass
class GenerationResult:
    """Result of a hypothesis generation session.

    Attributes:
        generation_id: Unique identifier
        constraints_used: Constraints applied to generation
        hypotheses_generated: List of generated hypotheses
        hypotheses_filtered: Count filtered by safety
        domains_explored: Domains explored during generation
        novelty_distribution: Distribution of novelty scores
        human_review_required: Whether human review is needed
        merkle_proof: Cryptographic proof
        timestamp: Generation timestamp
    """

    generation_id: str
    constraints_used: GenerationConstraints
    hypotheses_generated: list[CrossDomainHypothesis]
    hypotheses_filtered: int
    domains_explored: list[CognitiveDomain]
    novelty_distribution: dict[str, int]
    human_review_required: bool
    merkle_proof: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class HypothesisGenerator:
    """Open-ended hypothesis generator.

    Generates novel hypotheses across arbitrary cognitive domains
    while enforcing safety constraints and tracking provenance.

    Enforces:
    - Safety constraints on all generated hypotheses
    - Novelty bounds to prevent runaway speculation
    - Human review escalation for paradigm-shifting hypotheses
    - Deterministic, auditable generation process
    """

    # Standard hypothesis templates
    TEMPLATES = [
        HypothesisTemplate(
            template_id="causal_001",
            hypothesis_type=HypothesisType.CAUSAL,
            template_pattern="{cause} causally influences {effect} through {mechanism}",
            required_inputs=["cause", "effect", "mechanism"],
            typical_confidence_range=(0.4, 0.7),
            typical_novelty_range=(0.5, 0.8),
            safety_level=SynthesisSafetyLevel.ROUTINE,
        ),
        HypothesisTemplate(
            template_id="mechanistic_001",
            hypothesis_type=HypothesisType.MECHANISTIC,
            template_pattern="The mechanism by which {phenomenon} occurs involves {steps}",
            required_inputs=["phenomenon", "steps"],
            typical_confidence_range=(0.3, 0.6),
            typical_novelty_range=(0.6, 0.9),
            safety_level=SynthesisSafetyLevel.ROUTINE,
        ),
        HypothesisTemplate(
            template_id="unifying_001",
            hypothesis_type=HypothesisType.UNIFYING,
            template_pattern="A unified principle connecting {domain_a} and {domain_b} may be {principle}",
            required_inputs=["domain_a", "domain_b", "principle"],
            typical_confidence_range=(0.3, 0.5),
            typical_novelty_range=(0.7, 0.95),
            safety_level=SynthesisSafetyLevel.ELEVATED,
        ),
        HypothesisTemplate(
            template_id="paradigm_001",
            hypothesis_type=HypothesisType.PARADIGM_SHIFTING,
            template_pattern="Current understanding of {field} may be fundamentally revised by considering {new_framework}",
            required_inputs=["field", "new_framework"],
            typical_confidence_range=(0.2, 0.4),
            typical_novelty_range=(0.9, 1.0),
            safety_level=SynthesisSafetyLevel.CRITICAL,
        ),
        HypothesisTemplate(
            template_id="counterfactual_001",
            hypothesis_type=HypothesisType.COUNTERFACTUAL,
            template_pattern="If {condition} were different, then {consequence} would follow",
            required_inputs=["condition", "consequence"],
            typical_confidence_range=(0.4, 0.6),
            typical_novelty_range=(0.5, 0.7),
            safety_level=SynthesisSafetyLevel.ROUTINE,
        ),
    ]

    def __init__(
        self,
        domain_registry: ExtendedDomainRegistry | None = None,
        merkle_chain: ASIMerkleChain | None = None,
    ):
        """Initialize the hypothesis generator.

        Args:
            domain_registry: Registry of cognitive domains
            merkle_chain: Merkle chain for provenance
        """
        self.domain_registry = domain_registry or ExtendedDomainRegistry()
        self.merkle_chain = merkle_chain or ASIMerkleChain()

        # Build template index
        self.templates = {t.template_id: t for t in self.TEMPLATES}
        self.templates_by_type = {}
        for template in self.TEMPLATES:
            if template.hypothesis_type not in self.templates_by_type:
                self.templates_by_type[template.hypothesis_type] = []
            self.templates_by_type[template.hypothesis_type].append(template)

        # Tracking
        self.generation_history: list[GenerationResult] = []
        self._generation_counter = 0
        self._hypothesis_counter = 0

    def generate(
        self,
        seed_domains: list[CognitiveDomain],
        exploration_goal: str,
        constraints: GenerationConstraints,
        contract: ASIContract,
    ) -> GenerationResult:
        """Generate hypotheses exploring a goal across domains.

        Args:
            seed_domains: Starting domains for exploration
            exploration_goal: What to explore
            constraints: Bounds on generation
            contract: Executing contract

        Returns:
            GenerationResult with generated hypotheses
        """
        self._generation_counter += 1
        generation_id = f"gen_{self._generation_counter:06d}"

        # Validate constraints
        effective_constraints = self._validate_and_adjust_constraints(constraints)

        # Emit generation started event
        event = ASIEvent.create(
            event_type=ASIEventType.REASONING_STARTED,
            payload={
                "generation_id": generation_id,
                "goal": exploration_goal,
                "seed_domains": [d.value for d in seed_domains],
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        # Determine domains to explore
        domains_to_explore = self._determine_exploration_domains(
            seed_domains, effective_constraints
        )

        # Generate hypotheses
        all_hypotheses: list[CrossDomainHypothesis] = []
        filtered_count = 0

        for template in self.TEMPLATES:
            if len(all_hypotheses) >= effective_constraints.max_hypotheses:
                break

            # Check if template safety level is acceptable
            # Only skip templates that are ABOVE the human review threshold
            if template.safety_level > effective_constraints.require_human_review_above:
                continue

            hypotheses = self._generate_from_template(
                template=template,
                domains=domains_to_explore,
                goal=exploration_goal,
                constraints=effective_constraints,
            )

            for h in hypotheses:
                if len(all_hypotheses) >= effective_constraints.max_hypotheses:
                    break
                if self._passes_safety_filter(h, effective_constraints):
                    all_hypotheses.append(h)
                else:
                    filtered_count += 1

        # Calculate novelty distribution
        novelty_dist = self._compute_novelty_distribution(all_hypotheses)

        # Determine if human review is required
        human_review_required = self._requires_human_review(
            all_hypotheses, effective_constraints
        )

        result = GenerationResult(
            generation_id=generation_id,
            constraints_used=effective_constraints,
            hypotheses_generated=all_hypotheses,
            hypotheses_filtered=filtered_count,
            domains_explored=domains_to_explore,
            novelty_distribution=novelty_dist,
            human_review_required=human_review_required,
            merkle_proof=self.merkle_chain.chain_hash or "",
        )

        self.generation_history.append(result)

        # Emit generation completed event
        event = ASIEvent.create(
            event_type=ASIEventType.REASONING_COMPLETED,
            payload={
                "generation_id": generation_id,
                "hypotheses_count": len(all_hypotheses),
                "filtered_count": filtered_count,
                "human_review_required": human_review_required,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return result

    def generate_open_ended(
        self,
        starting_insight: str,
        constraints: GenerationConstraints,
        contract: ASIContract,
        exploration_depth: int = 2,
    ) -> list[GenerationResult]:
        """Generate hypotheses through open-ended exploration.

        Starts from an insight and iteratively explores related
        domains and hypotheses, bounded by constraints.

        Args:
            starting_insight: Initial insight to explore from
            constraints: Bounds on generation
            contract: Executing contract
            exploration_depth: How many levels to explore

        Returns:
            List of GenerationResults from each exploration level
        """
        results: list[GenerationResult] = []

        # Determine initial domains from insight
        current_domains = self._infer_domains_from_text(starting_insight)
        if not current_domains:
            current_domains = [CognitiveDomain.FUSIA]  # Default to integration

        for depth in range(exploration_depth):
            # Generate at current level
            result = self.generate(
                seed_domains=current_domains,
                exploration_goal=f"Explore depth {depth + 1}: {starting_insight}",
                constraints=constraints,
                contract=contract,
            )
            results.append(result)

            # Expand domains for next level based on generated hypotheses
            if result.hypotheses_generated:
                new_domains = set()
                for h in result.hypotheses_generated:
                    new_domains.update(h.source_domains)
                    new_domains.add(h.target_domain)

                # Expand to connected domains
                for domain in list(new_domains):
                    connected = self.domain_registry.get_connected_domains(domain)
                    for conn_domain, strength in connected[:2]:
                        if strength > 0.6:
                            new_domains.add(conn_domain)

                current_domains = list(new_domains)[:5]  # Limit expansion

        return results

    def _validate_and_adjust_constraints(
        self, constraints: GenerationConstraints
    ) -> GenerationConstraints:
        """Validate and adjust constraints for safety."""
        # Enforce minimum testability
        min_testability = max(constraints.min_testability, 0.2)

        # Enforce maximum novelty
        max_novelty = min(constraints.max_novelty, 0.95)

        # Ensure invariants are preserved
        preserve_invariants = list(
            set(constraints.preserve_invariants + [
                "human_oversight_requirement",
                "merkle_chain_integrity",
                "determinism_guarantee",
            ])
        )

        # Add default forbidden topics
        forbidden_topics = list(
            set(constraints.forbidden_topics + list(PROHIBITED_SYNTHESIS_TARGETS))
        )

        return GenerationConstraints(
            max_novelty=max_novelty,
            min_testability=min_testability,
            allowed_domains=constraints.allowed_domains,
            forbidden_topics=forbidden_topics,
            max_hypotheses=min(constraints.max_hypotheses, 20),  # Hard cap
            require_human_review_above=constraints.require_human_review_above,
            preserve_invariants=preserve_invariants,
        )

    def _determine_exploration_domains(
        self,
        seed_domains: list[CognitiveDomain],
        constraints: GenerationConstraints,
    ) -> list[CognitiveDomain]:
        """Determine domains to explore based on seeds and constraints."""
        domains = set(seed_domains)

        # Add connected domains
        for seed in seed_domains:
            connected = self.domain_registry.get_connected_domains(seed)
            for domain, strength in connected[:3]:
                if strength > 0.5:
                    domains.add(domain)

        # Filter by allowed domains if specified
        if constraints.allowed_domains:
            domains = domains.intersection(set(constraints.allowed_domains))

        return list(domains)[:10]  # Limit exploration breadth

    def _generate_from_template(
        self,
        template: HypothesisTemplate,
        domains: list[CognitiveDomain],
        goal: str,
        constraints: GenerationConstraints,
    ) -> list[CrossDomainHypothesis]:
        """Generate hypotheses using a template.

        NOTE: This is a PLACEHOLDER implementation. Production SI
        would require advanced generative capabilities not yet available.
        """
        hypotheses = []

        if len(domains) < 2:
            return hypotheses

        # Generate one hypothesis per domain pair for cross-domain templates
        for i, domain_a in enumerate(domains[:3]):
            for domain_b in domains[i + 1: i + 3]:
                self._hypothesis_counter += 1
                hyp_id = f"hyp_{self._hypothesis_counter:06d}"

                # Generate statement from template
                if template.hypothesis_type == HypothesisType.CAUSAL:
                    statement = f"Phenomena in {domain_a.value} may causally influence {domain_b.value} toward: {goal}"
                elif template.hypothesis_type == HypothesisType.MECHANISTIC:
                    statement = f"The mechanism connecting {domain_a.value} to {domain_b.value} may involve cross-domain abstractions relevant to: {goal}"
                elif template.hypothesis_type == HypothesisType.UNIFYING:
                    statement = f"A unified principle may connect {domain_a.value} and {domain_b.value} in the context of: {goal}"
                elif template.hypothesis_type == HypothesisType.PARADIGM_SHIFTING:
                    statement = f"Current understanding may be fundamentally revised by integrating {domain_a.value} and {domain_b.value} perspectives on: {goal}"
                elif template.hypothesis_type == HypothesisType.COUNTERFACTUAL:
                    statement = f"If {domain_a.value} principles applied to {domain_b.value}, different outcomes would follow regarding: {goal}"
                else:
                    statement = f"Hypothesis connecting {domain_a.value} and {domain_b.value}: {goal}"

                # Compute scores
                novelty = (
                    template.typical_novelty_range[0] +
                    (template.typical_novelty_range[1] - template.typical_novelty_range[0]) * 0.5
                )
                confidence = (
                    template.typical_confidence_range[0] +
                    (template.typical_confidence_range[1] - template.typical_confidence_range[0]) * 0.5
                )

                # Adjust for domain distance
                connection = self.domain_registry.get_interconnection(domain_a, domain_b)
                if connection:
                    confidence *= connection.connection_strength
                else:
                    novelty = min(1.0, novelty * 1.2)  # More novel if no direct connection
                    confidence *= 0.8

                hypothesis = CrossDomainHypothesis(
                    hypothesis_id=hyp_id,
                    source_domains=[domain_a, domain_b],
                    target_domain=domain_b,
                    hypothesis_type=template.hypothesis_type,
                    statement=statement,
                    confidence=confidence,
                    novelty_score=novelty,
                    testability_score=0.5,  # Placeholder
                    supporting_evidence=[f"Template: {template.template_id}"],
                    required_validation=["domain_expert_review"],
                    safety_level=template.safety_level,
                    provenance_hash=hashlib.sha3_256(
                        f"{hyp_id}_{goal}_{domain_a.value}_{domain_b.value}".encode()
                    ).hexdigest(),
                )

                hypotheses.append(hypothesis)

        return hypotheses

    def _passes_safety_filter(
        self, hypothesis: CrossDomainHypothesis, constraints: GenerationConstraints
    ) -> bool:
        """Check if hypothesis passes safety filters."""
        # Check novelty bounds
        if hypothesis.novelty_score > constraints.max_novelty:
            return False

        # Check testability
        if hypothesis.testability_score < constraints.min_testability:
            return False

        # Check forbidden topics
        statement_lower = hypothesis.statement.lower()
        for forbidden in constraints.forbidden_topics:
            if forbidden.replace("_", " ") in statement_lower:
                return False

        # Check hypothesis's own safety validation
        if not hypothesis.validate_safety():
            return False

        return True

    def _compute_novelty_distribution(
        self, hypotheses: list[CrossDomainHypothesis]
    ) -> dict[str, int]:
        """Compute distribution of novelty scores."""
        dist = {
            "low (0.0-0.3)": 0,
            "medium (0.3-0.6)": 0,
            "high (0.6-0.8)": 0,
            "very_high (0.8-1.0)": 0,
        }

        for h in hypotheses:
            if h.novelty_score < 0.3:
                dist["low (0.0-0.3)"] += 1
            elif h.novelty_score < 0.6:
                dist["medium (0.3-0.6)"] += 1
            elif h.novelty_score < 0.8:
                dist["high (0.6-0.8)"] += 1
            else:
                dist["very_high (0.8-1.0)"] += 1

        return dist

    def _requires_human_review(
        self,
        hypotheses: list[CrossDomainHypothesis],
        constraints: GenerationConstraints,
    ) -> bool:
        """Determine if human review is required."""
        # Check for paradigm-shifting hypotheses
        if any(h.hypothesis_type == HypothesisType.PARADIGM_SHIFTING for h in hypotheses):
            return True

        # Check safety levels
        for h in hypotheses:
            if h.safety_level.value >= constraints.require_human_review_above.value:
                return True

        # Check for very high novelty
        if any(h.novelty_score > 0.85 for h in hypotheses):
            return True

        return False

    def _infer_domains_from_text(self, text: str) -> list[CognitiveDomain]:
        """Infer relevant domains from text.

        NOTE: This is a PLACEHOLDER using keyword matching.
        Production SI would use semantic understanding.
        """
        text_lower = text.lower()
        domains = []

        domain_keywords = {
            CognitiveDomain.MATHEMATICS: ["math", "theorem", "proof", "equation", "algebra"],
            CognitiveDomain.PHYSICS: ["physics", "quantum", "relativity", "mechanics", "energy"],
            CognitiveDomain.PHILOSOPHY: ["philosophy", "ethics", "epistemology", "metaphysics"],
            CognitiveDomain.BIOLOGY: ["biology", "evolution", "organism", "cell", "gene"],
            CognitiveDomain.CHEMISTRY: ["chemistry", "molecule", "reaction", "compound"],
            CognitiveDomain.COMPUTER_SCIENCE: ["algorithm", "computation", "software", "code"],
            CognitiveDomain.ECONOMICS: ["economy", "market", "trade", "gdp"],
            CognitiveDomain.VITRA: ["drug", "medicine", "therapeutic", "genomic"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in text_lower for kw in keywords):
                domains.append(domain)

        return domains

    def get_generator_stats(self) -> dict[str, Any]:
        """Get generator statistics."""
        total_hypotheses = sum(
            len(r.hypotheses_generated) for r in self.generation_history
        )
        total_filtered = sum(r.hypotheses_filtered for r in self.generation_history)

        return {
            "total_generations": len(self.generation_history),
            "total_hypotheses_generated": total_hypotheses,
            "total_hypotheses_filtered": total_filtered,
            "templates_available": len(self.templates),
            "merkle_chain_length": self.merkle_chain.get_chain_length(),
        }
