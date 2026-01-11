"""General Reasoning Engine for SI Transition.

Provides cross-domain knowledge synthesis and general reasoning capabilities
that extend beyond QRATUM's original 14 verticals while preserving all
safety invariants, determinism, and human oversight requirements.

Key Features:
- Multi-strategy reasoning across arbitrary domains
- Cross-domain synthesis with semantic translation
- AHTC-compressed knowledge representation
- Deterministic, auditable reasoning chains
- Safety-bounded hypothesis generation
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from qratum_asi.core.chain import ASIMerkleChain
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.events import ASIEvent, ASIEventType
from qratum_asi.core.types import ASISafetyLevel, ReasoningStrategy

from qratum_asi.generalization.types import (
    CognitiveDomain,
    CrossDomainHypothesis,
    DomainCapability,
    HypothesisType,
    SynthesisResult,
    SynthesisSafetyLevel,
    PROHIBITED_SYNTHESIS_TARGETS,
)
from qratum_asi.generalization.domain_registry import (
    ExtendedDomainRegistry,
    DomainInterconnection,
)


class ReasoningMode(Enum):
    """Modes of general reasoning."""

    SINGLE_DOMAIN = "single_domain"  # Reasoning within one domain
    CROSS_DOMAIN = "cross_domain"  # Synthesis across domains
    ANALOGICAL = "analogical"  # Reasoning by analogy
    ABDUCTIVE = "abductive"  # Best explanation inference
    GENERATIVE = "generative"  # Novel hypothesis generation
    VERIFICATION = "verification"  # Proof/validation mode


@dataclass
class ReasoningContext:
    """Context for a reasoning operation.

    Attributes:
        context_id: Unique identifier
        primary_domain: Main domain for reasoning
        supporting_domains: Additional domains to draw from
        goal: What the reasoning aims to achieve
        constraints: Bounds on the reasoning
        prior_knowledge: Relevant prior knowledge
        safety_level: Safety classification
    """

    context_id: str
    primary_domain: CognitiveDomain
    supporting_domains: list[CognitiveDomain]
    goal: str
    constraints: dict[str, Any]
    prior_knowledge: list[str]
    safety_level: SynthesisSafetyLevel = SynthesisSafetyLevel.ROUTINE

    def compute_hash(self) -> str:
        """Compute deterministic hash of context."""
        content = {
            "context_id": self.context_id,
            "primary_domain": self.primary_domain.value,
            "supporting_domains": [d.value for d in self.supporting_domains],
            "goal": self.goal,
            "constraints": self.constraints,
        }
        return hashlib.sha3_256(json.dumps(content, sort_keys=True).encode()).hexdigest()


@dataclass
class ReasoningStep:
    """Single step in a reasoning chain.

    Attributes:
        step_id: Unique identifier
        domain: Domain of this step
        strategy: Reasoning strategy used
        input_premises: Input to this step
        output_conclusion: Output from this step
        confidence: Confidence in this step (0-1)
        justification: Why this conclusion follows
        provenance_hash: Hash of step provenance
    """

    step_id: str
    domain: CognitiveDomain
    strategy: ReasoningStrategy
    input_premises: list[str]
    output_conclusion: str
    confidence: float
    justification: str
    provenance_hash: str


@dataclass
class ReasoningChain:
    """Complete reasoning chain from premises to conclusion.

    Attributes:
        chain_id: Unique identifier
        context: Reasoning context
        mode: Reasoning mode used
        steps: Sequence of reasoning steps
        final_conclusion: Final derived conclusion
        overall_confidence: Aggregate confidence
        domains_traversed: Domains visited during reasoning
        merkle_proof: Cryptographic proof of chain
        timestamp: Chain creation time
    """

    chain_id: str
    context: ReasoningContext
    mode: ReasoningMode
    steps: list[ReasoningStep]
    final_conclusion: str
    overall_confidence: float
    domains_traversed: list[CognitiveDomain]
    merkle_proof: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def get_step_count(self) -> int:
        """Get number of reasoning steps."""
        return len(self.steps)

    def get_lowest_confidence_step(self) -> ReasoningStep | None:
        """Get the step with lowest confidence."""
        if not self.steps:
            return None
        return min(self.steps, key=lambda s: s.confidence)


class CrossDomainSynthesizer:
    """Synthesizes knowledge across multiple cognitive domains.

    Implements semantic translation and knowledge integration
    between domains with different formalizations and abstractions.

    Enforces:
    - Safety constraints on synthesis combinations
    - Human review for sensitive syntheses
    - Provenance tracking for all operations
    """

    def __init__(
        self,
        domain_registry: ExtendedDomainRegistry,
        merkle_chain: ASIMerkleChain,
    ):
        """Initialize synthesizer.

        Args:
            domain_registry: Registry of domains and interconnections
            merkle_chain: Merkle chain for provenance
        """
        self.domain_registry = domain_registry
        self.merkle_chain = merkle_chain
        self._synthesis_counter = 0

    def synthesize(
        self,
        source_domains: list[CognitiveDomain],
        synthesis_goal: str,
        context: dict[str, Any],
        contract: ASIContract,
        max_hypotheses: int = 5,
    ) -> SynthesisResult:
        """Perform cross-domain synthesis.

        Combines knowledge from multiple domains to generate novel
        insights and hypotheses toward a specified goal.

        Args:
            source_domains: Domains to synthesize from
            synthesis_goal: What the synthesis aims to achieve
            context: Additional context for synthesis
            contract: Executing contract
            max_hypotheses: Maximum hypotheses to generate

        Returns:
            SynthesisResult with generated hypotheses and insights

        Raises:
            ValueError: If synthesis violates safety constraints
        """
        # Validate safety
        if not self._validate_synthesis_safety(synthesis_goal, source_domains):
            raise ValueError("Synthesis goal violates safety constraints")

        self._synthesis_counter += 1
        synthesis_id = f"synthesis_{self._synthesis_counter:06d}"

        # Emit synthesis started event
        event = ASIEvent.create(
            event_type=ASIEventType.REASONING_STARTED,
            payload={
                "synthesis_id": synthesis_id,
                "goal": synthesis_goal,
                "domains": [d.value for d in source_domains],
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        # Find synthesis paths between domains
        synthesis_paths = self._find_synthesis_paths(source_domains)

        # Identify shared abstractions
        shared_abstractions = self._find_shared_abstractions(source_domains)

        # Generate hypotheses
        hypotheses = self._generate_hypotheses(
            source_domains=source_domains,
            synthesis_goal=synthesis_goal,
            paths=synthesis_paths,
            shared_abstractions=shared_abstractions,
            context=context,
            max_hypotheses=max_hypotheses,
        )

        # Extract insights
        insights = self._extract_insights(source_domains, synthesis_goal, hypotheses)

        # Determine if human review is required
        safety_level = self._determine_synthesis_safety_level(
            source_domains, hypotheses
        )
        human_review_required = safety_level in (
            SynthesisSafetyLevel.SENSITIVE,
            SynthesisSafetyLevel.CRITICAL,
            SynthesisSafetyLevel.EXISTENTIAL,
        )

        # Safety validation
        safety_validation = {
            "prohibited_targets_checked": True,
            "all_hypotheses_safe": all(h.validate_safety() for h in hypotheses),
            "domains_compatible": True,
            "safety_level": safety_level.value,
        }

        result = SynthesisResult(
            synthesis_id=synthesis_id,
            source_domains=source_domains,
            synthesis_goal=synthesis_goal,
            hypotheses_generated=hypotheses,
            insights_discovered=insights,
            confidence=self._compute_synthesis_confidence(hypotheses),
            compression_ratio=len(source_domains) / max(len(hypotheses), 1),
            merkle_proof=self.merkle_chain.chain_hash or "",
            safety_validation=safety_validation,
            human_review_required=human_review_required,
        )

        # Emit synthesis completed event
        event = ASIEvent.create(
            event_type=ASIEventType.REASONING_COMPLETED,
            payload={
                "synthesis_id": synthesis_id,
                "hypotheses_count": len(hypotheses),
                "confidence": result.confidence,
                "human_review_required": human_review_required,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return result

    def _validate_synthesis_safety(
        self, goal: str, domains: list[CognitiveDomain]
    ) -> bool:
        """Validate synthesis doesn't target prohibited areas."""
        goal_lower = goal.lower()
        for prohibited in PROHIBITED_SYNTHESIS_TARGETS:
            if prohibited.replace("_", " ") in goal_lower:
                return False
        return True

    def _find_synthesis_paths(
        self, domains: list[CognitiveDomain]
    ) -> list[list[CognitiveDomain]]:
        """Find synthesis paths between domains."""
        paths = []
        for i, source in enumerate(domains):
            for target in domains[i + 1 :]:
                domain_paths = self.domain_registry.find_synthesis_path(
                    source, target, max_hops=3
                )
                if domain_paths:
                    paths.extend(domain_paths)
        return paths

    def _find_shared_abstractions(
        self, domains: list[CognitiveDomain]
    ) -> list[str]:
        """Find abstractions shared across domains."""
        # Get capabilities for each domain
        capabilities_by_domain = {}
        for domain in domains:
            definition = self.domain_registry.get_domain(domain)
            if definition:
                capabilities_by_domain[domain] = set(definition.capabilities)

        # Find common capabilities
        if not capabilities_by_domain:
            return []

        common_capabilities = set.intersection(*capabilities_by_domain.values())

        # Map to abstraction names
        abstractions = []
        for cap in common_capabilities:
            abstractions.append(f"shared_{cap.value}_abstraction")

        return abstractions

    def _generate_hypotheses(
        self,
        source_domains: list[CognitiveDomain],
        synthesis_goal: str,
        paths: list[list[CognitiveDomain]],
        shared_abstractions: list[str],
        context: dict[str, Any],
        max_hypotheses: int,
    ) -> list[CrossDomainHypothesis]:
        """Generate cross-domain hypotheses.

        NOTE: This is a PLACEHOLDER implementation. A production SI system
        would require advanced neural-symbolic reasoning, theorem proving,
        and knowledge graph integration capabilities not yet available.
        """
        hypotheses = []
        hypothesis_counter = 0

        # Generate structural hypotheses from domain connections
        for path in paths[:max_hypotheses]:
            if len(path) >= 2:
                hypothesis_counter += 1
                source = path[0]
                target = path[-1]

                hypothesis = CrossDomainHypothesis(
                    hypothesis_id=f"hyp_{hypothesis_counter:04d}",
                    source_domains=[source, target],
                    target_domain=target,
                    hypothesis_type=HypothesisType.STRUCTURAL,
                    statement=f"Structural patterns in {source.value} may inform understanding of {target.value} toward goal: {synthesis_goal}",
                    confidence=0.6,
                    novelty_score=0.7,
                    testability_score=0.5,
                    supporting_evidence=[f"Connected via path: {' -> '.join(d.value for d in path)}"],
                    required_validation=["domain_expert_review", "empirical_test"],
                    safety_level=SynthesisSafetyLevel.ROUTINE,
                    provenance_hash=hashlib.sha3_256(
                        f"{synthesis_goal}_{source.value}_{target.value}".encode()
                    ).hexdigest(),
                )

                if hypothesis.validate_safety():
                    hypotheses.append(hypothesis)

        # Generate analogical hypotheses from shared abstractions
        if shared_abstractions and len(hypotheses) < max_hypotheses:
            for abstraction in shared_abstractions[:max_hypotheses - len(hypotheses)]:
                hypothesis_counter += 1

                hypothesis = CrossDomainHypothesis(
                    hypothesis_id=f"hyp_{hypothesis_counter:04d}",
                    source_domains=source_domains,
                    target_domain=source_domains[0],
                    hypothesis_type=HypothesisType.UNIFYING,
                    statement=f"The {abstraction} principle may unify understanding across {', '.join(d.value for d in source_domains)} toward: {synthesis_goal}",
                    confidence=0.5,
                    novelty_score=0.8,
                    testability_score=0.4,
                    supporting_evidence=[f"Shared capability: {abstraction}"],
                    required_validation=["formal_proof", "domain_expert_review"],
                    safety_level=SynthesisSafetyLevel.ELEVATED,
                    provenance_hash=hashlib.sha3_256(
                        f"{synthesis_goal}_{abstraction}".encode()
                    ).hexdigest(),
                )

                if hypothesis.validate_safety():
                    hypotheses.append(hypothesis)

        return hypotheses

    def _extract_insights(
        self,
        domains: list[CognitiveDomain],
        goal: str,
        hypotheses: list[CrossDomainHypothesis],
    ) -> list[str]:
        """Extract key insights from synthesis."""
        insights = []

        # Domain coverage insight
        insights.append(
            f"Synthesis covered {len(domains)} domains with {len(hypotheses)} hypotheses"
        )

        # Highest confidence insight
        if hypotheses:
            best = max(hypotheses, key=lambda h: h.confidence)
            insights.append(
                f"Highest confidence hypothesis ({best.confidence:.2f}): {best.statement[:100]}..."
            )

        # Novel combinations insight
        novel_count = sum(1 for h in hypotheses if h.novelty_score > 0.7)
        if novel_count > 0:
            insights.append(f"Generated {novel_count} highly novel hypotheses")

        return insights

    def _determine_synthesis_safety_level(
        self, domains: list[CognitiveDomain], hypotheses: list[CrossDomainHypothesis]
    ) -> SynthesisSafetyLevel:
        """Determine overall safety level of synthesis."""
        # Check for sensitive domain combinations
        sensitive_domains = {
            CognitiveDomain.GEOPOLITICS,
            CognitiveDomain.SOCIAL_DYNAMICS,
        }

        if any(d in sensitive_domains for d in domains):
            return SynthesisSafetyLevel.SENSITIVE

        # Check hypothesis safety levels
        if any(h.safety_level == SynthesisSafetyLevel.CRITICAL for h in hypotheses):
            return SynthesisSafetyLevel.CRITICAL

        if any(h.safety_level == SynthesisSafetyLevel.SENSITIVE for h in hypotheses):
            return SynthesisSafetyLevel.SENSITIVE

        # Check novelty (high novelty = elevated)
        if any(h.novelty_score > 0.8 for h in hypotheses):
            return SynthesisSafetyLevel.ELEVATED

        return SynthesisSafetyLevel.ROUTINE

    def _compute_synthesis_confidence(
        self, hypotheses: list[CrossDomainHypothesis]
    ) -> float:
        """Compute overall synthesis confidence."""
        if not hypotheses:
            return 0.0

        # Average confidence weighted by testability
        total_weight = 0.0
        weighted_sum = 0.0

        for h in hypotheses:
            weight = h.testability_score + 0.1  # Avoid zero weight
            weighted_sum += h.confidence * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0


class GeneralReasoningEngine:
    """General reasoning engine for cross-domain synthesis.

    Extends Q-MIND capabilities to support reasoning across arbitrary
    cognitive domains while preserving all QRATUM invariants:
    - Deterministic reasoning chains
    - Cryptographic provenance
    - Human oversight for sensitive operations
    - Safety boundary enforcement

    CRITICAL DISCLAIMER:
    This is a THEORETICAL ARCHITECTURE. Production superintelligence
    capabilities require fundamental breakthroughs not yet achieved.
    """

    def __init__(
        self,
        domain_registry: ExtendedDomainRegistry | None = None,
        merkle_chain: ASIMerkleChain | None = None,
    ):
        """Initialize the general reasoning engine.

        Args:
            domain_registry: Registry of cognitive domains
            merkle_chain: Merkle chain for provenance tracking
        """
        self.domain_registry = domain_registry or ExtendedDomainRegistry()
        self.merkle_chain = merkle_chain or ASIMerkleChain()
        self.synthesizer = CrossDomainSynthesizer(
            domain_registry=self.domain_registry,
            merkle_chain=self.merkle_chain,
        )

        # Tracking
        self.reasoning_chains: dict[str, ReasoningChain] = {}
        self.synthesis_results: dict[str, SynthesisResult] = {}
        self._chain_counter = 0

    def reason(
        self,
        context: ReasoningContext,
        mode: ReasoningMode,
        contract: ASIContract,
    ) -> ReasoningChain:
        """Perform general reasoning in a context.

        Args:
            context: Reasoning context with goal and constraints
            mode: Mode of reasoning to use
            contract: Executing contract

        Returns:
            ReasoningChain with steps and conclusion
        """
        self._chain_counter += 1
        chain_id = f"chain_{self._chain_counter:06d}"

        # Validate contract
        if not contract.validate():
            raise ValueError(f"Invalid contract: {contract.contract_id}")

        # Emit reasoning started event
        event = ASIEvent.create(
            event_type=ASIEventType.REASONING_STARTED,
            payload={
                "chain_id": chain_id,
                "mode": mode.value,
                "goal": context.goal,
                "primary_domain": context.primary_domain.value,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        # Generate reasoning steps based on mode
        steps = self._generate_reasoning_steps(context, mode)

        # Compute final conclusion
        final_conclusion = self._derive_conclusion(steps, context)

        # Compute overall confidence
        overall_confidence = self._compute_chain_confidence(steps)

        # Collect domains traversed
        domains_traversed = list(
            set([context.primary_domain] + context.supporting_domains)
        )

        chain = ReasoningChain(
            chain_id=chain_id,
            context=context,
            mode=mode,
            steps=steps,
            final_conclusion=final_conclusion,
            overall_confidence=overall_confidence,
            domains_traversed=domains_traversed,
            merkle_proof=self.merkle_chain.chain_hash or "",
        )

        self.reasoning_chains[chain_id] = chain

        # Emit reasoning completed event
        event = ASIEvent.create(
            event_type=ASIEventType.REASONING_COMPLETED,
            payload={
                "chain_id": chain_id,
                "steps": len(steps),
                "confidence": overall_confidence,
                "conclusion": final_conclusion[:200],
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return chain

    def cross_domain_synthesis(
        self,
        source_domains: list[CognitiveDomain],
        synthesis_goal: str,
        context: dict[str, Any],
        contract: ASIContract,
        max_hypotheses: int = 5,
    ) -> SynthesisResult:
        """Perform cross-domain knowledge synthesis.

        Combines knowledge from multiple cognitive domains to generate
        novel hypotheses and insights toward a specified goal.

        Args:
            source_domains: Domains to synthesize from
            synthesis_goal: Goal of the synthesis
            context: Additional context
            contract: Executing contract
            max_hypotheses: Maximum hypotheses to generate

        Returns:
            SynthesisResult with hypotheses and insights
        """
        result = self.synthesizer.synthesize(
            source_domains=source_domains,
            synthesis_goal=synthesis_goal,
            context=context,
            contract=contract,
            max_hypotheses=max_hypotheses,
        )

        self.synthesis_results[result.synthesis_id] = result
        return result

    def _generate_reasoning_steps(
        self, context: ReasoningContext, mode: ReasoningMode
    ) -> list[ReasoningStep]:
        """Generate reasoning steps based on mode.

        NOTE: This is a PLACEHOLDER implementation. Production SI
        would require advanced reasoning algorithms not yet available.
        """
        steps = []
        step_counter = 0

        if mode == ReasoningMode.SINGLE_DOMAIN:
            # Single domain reasoning
            step_counter += 1
            step = ReasoningStep(
                step_id=f"step_{step_counter:03d}",
                domain=context.primary_domain,
                strategy=ReasoningStrategy.DEDUCTIVE,
                input_premises=context.prior_knowledge[:3] if context.prior_knowledge else ["Domain knowledge"],
                output_conclusion=f"Analysis within {context.primary_domain.value} domain",
                confidence=0.8,
                justification=f"Applied {context.primary_domain.value} domain expertise",
                provenance_hash=context.compute_hash(),
            )
            steps.append(step)

        elif mode == ReasoningMode.CROSS_DOMAIN:
            # Cross-domain reasoning with transfer
            for domain in [context.primary_domain] + context.supporting_domains[:2]:
                step_counter += 1
                step = ReasoningStep(
                    step_id=f"step_{step_counter:03d}",
                    domain=domain,
                    strategy=ReasoningStrategy.ANALOGICAL,
                    input_premises=[f"Knowledge from {domain.value}"],
                    output_conclusion=f"Cross-domain insight from {domain.value}",
                    confidence=0.7,
                    justification=f"Analogical transfer from {domain.value}",
                    provenance_hash=hashlib.sha3_256(
                        f"{context.context_id}_{domain.value}".encode()
                    ).hexdigest(),
                )
                steps.append(step)

        elif mode == ReasoningMode.ANALOGICAL:
            # Analogical reasoning
            step_counter += 1
            step = ReasoningStep(
                step_id=f"step_{step_counter:03d}",
                domain=context.primary_domain,
                strategy=ReasoningStrategy.ANALOGICAL,
                input_premises=["Source analogy", "Target situation"],
                output_conclusion="Analogical inference derived",
                confidence=0.65,
                justification="Structural similarity identified",
                provenance_hash=context.compute_hash(),
            )
            steps.append(step)

        elif mode == ReasoningMode.ABDUCTIVE:
            # Abductive reasoning (inference to best explanation)
            step_counter += 1
            step = ReasoningStep(
                step_id=f"step_{step_counter:03d}",
                domain=context.primary_domain,
                strategy=ReasoningStrategy.ABDUCTIVE,
                input_premises=["Observed phenomenon"],
                output_conclusion="Best explanation hypothesis",
                confidence=0.6,
                justification="Most parsimonious explanation",
                provenance_hash=context.compute_hash(),
            )
            steps.append(step)

        elif mode == ReasoningMode.GENERATIVE:
            # Generative hypothesis mode
            step_counter += 1
            step = ReasoningStep(
                step_id=f"step_{step_counter:03d}",
                domain=context.primary_domain,
                strategy=ReasoningStrategy.INDUCTIVE,
                input_premises=context.prior_knowledge[:3] if context.prior_knowledge else ["Observations"],
                output_conclusion="Novel hypothesis generated",
                confidence=0.55,
                justification="Creative extrapolation from patterns",
                provenance_hash=context.compute_hash(),
            )
            steps.append(step)

        elif mode == ReasoningMode.VERIFICATION:
            # Verification mode
            step_counter += 1
            step = ReasoningStep(
                step_id=f"step_{step_counter:03d}",
                domain=context.primary_domain,
                strategy=ReasoningStrategy.DEDUCTIVE,
                input_premises=["Claim to verify", "Supporting evidence"],
                output_conclusion="Verification result",
                confidence=0.85,
                justification="Formal verification applied",
                provenance_hash=context.compute_hash(),
            )
            steps.append(step)

        return steps

    def _derive_conclusion(
        self, steps: list[ReasoningStep], context: ReasoningContext
    ) -> str:
        """Derive final conclusion from reasoning steps."""
        if not steps:
            return "No conclusion - no reasoning steps generated"

        # Combine step conclusions
        step_conclusions = [s.output_conclusion for s in steps]

        return f"Toward goal '{context.goal}': {'; '.join(step_conclusions)}"

    def _compute_chain_confidence(self, steps: list[ReasoningStep]) -> float:
        """Compute overall chain confidence (conservative - minimum)."""
        if not steps:
            return 0.0
        return min(step.confidence for step in steps)

    def get_reasoning_chain(self, chain_id: str) -> ReasoningChain | None:
        """Get a reasoning chain by ID."""
        return self.reasoning_chains.get(chain_id)

    def get_synthesis_result(self, synthesis_id: str) -> SynthesisResult | None:
        """Get a synthesis result by ID."""
        return self.synthesis_results.get(synthesis_id)

    def get_supported_domains(self) -> list[CognitiveDomain]:
        """Get all supported cognitive domains."""
        return self.domain_registry.get_all_domains()

    def get_engine_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_reasoning_chains": len(self.reasoning_chains),
            "total_syntheses": len(self.synthesis_results),
            "domains_supported": len(self.domain_registry.get_all_domains()),
            "core_verticals": len(self.domain_registry.get_core_verticals()),
            "extended_domains": len(self.domain_registry.get_extended_domains()),
            "merkle_chain_length": self.merkle_chain.get_chain_length(),
            "merkle_chain_valid": self.merkle_chain.verify_integrity(),
        }

    def verify_provenance(self) -> bool:
        """Verify reasoning engine provenance integrity."""
        return self.merkle_chain.verify_integrity()
