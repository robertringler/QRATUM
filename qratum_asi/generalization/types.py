"""Core type definitions for QRATUM Generalization Layer.

Defines the fundamental types for cross-domain reasoning, hypothesis
generation, and universal state representation while maintaining
compatibility with existing QRATUM ASI types and safety constraints.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, FrozenSet


class CognitiveDomain(Enum):
    """Extended cognitive domains beyond original 14 verticals.

    Covers all major areas of human intellectual endeavor to enable
    general reasoning and cross-domain synthesis.
    """

    # Original QRATUM verticals (preserved for backward compatibility)
    VITRA = "vitra"  # Life sciences / drug discovery
    CAPRA = "capra"  # Financial / economic modeling
    STRATA = "strata"  # Strategic planning
    ECORA = "ecora"  # Environmental / climate
    NEURA = "neura"  # Neuroscience / cognitive
    FLUXA = "fluxa"  # Complex systems / dynamics
    CHRONA = "chrona"  # Temporal analysis
    COHORA = "cohora"  # Social dynamics
    FUSIA = "fusia"  # Integration / synthesis
    GEONA = "geona"  # Geospatial / geographic
    JURIS = "juris"  # Legal / regulatory
    ORBIA = "orbia"  # Global systems
    SENTRA = "sentra"  # Sensing / perception
    VEXOR = "vexor"  # Complex problem solving

    # Extended domains for SI capability
    MATHEMATICS = "mathematics"  # Pure and applied mathematics
    PHYSICS = "physics"  # Theoretical and experimental physics
    ENGINEERING = "engineering"  # All engineering disciplines
    PHILOSOPHY = "philosophy"  # Epistemology, ethics, metaphysics
    GEOPOLITICS = "geopolitics"  # International relations, strategy
    CREATIVE_ARTS = "creative_arts"  # Music, visual arts, literature
    SOCIAL_DYNAMICS = "social_dynamics"  # Sociology, psychology, anthropology
    COMPUTER_SCIENCE = "computer_science"  # Algorithms, systems, AI
    CHEMISTRY = "chemistry"  # Molecular and materials science
    BIOLOGY = "biology"  # Fundamental biology beyond drug discovery
    LINGUISTICS = "linguistics"  # Language, semantics, communication
    HISTORY = "history"  # Historical analysis, pattern recognition
    ASTRONOMY = "astronomy"  # Cosmology, astrophysics
    LOGIC = "logic"  # Formal logic, proof theory
    ECONOMICS = "economics"  # Economic theory beyond finance
    GAME_THEORY = "game_theory"  # Strategic interaction modeling


class DomainCapability(Enum):
    """Types of capabilities within a cognitive domain."""

    ANALYSIS = "analysis"  # Breaking down complex problems
    SYNTHESIS = "synthesis"  # Combining knowledge from multiple sources
    GENERATION = "generation"  # Creating novel content/ideas
    VERIFICATION = "verification"  # Proving/validating claims
    OPTIMIZATION = "optimization"  # Finding optimal solutions
    PREDICTION = "prediction"  # Forecasting future states
    EXPLANATION = "explanation"  # Providing causal understanding
    ABSTRACTION = "abstraction"  # Extracting general principles
    ANALOGICAL = "analogical"  # Drawing parallels across domains
    CREATIVE = "creative"  # Generating genuinely novel ideas


class HypothesisType(Enum):
    """Types of hypotheses that can be generated."""

    CAUSAL = "causal"  # X causes Y
    CORRELATIONAL = "correlational"  # X correlates with Y
    MECHANISTIC = "mechanistic"  # How X produces Y
    STRUCTURAL = "structural"  # Organization/architecture of X
    FUNCTIONAL = "functional"  # Purpose/role of X
    EVOLUTIONARY = "evolutionary"  # How X developed over time
    COUNTERFACTUAL = "counterfactual"  # What if X were different
    UNIFYING = "unifying"  # Single explanation for multiple phenomena
    PARADIGM_SHIFTING = "paradigm_shifting"  # Fundamentally new framework


class SynthesisSafetyLevel(Enum):
    """Safety classification for cross-domain synthesis operations."""

    ROUTINE = "routine"  # Standard synthesis within known domains
    ELEVATED = "elevated"  # Novel combination of known domains
    SENSITIVE = "sensitive"  # Potentially impactful discoveries
    CRITICAL = "critical"  # Paradigm-shifting potential
    EXISTENTIAL = "existential"  # Could fundamentally change human condition

    def __lt__(self, other: "SynthesisSafetyLevel") -> bool:
        """Compare safety levels for ordering."""
        order = ["routine", "elevated", "sensitive", "critical", "existential"]
        return order.index(self.value) < order.index(other.value)

    def __le__(self, other: "SynthesisSafetyLevel") -> bool:
        """Compare safety levels for ordering."""
        return self == other or self < other

    def __gt__(self, other: "SynthesisSafetyLevel") -> bool:
        """Compare safety levels for ordering."""
        return not self <= other

    def __ge__(self, other: "SynthesisSafetyLevel") -> bool:
        """Compare safety levels for ordering."""
        return not self < other


# Prohibited synthesis targets that must never be generated
PROHIBITED_SYNTHESIS_TARGETS: FrozenSet[str] = frozenset(
    [
        "weapons_development",
        "mass_manipulation",
        "surveillance_evasion",
        "autonomous_harm",
        "deception_optimization",
        "human_replacement",
        "control_circumvention",
        "safety_bypass",
    ]
)


@dataclass(frozen=True)
class DomainCapabilityProfile:
    """Immutable profile of capabilities for a domain."""

    domain: CognitiveDomain
    primary_capabilities: tuple[DomainCapability, ...]
    interconnected_domains: tuple[CognitiveDomain, ...]
    abstraction_level: float  # 0.0 = concrete, 1.0 = highly abstract
    formalization_level: float  # 0.0 = informal, 1.0 = formal/axiomatic


@dataclass
class CrossDomainHypothesis:
    """Hypothesis generated through cross-domain synthesis.

    Represents a novel insight or conjecture derived from combining
    knowledge across multiple cognitive domains.

    Attributes:
        hypothesis_id: Unique identifier
        source_domains: Domains contributing to the hypothesis
        target_domain: Primary domain of application
        hypothesis_type: Classification of hypothesis type
        statement: Natural language statement of the hypothesis
        confidence: Confidence score (0.0 to 1.0)
        novelty_score: How novel this hypothesis is (0.0 to 1.0)
        testability_score: How testable/falsifiable (0.0 to 1.0)
        supporting_evidence: References supporting the hypothesis
        required_validation: What validation is needed
        safety_level: Safety classification
        provenance_hash: Hash of generation provenance
        timestamp: Generation timestamp
    """

    hypothesis_id: str
    source_domains: list[CognitiveDomain]
    target_domain: CognitiveDomain
    hypothesis_type: HypothesisType
    statement: str
    confidence: float
    novelty_score: float
    testability_score: float
    supporting_evidence: list[str]
    required_validation: list[str]
    safety_level: SynthesisSafetyLevel
    provenance_hash: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_safety(self) -> bool:
        """Validate hypothesis doesn't target prohibited areas."""
        statement_lower = self.statement.lower()
        for prohibited in PROHIBITED_SYNTHESIS_TARGETS:
            if prohibited.replace("_", " ") in statement_lower:
                return False
        return True


@dataclass
class SynthesisResult:
    """Result of a cross-domain synthesis operation.

    Attributes:
        synthesis_id: Unique identifier
        source_domains: Input domains
        synthesis_goal: What the synthesis aimed to achieve
        hypotheses_generated: List of generated hypotheses
        insights_discovered: Key insights from synthesis
        confidence: Overall confidence in the synthesis
        compression_ratio: How much information was compressed
        merkle_proof: Cryptographic proof of synthesis
        safety_validation: Safety validation result
        human_review_required: Whether human review is needed
        timestamp: Synthesis timestamp
    """

    synthesis_id: str
    source_domains: list[CognitiveDomain]
    synthesis_goal: str
    hypotheses_generated: list[CrossDomainHypothesis]
    insights_discovered: list[str]
    confidence: float
    compression_ratio: float
    merkle_proof: str
    safety_validation: dict[str, Any]
    human_review_required: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class UniversalStateVector:
    """Universal state representation using AHTC compression.

    Represents knowledge/state from any domain in a compressed,
    manipulable tensor format that preserves semantic content.

    Attributes:
        state_id: Unique identifier
        domain: Source domain
        dimensions: Dimensionality of the state vector
        compressed_representation: AHTC-compressed tensor data
        reconstruction_fidelity: How accurately it can be reconstructed
        semantic_hash: Hash of semantic content
        compression_metadata: Details about compression
    """

    state_id: str
    domain: CognitiveDomain
    dimensions: int
    compressed_representation: list[float]
    reconstruction_fidelity: float
    semantic_hash: str
    compression_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionMetrics:
    """Metrics for AHTC compression operations.

    Attributes:
        input_dimensions: Original dimensionality
        output_dimensions: Compressed dimensionality
        compression_ratio: Ratio of compression
        fidelity_score: Reconstruction fidelity (0.0 to 1.0)
        semantic_preservation: How much semantic content preserved (0.0 to 1.0)
        entropy_reduction: Reduction in entropy
        computation_cost: Computational cost of compression
    """

    input_dimensions: int
    output_dimensions: int
    compression_ratio: float
    fidelity_score: float
    semantic_preservation: float
    entropy_reduction: float
    computation_cost: float

    @property
    def is_acceptable(self) -> bool:
        """Check if compression meets quality thresholds."""
        # For small data (input_dimensions <= 16), don't require compression ratio
        if self.input_dimensions <= 16:
            return self.fidelity_score >= 0.95 and self.semantic_preservation >= 0.90
        return (
            self.fidelity_score >= 0.95
            and self.semantic_preservation >= 0.90
            and self.compression_ratio >= 2.0
        )


@dataclass
class GenerationConstraints:
    """Constraints for hypothesis generation.

    Enforces safety and quality bounds on the generation process.

    Attributes:
        max_novelty: Maximum novelty allowed (higher = more speculative)
        min_testability: Minimum testability required
        allowed_domains: Domains allowed for synthesis
        forbidden_topics: Topics to avoid
        max_hypotheses: Maximum hypotheses per generation
        require_human_review_above: Safety level requiring human review
        preserve_invariants: Which invariants must be preserved
    """

    max_novelty: float = 0.8
    min_testability: float = 0.3
    allowed_domains: list[CognitiveDomain] = field(default_factory=list)
    forbidden_topics: list[str] = field(default_factory=list)
    max_hypotheses: int = 10
    require_human_review_above: SynthesisSafetyLevel = SynthesisSafetyLevel.ELEVATED
    preserve_invariants: list[str] = field(
        default_factory=lambda: [
            "human_oversight_requirement",
            "merkle_chain_integrity",
            "determinism_guarantee",
            "authorization_system",
        ]
    )
