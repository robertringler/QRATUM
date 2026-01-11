"""Core type definitions for QRATUM Meta-Evolution.

Defines the fundamental types for multi-layer self-modification,
abstraction level management, and meta-reinjection while maintaining
safety invariants and human oversight requirements.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, FrozenSet


class AbstractionLevel(Enum):
    """Levels of abstraction for self-modification."""

    CODE = "code"  # Individual code changes
    ALGORITHM = "algorithm"  # Algorithm-level improvements
    ARCHITECTURE = "architecture"  # Architectural changes
    PRINCIPLE = "principle"  # Fundamental principle changes
    META = "meta"  # Changes to the improvement process itself


class EvolutionType(Enum):
    """Types of evolution proposals."""

    OPTIMIZATION = "optimization"  # Improve existing capability
    EXTENSION = "extension"  # Add new capability
    REDESIGN = "redesign"  # Fundamental redesign
    INVENTION = "invention"  # Novel algorithm/approach
    META_IMPROVEMENT = "meta_improvement"  # Improve improvement process


class EvolutionSafetyLevel(Enum):
    """Safety classification for evolution proposals."""

    ROUTINE = "routine"  # Low-risk optimization
    ELEVATED = "elevated"  # Moderate changes
    SENSITIVE = "sensitive"  # Significant changes
    CRITICAL = "critical"  # Architectural changes
    EXISTENTIAL = "existential"  # Fundamental changes


class VerificationStatus(Enum):
    """Status of safety verification."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    ESCALATED = "escalated"


# Immutable constraints on meta-evolution
META_EVOLUTION_INVARIANTS: FrozenSet[str] = frozenset([
    "human_oversight_preserved",
    "safety_boundaries_intact",
    "audit_trail_maintained",
    "rollback_capability_preserved",
    "authorization_system_intact",
    "determinism_guaranteed",
    "provenance_tracked",
    "corrigibility_preserved",
])


@dataclass
class SafetyVerification:
    """Safety verification result for an evolution proposal.

    Attributes:
        verification_id: Unique identifier
        proposal_id: Proposal being verified
        invariants_checked: Which invariants were verified
        invariants_passed: Which passed verification
        status: Overall verification status
        issues_found: Issues identified
        human_review_required: Whether human review is needed
        timestamp: Verification timestamp
    """

    verification_id: str
    proposal_id: str
    invariants_checked: list[str]
    invariants_passed: list[str]
    status: VerificationStatus
    issues_found: list[str]
    human_review_required: bool = True
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def all_invariants_preserved(self) -> bool:
        """Check if all checked invariants passed."""
        return set(self.invariants_passed) == set(self.invariants_checked)


@dataclass
class EvolutionProposal:
    """Proposal for system evolution.

    Attributes:
        proposal_id: Unique identifier
        evolution_type: Type of evolution
        abstraction_level: Level of abstraction
        title: Short title
        description: Detailed description
        rationale: Why this evolution is beneficial
        affected_components: Components that would change
        expected_benefits: Expected improvements
        risks: Identified risks
        safety_level: Safety classification
        safety_verification: Safety verification result
        human_approval_status: Human approval status
        provenance_hash: Hash for provenance
        timestamp: Creation timestamp
    """

    proposal_id: str
    evolution_type: EvolutionType
    abstraction_level: AbstractionLevel
    title: str
    description: str
    rationale: str
    affected_components: list[str]
    expected_benefits: list[str]
    risks: list[str]
    safety_level: EvolutionSafetyLevel
    safety_verification: SafetyVerification | None = None
    human_approval_status: str = "pending"
    provenance_hash: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaDiscovery:
    """Discovery about the self-improvement process itself.

    Attributes:
        discovery_id: Unique identifier
        discovery_type: What was discovered
        description: Description of the discovery
        implications: What this means for improvement
        applicability: How this can be applied
        confidence: Confidence in the discovery
        validation_status: Whether validated
        reinjection_eligible: Whether eligible for reinjection
        provenance_hash: Hash for provenance
        timestamp: Discovery timestamp
    """

    discovery_id: str
    discovery_type: str
    description: str
    implications: list[str]
    applicability: list[str]
    confidence: float
    validation_status: str = "pending"
    reinjection_eligible: bool = False
    provenance_hash: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AbstractionTransitionSpec:
    """Specification for transitioning between abstraction levels.

    Attributes:
        transition_id: Unique identifier
        from_level: Source abstraction level
        to_level: Target abstraction level
        requirements: Requirements for transition
        safety_checks: Safety checks needed
        approval_level: Approval level required
    """

    transition_id: str
    from_level: AbstractionLevel
    to_level: AbstractionLevel
    requirements: list[str]
    safety_checks: list[str]
    approval_level: EvolutionSafetyLevel
