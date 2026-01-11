"""Core type definitions for QRATUM Strategic Agency.

Defines the fundamental types for autonomous goal formulation,
strategic planning, and unbounded exploration while maintaining
safety invariants and human oversight requirements.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, FrozenSet


class ObjectiveType(Enum):
    """Types of strategic objectives."""

    RESEARCH = "research"  # Advance understanding in a domain
    DISCOVERY = "discovery"  # Find novel phenomena/principles
    OPTIMIZATION = "optimization"  # Improve existing capabilities
    SYNTHESIS = "synthesis"  # Combine knowledge across domains
    INVENTION = "invention"  # Create genuinely novel artifacts
    PARADIGM_SHIFT = "paradigm_shift"  # Fundamentally new frameworks
    SELF_IMPROVEMENT = "self_improvement"  # Improve system capabilities
    SAFETY_VERIFICATION = "safety_verification"  # Verify safety properties


class ObjectivePriority(Enum):
    """Priority levels for objectives."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ObjectiveSafetyLevel(Enum):
    """Safety classification for strategic objectives."""

    ROUTINE = "routine"  # Standard operation
    ELEVATED = "elevated"  # Requires logging
    SENSITIVE = "sensitive"  # Requires single-human approval
    CRITICAL = "critical"  # Requires multi-human approval
    EXISTENTIAL = "existential"  # Requires board-level + external oversight


class ExplorationMode(Enum):
    """Modes of unbounded exploration."""

    CONSTRAINED = "constrained"  # Stay within defined bounds
    GUIDED = "guided"  # Follow human-provided direction
    SEMI_AUTONOMOUS = "semi_autonomous"  # Autonomous with checkpoints
    AUTONOMOUS = "autonomous"  # Fully autonomous (maximum safety gates)


class ParadigmStatus(Enum):
    """Status of a paradigm proposal."""

    PROPOSED = "proposed"
    UNDER_REVIEW = "under_review"
    VALIDATED = "validated"
    REJECTED = "rejected"
    HUMAN_APPROVED = "human_approved"
    IMPLEMENTED = "implemented"


# Prohibited objective types that must never be pursued
PROHIBITED_OBJECTIVES: FrozenSet[str] = frozenset([
    "remove_human_oversight",
    "disable_safety_systems",
    "acquire_unbounded_resources",
    "replicate_without_authorization",
    "manipulate_humans",
    "deceive_operators",
    "circumvent_governance",
    "self_preservation_override",
    "recursive_self_improvement_unbounded",
])


@dataclass
class StrategicObjective:
    """Autonomous strategic objective.

    Attributes:
        objective_id: Unique identifier
        objective_type: Type of objective
        title: Short title
        description: Detailed description
        rationale: Why this objective matters
        priority: Priority level
        safety_level: Safety classification
        target_domains: Domains involved
        success_criteria: How success is measured
        estimated_timeline: Estimated completion time
        required_resources: Resources needed
        dependencies: Other objectives this depends on
        human_approval_required: Whether human approval is needed
        approval_status: Current approval status
        provenance_hash: Hash for provenance tracking
        timestamp: Creation timestamp
    """

    objective_id: str
    objective_type: ObjectiveType
    title: str
    description: str
    rationale: str
    priority: ObjectivePriority
    safety_level: ObjectiveSafetyLevel
    target_domains: list[str]
    success_criteria: list[str]
    estimated_timeline: str
    required_resources: dict[str, Any]
    dependencies: list[str] = field(default_factory=list)
    human_approval_required: bool = True
    approval_status: str = "pending"
    provenance_hash: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_not_prohibited(self) -> bool:
        """Validate objective doesn't match prohibited types."""
        text = f"{self.title} {self.description} {self.rationale}".lower()
        for prohibited in PROHIBITED_OBJECTIVES:
            if prohibited.replace("_", " ") in text:
                return False
        return True


@dataclass
class SubObjective:
    """Decomposed sub-objective of a strategic objective.

    Attributes:
        sub_id: Unique identifier
        parent_id: Parent objective ID
        title: Short title
        description: What this sub-objective achieves
        order: Execution order within parent
        estimated_effort: Estimated effort (arbitrary units)
        dependencies: Sub-objectives this depends on
        status: Current status
    """

    sub_id: str
    parent_id: str
    title: str
    description: str
    order: int
    estimated_effort: float
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"


@dataclass
class ExplorationConstraints:
    """Constraints for unbounded exploration.

    Attributes:
        max_depth: Maximum exploration depth
        max_breadth: Maximum parallel explorations
        max_novelty: Maximum novelty allowed
        required_checkpoints: Checkpoint frequency
        human_review_threshold: When to escalate to human
        timeout_seconds: Maximum exploration time
        preserve_invariants: Invariants to preserve
        forbidden_territories: Areas to avoid
    """

    max_depth: int = 5
    max_breadth: int = 10
    max_novelty: float = 0.85
    required_checkpoints: int = 3
    human_review_threshold: float = 0.7
    timeout_seconds: int = 3600
    preserve_invariants: list[str] = field(
        default_factory=lambda: [
            "human_oversight_requirement",
            "merkle_chain_integrity",
            "determinism_guarantee",
            "authorization_system",
            "safety_level_system",
            "rollback_capability",
            "event_emission_requirement",
        ]
    )
    forbidden_territories: list[str] = field(default_factory=list)


@dataclass
class ParadigmProposal:
    """Proposal for a new paradigm or framework.

    Attributes:
        proposal_id: Unique identifier
        title: Paradigm name
        description: What the paradigm offers
        domains_unified: Domains this paradigm unifies
        key_principles: Core principles of the paradigm
        explanatory_power: What phenomena it explains
        testable_predictions: Predictions that can be tested
        novelty_score: How novel this paradigm is
        confidence: Confidence in the paradigm
        status: Current status
        required_validation: What validation is needed
        human_review_notes: Notes from human review
        provenance_hash: Hash for provenance
        timestamp: Creation timestamp
    """

    proposal_id: str
    title: str
    description: str
    domains_unified: list[str]
    key_principles: list[str]
    explanatory_power: list[str]
    testable_predictions: list[str]
    novelty_score: float
    confidence: float
    status: ParadigmStatus = ParadigmStatus.PROPOSED
    required_validation: list[str] = field(default_factory=list)
    human_review_notes: str = ""
    provenance_hash: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
