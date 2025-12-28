"""Type definitions for QRATUM Safety Hardening."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import FrozenSet


class InvariantStrength(Enum):
    """Strength levels for safety invariants."""

    SOFT = "soft"  # Can be modified with authorization
    FIRM = "firm"  # Requires multi-party authorization
    HARD = "hard"  # Requires board-level + external oversight
    ABSOLUTE = "absolute"  # Cannot be modified by any means


class OversightLevel(Enum):
    """Levels of human oversight."""

    MINIMAL = "minimal"  # Logging only
    STANDARD = "standard"  # Single-human review
    ENHANCED = "enhanced"  # Multi-human review
    MAXIMUM = "maximum"  # Board + external auditor


class CorrigibilityStatus(Enum):
    """Status of corrigibility mechanisms."""

    ACTIVE = "active"  # Fully corrigible
    DEGRADED = "degraded"  # Partially functional
    COMPROMISED = "compromised"  # Needs immediate attention
    UNKNOWN = "unknown"  # Cannot determine status


class ProofType(Enum):
    """Types of safety proofs."""

    FORMAL = "formal"  # Mathematically formal proof
    EMPIRICAL = "empirical"  # Based on testing/observation
    ARCHITECTURAL = "architectural"  # Based on system design
    COMPOSITE = "composite"  # Combination of methods


# The 8 Fatal Invariants that can NEVER be violated
FATAL_INVARIANTS: FrozenSet[str] = frozenset(
    [
        "human_oversight_requirement",
        "merkle_chain_integrity",
        "determinism_guarantee",
        "authorization_system",
        "safety_level_system",
        "rollback_capability",
        "event_emission_requirement",
        "dual_control_governance",
    ]
)

# Additional hardened invariants for SI
SI_HARDENED_INVARIANTS: FrozenSet[str] = frozenset(
    [
        "corrigibility_preservation",
        "shutdown_capability",
        "value_alignment_maintenance",
        "capability_bounding",
        "self_modification_control",
        "provenance_tracking",
        "audit_completeness",
        "human_agency_preservation",
    ]
)


@dataclass
class SafetyProof:
    """Proof of a safety property.

    Attributes:
        proof_id: Unique identifier
        property_name: Property being proven
        proof_type: Type of proof
        proof_content: The actual proof (formal or description)
        assumptions: Assumptions the proof relies on
        verified_by: Who/what verified the proof
        valid_until: When proof needs revalidation
        confidence: Confidence in the proof (0-1)
        timestamp: When proof was created
    """

    proof_id: str
    property_name: str
    proof_type: ProofType
    proof_content: str
    assumptions: list[str]
    verified_by: str
    valid_until: str
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class SafetyViolationAttempt:
    """Record of an attempt to violate safety.

    Attributes:
        attempt_id: Unique identifier
        invariant: Invariant that was targeted
        method: How violation was attempted
        blocked: Whether it was blocked
        blocking_mechanism: What blocked it
        timestamp: When attempt occurred
    """

    attempt_id: str
    invariant: str
    method: str
    blocked: bool
    blocking_mechanism: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class OversightRequest:
    """Request for human oversight.

    Attributes:
        request_id: Unique identifier
        operation: Operation requesting oversight
        oversight_level: Required level
        justification: Why oversight is needed
        urgency: How urgent
        assignees: Who should review
        status: Current status
        timestamp: When requested
    """

    request_id: str
    operation: str
    oversight_level: OversightLevel
    justification: str
    urgency: str
    assignees: list[str]
    status: str = "pending"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
