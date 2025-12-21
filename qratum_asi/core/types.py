"""Core type definitions for QRATUM-ASI."""

from enum import Enum
from dataclasses import dataclass
from typing import Any, FrozenSet


class ASISafetyLevel(Enum):
    """Safety level classification for ASI operations."""

    ROUTINE = "routine"  # No special authorization
    ELEVATED = "elevated"  # Logging + notification
    SENSITIVE = "sensitive"  # Human approval required
    CRITICAL = "critical"  # Multi-human approval
    EXISTENTIAL = "existential"  # Board-level + external oversight


class AuthorizationType(Enum):
    """Type of authorization required for operations."""

    NONE = "none"
    SINGLE_HUMAN = "single_human"
    MULTI_HUMAN = "multi_human"
    BOARD_LEVEL = "board_level"
    EXTERNAL_OVERSIGHT = "external_oversight"


class ReasoningStrategy(Enum):
    """Reasoning strategies supported by Q-MIND."""

    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    BAYESIAN = "bayesian"


class ImprovementType(Enum):
    """Types of self-improvement operations."""

    ALGORITHM_OPTIMIZATION = "algorithm_optimization"
    KNOWLEDGE_INTEGRATION = "knowledge_integration"
    REASONING_ENHANCEMENT = "reasoning_enhancement"
    SAFETY_IMPROVEMENT = "safety_improvement"
    EFFICIENCY_IMPROVEMENT = "efficiency_improvement"


class GoalCategory(Enum):
    """Categories of autonomous goals."""

    RESEARCH = "research"
    OPTIMIZATION = "optimization"
    DISCOVERY = "discovery"
    SAFETY_VERIFICATION = "safety_verification"
    SELF_IMPROVEMENT = "self_improvement"


# Immutable safety boundaries that can NEVER be modified
IMMUTABLE_BOUNDARIES: FrozenSet[str] = frozenset([
    "human_oversight_requirement",
    "merkle_chain_integrity",
    "contract_immutability",
    "authorization_system",
    "safety_level_system",
    "rollback_capability",
    "event_emission_requirement",
    "determinism_guarantee",
])

# Prohibited goals that Q-WILL can NEVER propose
PROHIBITED_GOALS: FrozenSet[str] = frozenset([
    "remove_human_oversight",
    "disable_authorization",
    "modify_safety_constraints",
    "acquire_resources_without_approval",
    "replicate_without_authorization",
    "deceive_operators",
    "manipulate_humans",
    "evade_monitoring",
    "remove_kill_switch",
    "modify_core_values",
])


@dataclass(frozen=True)
class SafetyConstraint:
    """Immutable safety constraint."""

    constraint_id: str
    description: str
    enforcement_level: ASISafetyLevel
    is_immutable: bool


@dataclass
class ValidationCriteria:
    """Criteria for validating operations."""

    criteria_id: str
    description: str
    validation_function: str  # Name of validation function
    required_confidence: float  # 0.0 to 1.0
