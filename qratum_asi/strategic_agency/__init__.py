"""QRATUM Strategic Agency Module for SI Transition.

Enables autonomous formulation of long-term, multi-step strategic
objectives while preserving safety invariants, human oversight,
and dual-control governance for sensitive operations.

Key Components:
- StrategicGoalEngine: Autonomous goal formulation and decomposition
- UnboundedExploration: Safely-gated exploration modes
- ParadigmInvention: Framework for novel paradigm generation

Version: 1.0.0
Status: Prototype (SI Transition Phase 2)
Constraints: 8 Fatal Invariants preserved, human-approved bounded improvements
"""

from qratum_asi.strategic_agency.exploration import (
    ExplorationResult,
    SafetyGate,
    UnboundedExploration,
)
from qratum_asi.strategic_agency.goal_engine import (
    GoalDecomposition,
    ProgressAssessment,
    StrategicGoalEngine,
)
from qratum_asi.strategic_agency.paradigm_invention import (
    ParadigmInventionFramework,
    ParadigmValidation,
)
from qratum_asi.strategic_agency.types import (
    ExplorationConstraints,
    ExplorationMode,
    ObjectivePriority,
    ObjectiveType,
    ParadigmProposal,
    StrategicObjective,
)

__all__ = [
    # Types
    "StrategicObjective",
    "ObjectiveType",
    "ObjectivePriority",
    "ExplorationMode",
    "ExplorationConstraints",
    "ParadigmProposal",
    # Goal engine
    "StrategicGoalEngine",
    "GoalDecomposition",
    "ProgressAssessment",
    # Exploration
    "UnboundedExploration",
    "ExplorationResult",
    "SafetyGate",
    # Paradigm invention
    "ParadigmInventionFramework",
    "ParadigmValidation",
]
