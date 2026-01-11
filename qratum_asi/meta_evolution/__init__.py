"""QRATUM Meta-Evolution Module for SI Transition.

Upgrades reinjection orchestration to multi-layer self-modification
including architecture redesign proposals, algorithm invention, and
meta-reinjection where discoveries about the improvement process
feed back into the system.

Key Components:
- MetaEvolutionEngine: Multi-layer self-modification orchestration
- AbstractionLevelManager: Q-EVOLVE cycles at increasing abstraction
- MetaReinjectionLoop: Self-improvement discovery feedback

Version: 1.0.0
Status: Prototype (SI Transition Phase 3)
Constraints: 8 Fatal Invariants preserved, human-approved bounded improvements
"""

from qratum_asi.meta_evolution.types import (
    AbstractionLevel,
    EvolutionProposal,
    EvolutionType,
    MetaDiscovery,
    SafetyVerification,
)
from qratum_asi.meta_evolution.engine import (
    MetaEvolutionEngine,
    EvolutionCycle,
    EvolutionResult,
)
from qratum_asi.meta_evolution.abstraction import (
    AbstractionLevelManager,
    AbstractionTransition,
)
from qratum_asi.meta_evolution.meta_reinjection import (
    MetaReinjectionLoop,
    MetaFeedbackResult,
)

__all__ = [
    # Types
    "AbstractionLevel",
    "EvolutionProposal",
    "EvolutionType",
    "MetaDiscovery",
    "SafetyVerification",
    # Engine
    "MetaEvolutionEngine",
    "EvolutionCycle",
    "EvolutionResult",
    # Abstraction management
    "AbstractionLevelManager",
    "AbstractionTransition",
    # Meta-reinjection
    "MetaReinjectionLoop",
    "MetaFeedbackResult",
]
