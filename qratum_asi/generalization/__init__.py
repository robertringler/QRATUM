"""QRATUM Generalization Layer for SI Transition.

This module extends QRATUM's bounded domain-specialized capabilities toward
general reasoning across arbitrary cognitive domains while preserving all
8 Fatal Invariants, determinism, provenance, and dual-control governance.

Key Components:
- GeneralReasoningEngine: Cross-domain knowledge synthesis
- ExtendedDomainRegistry: Beyond 14 verticals to arbitrary domains
- UniversalStateSpace: AHTC-compressed universal state representations
- HypothesisGenerator: Open-ended hypothesis generation framework

Version: 1.0.0
Status: Prototype (SI Transition Phase 1)
Constraints: 8 Fatal Invariants preserved, human-approved bounded improvements
"""

from qratum_asi.generalization.types import (
    CognitiveDomain,
    DomainCapability,
    CrossDomainHypothesis,
    SynthesisResult,
    UniversalStateVector,
    CompressionMetrics,
)
from qratum_asi.generalization.domain_registry import (
    ExtendedDomainRegistry,
    DomainDefinition,
    DomainInterconnection,
)
from qratum_asi.generalization.reasoning_engine import (
    GeneralReasoningEngine,
    ReasoningMode,
    CrossDomainSynthesizer,
)
from qratum_asi.generalization.hypothesis_generator import (
    HypothesisGenerator,
    HypothesisType,
    GenerationConstraints,
)
from qratum_asi.generalization.state_space import (
    UniversalStateSpace,
    StateCompressor,
    AHTCEncoder,
)

__all__ = [
    # Types
    "CognitiveDomain",
    "DomainCapability",
    "CrossDomainHypothesis",
    "SynthesisResult",
    "UniversalStateVector",
    "CompressionMetrics",
    # Domain registry
    "ExtendedDomainRegistry",
    "DomainDefinition",
    "DomainInterconnection",
    # Reasoning engine
    "GeneralReasoningEngine",
    "ReasoningMode",
    "CrossDomainSynthesizer",
    # Hypothesis generation
    "HypothesisGenerator",
    "HypothesisType",
    "GenerationConstraints",
    # State space
    "UniversalStateSpace",
    "StateCompressor",
    "AHTCEncoder",
]
