"""QRATUM Safety Hardening Module for SI Transition.

Strengthens safety invariants for general agency, implements
scalable oversight mechanisms, and adds corrigibility mechanisms
that survive self-modification.

Key Components:
- InvariantHardener: Strengthen invariants with impossibility proofs
- ScalableOversight: Human-in-the-loop escalation for novel domains
- CorrigibilityPreserver: Corrigibility that survives self-modification

Version: 1.0.0
Status: Prototype (SI Transition Phase 5)
Constraints: 8 Fatal Invariants preserved and strengthened
"""

from qratum_asi.safety_hardening.types import (
    InvariantStrength,
    OversightLevel,
    CorrigibilityStatus,
    SafetyProof,
)
from qratum_asi.safety_hardening.invariant_hardener import (
    InvariantHardener,
    HardenedInvariant,
    ImpossibilityProof,
)
from qratum_asi.safety_hardening.scalable_oversight import (
    ScalableOversight,
    OversightEscalation,
    NovelDomainHandler,
)
from qratum_asi.safety_hardening.corrigibility import (
    CorrigibilityPreserver,
    CorrigibilityCheck,
    ShutdownCapability,
)

__all__ = [
    # Types
    "InvariantStrength",
    "OversightLevel",
    "CorrigibilityStatus",
    "SafetyProof",
    # Invariant hardener
    "InvariantHardener",
    "HardenedInvariant",
    "ImpossibilityProof",
    # Scalable oversight
    "ScalableOversight",
    "OversightEscalation",
    "NovelDomainHandler",
    # Corrigibility
    "CorrigibilityPreserver",
    "CorrigibilityCheck",
    "ShutdownCapability",
]
