"""AION Proof Module.

Implements proof-carrying execution with:
- Proof synthesis at compile time
- Proof verification at load time
- Proof object serialization

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from .synthesis import (
    BorrowProof,
    EffectProof,
    ProofSynthesizer,
    RegionProof,
    SMTSolver,
)
from .verifier import (
    ProofContext,
    ProofTerm,
    ProofVerifier,
    SafetyTheorem,
)

__all__ = [
    "ProofVerifier",
    "ProofTerm",
    "ProofContext",
    "SafetyTheorem",
    "ProofSynthesizer",
    "SMTSolver",
    "BorrowProof",
    "EffectProof",
    "RegionProof",
]
