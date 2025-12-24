"""AION Proof Module.

Implements proof-carrying execution with:
- Proof synthesis at compile time
- Proof verification at load time
- Proof object serialization

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from .verifier import (
    ProofVerifier,
    ProofTerm,
    ProofContext,
    SafetyTheorem,
)
from .synthesis import (
    ProofSynthesizer,
    SMTSolver,
    BorrowProof,
    EffectProof,
    RegionProof,
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
