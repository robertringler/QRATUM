"""Topological Observer - Read-only Persistent Homology Instrumentation Layer.

This module provides non-authoritative topological annotations for the QRATUM
epistemic substrate. All diagnostic outputs are informational and never override
jurisdictional execution.

Core Principles:
- Read-only observation (no mutation of observed data)
- Non-authoritative annotations (inform but never override)
- Verifiable topological metrics
- Cryptographically attestable observations

Implemented Features:
- Persistent homology computation (Betti numbers β₀, β₁, β₂)
- Persistence diagram generation
- Topological feature extraction
- Invariant-preserving observation protocol

Version: 1.0.0
Status: Production
"""

__version__ = "1.0.0"
__author__ = "QRATUM Team"

from .homology import (
    BettiNumbers,
    PersistenceDiagram,
    PersistentHomologyObserver,
    TopologicalAnnotation,
    compute_betti_numbers,
    compute_persistent_homology,
)
from .observer import (
    InvariantAssertion,
    ObservationResult,
    TopologicalInstrumentationLayer,
)

__all__ = [
    "PersistentHomologyObserver",
    "BettiNumbers",
    "PersistenceDiagram",
    "TopologicalAnnotation",
    "compute_persistent_homology",
    "compute_betti_numbers",
    "TopologicalInstrumentationLayer",
    "ObservationResult",
    "InvariantAssertion",
]
