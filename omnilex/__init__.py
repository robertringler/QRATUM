"""QRATUM-OMNILEX v1.0 - Sovereign Deterministic Legal Analysis Engine.

This module provides comprehensive legal analysis capabilities that run natively
as QRATUM workloads with full deterministic replay, auditability, and sovereignty.

All legal reasoning executes as immutable, hash-chained contracts dispatched to
the Frankenstein Cluster, guaranteeing deterministic replay and auditability.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from omnilex.engine import QRATUMOmniLexEngine
from omnilex.ontology import (
    InterpretiveCanon,
    Jurisdiction,
    LegalDomain,
    LegalTradition,
    ReasoningFramework,
)
from omnilex.qil_legal import LegalQILIntent

__all__ = [
    "QRATUMOmniLexEngine",
    "LegalQILIntent",
    "LegalTradition",
    "LegalDomain",
    "ReasoningFramework",
    "InterpretiveCanon",
    "Jurisdiction",
]

__version__ = "1.0.0"
