"""QRATUM-OMNILEX v1.0 - Sovereign Deterministic Legal Analysis Engine.

This module provides comprehensive legal analysis capabilities that run natively
as QRATUM workloads with full deterministic replay, auditability, and sovereignty.

All legal reasoning executes as immutable, hash-chained contracts dispatched to
the Frankenstein Cluster, guaranteeing deterministic replay and auditability.

Includes QRATUM-JURIS: Ultra-Detailed Legal Analysis & Public-Record Exhaustion
for criminal case forensic analysis.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from omnilex.engine import QRATUMOmniLexEngine
from omnilex.juris import (
    CriminalCaseIntent,
    QRATUMJurisEngine,
    create_ohio_v_ringler_intent,
    generate_juris_intent_id,
)
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
    "QRATUMJurisEngine",
    "LegalQILIntent",
    "CriminalCaseIntent",
    "LegalTradition",
    "LegalDomain",
    "ReasoningFramework",
    "InterpretiveCanon",
    "Jurisdiction",
    "create_ohio_v_ringler_intent",
    "generate_juris_intent_id",
]

__version__ = "1.0.0"
