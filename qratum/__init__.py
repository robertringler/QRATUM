"""
QRATUM Sovereign AI Platform v2.0

A production-grade platform for deterministic, auditable AI computations
across 14 vertical domains with full QRATUM invariant enforcement.

Core Modules (per QRATUM ASCENSION DIRECTIVE):
- metrics: QRATUM metrics (OSR, CEI, SF, HRD)
- discovery: Self-Discovery Cycle (SDC) implementation
- sovereign_stack: Sovereign Stack layer formalization
"""

__version__ = "2.0.0"
__legacy_name__ = "QuASIM"
__license__ = "Apache-2.0"
__url__ = "https://qratum.io"
__github__ = "https://github.com/robertringler/QRATUM"

# Quantum core (real numpy state-vector simulator).
from .quantum_core import Circuit, Result, Simulator, StateVector, gates

__all__ = [
    "platform",
    "verticals",
    # Core directive modules
    "metrics",
    "discovery",
    "sovereign_stack",
    # Quantum core
    "Circuit",
    "Simulator",
    "StateVector",
    "Result",
    "gates",
]
