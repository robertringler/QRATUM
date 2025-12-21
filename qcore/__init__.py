"""QCORE - Canonical semantic problem definitions and Hamiltonian abstractions.

This module provides the foundational abstractions for quantum computing problems,
including semantic state representation and Hamiltonian encoding.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from qcore.hamiltonian import Hamiltonian, PauliTerm
from qcore.semantic_state import (
    ChemistryValidator,
    DomainValidator,
    FinanceValidator,
    OptimizationValidator,
    SemanticState,
)

__all__ = [
    "SemanticState",
    "DomainValidator",
    "ChemistryValidator",
    "OptimizationValidator",
    "FinanceValidator",
    "Hamiltonian",
    "PauliTerm",
]

__version__ = "1.0.0"
