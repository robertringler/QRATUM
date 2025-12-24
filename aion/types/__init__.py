"""AION Types Module.

Implements the dependent type system with:
- DLETS logic (dependent types + linear resources + separation + graded effects)
- Type judgments
- SMT integration for refinements

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from .type_system import (
    AIONTypeSystem,
    TypeContext,
    TypeJudgment,
    LinearContext,
    RefinementType,
    DependentType,
    TypeChecker,
)

__all__ = [
    "AIONTypeSystem",
    "TypeContext",
    "TypeJudgment",
    "LinearContext",
    "RefinementType",
    "DependentType",
    "TypeChecker",
]
