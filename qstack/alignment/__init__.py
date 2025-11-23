"""Alignment and constitutional safety layer for Q-Stack."""

from qstack.alignment.constitution import Constitution, ConstitutionalArticle, DEFAULT_CONSTITUTION
from qstack.alignment.evaluator import AlignmentEvaluator
from qstack.alignment.violations import AlignmentViolation, ViolationSeverity

__all__ = [
    "AlignmentEvaluator",
    "AlignmentViolation",
    "ViolationSeverity",
    "Constitution",
    "ConstitutionalArticle",
    "DEFAULT_CONSTITUTION",
]
