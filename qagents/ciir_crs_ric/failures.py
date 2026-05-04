"""failures.py — Type I-IV failure taxonomy.

Each failure type is an explicit exception carrying a deterministic, inspectable
record. Status codes (TYPE_I..TYPE_IV) are also exported as string literals for
use in trace entries by `executor.py`.

No silent failures: every constraint, parse, transition, or invariant breach
produces exactly one of these named errors with a structured payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Status literals appearing in TraceEntry.status -----------------------------
STATUS_OK: str = "OK"
STATUS_TYPE_I: str = "TYPE_I"  # Constraint violation (pre or post)
STATUS_TYPE_II: str = "TYPE_II"  # Intent parse failure (pi -> bottom)
STATUS_TYPE_III: str = "TYPE_III"  # Invalid transition (T_C -> bottom)
STATUS_TYPE_IV: str = "TYPE_IV"  # Invariant violation


@dataclass(frozen=True)
class FailureRecord:
    """Deterministic, inspectable record of a failure."""

    code: str  # one of TYPE_I..TYPE_IV
    name: str  # exception class name
    message: str  # human-readable message
    details: Dict[str, Any] = field(default_factory=dict)


class CIIRFailure(Exception):
    """Base class for all CIIR-CRS-RIC failures.

    Every subclass MUST set `code` and `name` as class attributes so that the
    executor can deterministically map an exception to a trace status.
    """

    code: str = "UNKNOWN"
    name: str = "CIIRFailure"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message: str = message
        self.details: Dict[str, Any] = dict(details)

    def record(self) -> FailureRecord:
        return FailureRecord(
            code=self.code,
            name=self.name,
            message=self.message,
            details=dict(self.details),
        )


class ConstraintViolation(CIIRFailure):
    """Type I — phi(s, c) = False on pre- or post-state."""

    code = STATUS_TYPE_I
    name = "ConstraintViolation"


class IntentParseFailure(CIIRFailure):
    """Type II — pi(raw) returned bottom (None)."""

    code = STATUS_TYPE_II
    name = "IntentParseFailure"


class InvalidTransition(CIIRFailure):
    """Type III — T_C(s, a) returned bottom (None)."""

    code = STATUS_TYPE_III
    name = "InvalidTransition"


class InvariantViolation(CIIRFailure):
    """Type IV — Inv(s) = False (pre or post execution)."""

    code = STATUS_TYPE_IV
    name = "InvariantViolation"


def classify(exc: BaseException) -> Optional[FailureRecord]:
    """Map a raised exception to its FailureRecord, or None if not a CIIRFailure."""
    if isinstance(exc, CIIRFailure):
        return exc.record()
    return None


__all__ = [
    "STATUS_OK",
    "STATUS_TYPE_I",
    "STATUS_TYPE_II",
    "STATUS_TYPE_III",
    "STATUS_TYPE_IV",
    "FailureRecord",
    "CIIRFailure",
    "ConstraintViolation",
    "IntentParseFailure",
    "InvalidTransition",
    "InvariantViolation",
    "classify",
]
