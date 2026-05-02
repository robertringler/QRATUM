"""Alignment violation models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ViolationSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


@dataclass(frozen=True)
class AlignmentViolation:
    """Represents a single alignment violation instance."""

    operation: str
    article_id: str
    policy_id: str
    message: str
    severity: ViolationSeverity

    def as_dict(self) -> dict[str, str]:
        return {
            "operation": self.operation,
            "article_id": self.article_id,
            "policy_id": self.policy_id,
            "message": self.message,
            "severity": self.severity.value,
        }
