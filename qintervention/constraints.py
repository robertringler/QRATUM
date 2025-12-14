"""Constraints for interventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

from qintervention.actions import ScheduledAction


@dataclass
class Constraint:
    name: str
    check: Callable[[ScheduledAction], bool]

    def allows(self, action: ScheduledAction) -> bool:
        return self.check(action)


@dataclass
class ConstraintSet:
    constraints: List[Constraint] = field(default_factory=list)

    def add(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)

    def allows(self, action: ScheduledAction) -> bool:
        return all(constraint.allows(action) for constraint in self.constraints)


DEFAULT_SAFE_DOMAINS = {"finance", "aerospace", "grid", "network", "pharma"}


def domain_whitelist(domains: set[str] | None = None) -> Constraint:
    allowed = domains or DEFAULT_SAFE_DOMAINS
    return Constraint("domain_whitelist", lambda action: action.action.kind in allowed)
