"""Runtime invariant monitoring utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class RuntimeInvariant:
    name: str
    predicate: Callable[[dict[str, float]], bool]
    severity: str = "critical"


@dataclass
class RuntimeMonitor:
    invariants: list[RuntimeInvariant] = field(default_factory=list)
    violations: list[dict[str, str]] = field(default_factory=list)

    def register(self, invariant: RuntimeInvariant) -> None:
        self.invariants.append(invariant)

    def evaluate(self, state: dict[str, float]) -> bool:
        self.violations.clear()
        ok = True
        for inv in self.invariants:
            if not inv.predicate(state):
                self.violations.append({"name": inv.name, "severity": inv.severity})
                ok = False
        return ok

    def report(self) -> list[dict[str, str]]:
        return list(self.violations)
