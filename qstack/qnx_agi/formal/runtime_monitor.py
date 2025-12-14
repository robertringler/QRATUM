"""Runtime invariant monitoring utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass
class RuntimeInvariant:
    name: str
    predicate: Callable[[Dict[str, float]], bool]
    severity: str = "critical"


@dataclass
class RuntimeMonitor:
    invariants: List[RuntimeInvariant] = field(default_factory=list)
    violations: List[Dict[str, str]] = field(default_factory=list)

    def register(self, invariant: RuntimeInvariant) -> None:
        self.invariants.append(invariant)

    def evaluate(self, state: Dict[str, float]) -> bool:
        self.violations.clear()
        ok = True
        for inv in self.invariants:
            if not inv.predicate(state):
                self.violations.append({"name": inv.name, "severity": inv.severity})
                ok = False
        return ok

    def report(self) -> List[Dict[str, str]]:
        return list(self.violations)
