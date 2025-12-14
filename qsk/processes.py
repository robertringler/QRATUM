"""Deterministic process abstraction over DAGs."""
from __future__ import annotations

from dataclasses import dataclass, field

from qsk.scheduler import DeterministicScheduler


@dataclass
class DeterministicProcess:
    pid: str
    dag: list[dict[str, object]]
    scheduler: DeterministicScheduler = field(default_factory=DeterministicScheduler)

    def ordered_steps(self) -> list[dict[str, object]]:
        return self.scheduler.order(self.dag)

    def run(self) -> tuple[list[str], list[str]]:
        trace: list[str] = []
        for node in self.ordered_steps():
            trace.append(f"{self.pid}:{node['id']}")
        verification = ["verified" if self.scheduler.verify(trace) else "invalid"]
        return trace, verification
