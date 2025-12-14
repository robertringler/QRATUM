"""Deterministic process abstraction over DAGs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from qsk.scheduler import DeterministicScheduler


@dataclass
class DeterministicProcess:
    pid: str
    dag: List[Dict[str, object]]
    scheduler: DeterministicScheduler = field(default_factory=DeterministicScheduler)

    def ordered_steps(self) -> List[Dict[str, object]]:
        return self.scheduler.order(self.dag)

    def run(self) -> Tuple[List[str], List[str]]:
        trace: List[str] = []
        for node in self.ordered_steps():
            trace.append(f"{self.pid}:{node['id']}")
        verification = ["verified" if self.scheduler.verify(trace) else "invalid"]
        return trace, verification
