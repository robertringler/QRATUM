"""Constraint graph for planning."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConstraintGraph:
    constraints: dict[str, list[str]] = field(default_factory=dict)

    def add_constraint(self, node: str, depends_on: str):
        self.constraints.setdefault(node, []).append(depends_on)
