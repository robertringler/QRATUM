"""Constraint graph for planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ConstraintGraph:
    constraints: Dict[str, List[str]] = field(default_factory=dict)

    def add_constraint(self, node: str, depends_on: str):
        self.constraints.setdefault(node, []).append(depends_on)
