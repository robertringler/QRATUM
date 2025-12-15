"""Symbolic executor for QDL programs."""

from __future__ import annotations

from typing import Dict

from .abstract_domains.interval import Interval
from .constraint_solver import ConstraintSolver


class SymbolicExecutor:
    def __init__(self):
        self.solver = ConstraintSolver()

    def execute(self, program) -> Dict[str, Interval]:
        state: Dict[str, Interval] = {}
        # placeholder symbolic propagation
        return state
