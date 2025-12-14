"""Symbolic executor for QDL programs."""
from __future__ import annotations


from .abstract_domains.interval import Interval
from .constraint_solver import ConstraintSolver


class SymbolicExecutor:
    def __init__(self):
        self.solver = ConstraintSolver()

    def execute(self, program) -> dict[str, Interval]:
        state: dict[str, Interval] = {}
        # placeholder symbolic propagation
        return state
