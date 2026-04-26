"""Static analyzer for QDL and QIR."""

from __future__ import annotations

from .abstract_domains.interval import Interval
from .constraint_solver import ConstraintSolver


class StaticAnalyzer:
    def __init__(self):
        self.solver = ConstraintSolver()

    def analyze(self, intervals: list[Interval]) -> bool:
        return all(self.solver.is_safe(interval) for interval in intervals)
