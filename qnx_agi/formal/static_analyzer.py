"""Static analyzer for QDL and QIR."""

from __future__ import annotations

from typing import List

from .abstract_domains.interval import Interval
from .constraint_solver import ConstraintSolver


class StaticAnalyzer:
    def __init__(self):
        self.solver = ConstraintSolver()

    def analyze(self, intervals: List[Interval]) -> bool:
        for interval in intervals:
            if not self.solver.is_safe(interval):
                return False
        return True
