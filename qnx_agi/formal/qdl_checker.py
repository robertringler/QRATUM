"""QDL checker invoking formal analysis."""

from __future__ import annotations

from typing import List

from .abstract_domains.interval import Interval
from .static_analyzer import StaticAnalyzer


class QDLChecker:
    def __init__(self):
        self.analyzer = StaticAnalyzer()

    def check_intervals(self, intervals: List[Interval]) -> bool:
        return self.analyzer.analyze(intervals)
