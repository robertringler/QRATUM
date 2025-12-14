"""QDL checker invoking formal analysis."""
from __future__ import annotations

from .abstract_domains.interval import Interval
from .static_analyzer import StaticAnalyzer


class QDLChecker:
    def __init__(self):
        self.analyzer = StaticAnalyzer()

    def check_intervals(self, intervals: list[Interval]) -> bool:
        return self.analyzer.analyze(intervals)
