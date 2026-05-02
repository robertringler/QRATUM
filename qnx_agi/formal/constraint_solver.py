"""Constraint solver using simple interval checks."""

from __future__ import annotations

from .abstract_domains.interval import Interval


class ConstraintSolver:
    def is_safe(self, interval: Interval) -> bool:
        return interval.lower <= interval.upper
