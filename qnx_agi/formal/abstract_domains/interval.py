"""Interval abstract domain."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Interval:
    lower: float
    upper: float

    def join(self, other: Interval) -> Interval:
        return Interval(min(self.lower, other.lower), max(self.upper, other.upper))
