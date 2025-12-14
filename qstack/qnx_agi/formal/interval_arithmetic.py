"""Deterministic interval arithmetic for runtime reasoning.

This module implements a compact interval type with common arithmetic
operations, enclosure checks, and affine propagation utilities used by
planners and formal analyzers. All operations avoid nondeterminism and favor
explicit bounds to align with safety-critical review.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Interval:
    """Closed interval [low, high]."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("interval lower bound greater than upper bound")

    def width(self) -> float:
        return float(self.high - self.low)

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0

    def add(self, other: "Interval") -> "Interval":
        return Interval(self.low + other.low, self.high + other.high)

    def sub(self, other: "Interval") -> "Interval":
        return Interval(self.low - other.high, self.high - other.low)

    def mul(self, other: "Interval") -> "Interval":
        candidates = (
            self.low * other.low,
            self.low * other.high,
            self.high * other.low,
            self.high * other.high,
        )
        return Interval(min(candidates), max(candidates))

    def div(self, other: "Interval") -> "Interval":
        if other.low <= 0.0 <= other.high:
            raise ZeroDivisionError("interval division crosses zero")
        candidates = (
            self.low / other.low,
            self.low / other.high,
            self.high / other.low,
            self.high / other.high,
        )
        return Interval(min(candidates), max(candidates))

    def intersect(self, other: "Interval") -> "Interval":
        low = max(self.low, other.low)
        high = min(self.high, other.high)
        if low > high:
            raise ValueError("empty interval intersection")
        return Interval(low, high)

    def union(self, other: "Interval") -> "Interval":
        return Interval(min(self.low, other.low), max(self.high, other.high))

    def clamp(self, bounds: "Interval") -> "Interval":
        return self.intersect(bounds)

    def to_tuple(self) -> tuple[float, float]:
        return (self.low, self.high)


def propagate_affine(intervals: dict[str, Interval], weights: dict[str, float], bias: float = 0.0) -> Interval:
    r"""Propagate an affine transform \sum w_i * x_i + bias over intervals."""
    low = bias
    high = bias
    for name, weight in sorted(weights.items(), key=lambda kv: kv[0]):
        iv = intervals.get(name)
        if iv is None:
            raise KeyError(f"missing interval for variable {name}")
        candidates = (weight * iv.low, weight * iv.high)
        low += min(candidates)
        high += max(candidates)
    return Interval(low, high)


class IntervalEnvironment:
    """Environment mapping variable names to intervals with deterministic updates."""

    def __init__(self, mapping: dict[str, Interval] | None = None):
        self._mapping: dict[str, Interval] = mapping or {}

    def assign(self, name: str, interval: Interval) -> None:
        self._mapping[name] = interval

    def read(self, name: str) -> Interval:
        if name not in self._mapping:
            raise KeyError(f"interval for {name} not defined")
        return self._mapping[name]

    def propagate_linear(self, coefficients: dict[str, float], bias: float = 0.0) -> Interval:
        return propagate_affine(self._mapping, coefficients, bias)

    def narrow(self, name: str, constraint: Interval) -> Interval:
        current = self.read(name)
        narrowed = current.intersect(constraint)
        self.assign(name, narrowed)
        return narrowed

    def snapshot(self) -> dict[str, tuple[float, float]]:
        return {k: v.to_tuple() for k, v in sorted(self._mapping.items(), key=lambda kv: kv[0])}

    def intervals(self) -> dict[str, Interval]:
        """Return a shallow copy of the current interval mapping."""
        return dict(self._mapping)

    def bulk_update(self, updates: dict[str, Interval]) -> None:
        for key, interval in updates.items():
            self.assign(key, interval)

    def propagate_non_linear(self, products: Iterable[tuple[str, str]]) -> dict[str, Interval]:
        """Propagate pairwise products for coarse abstract interpretation."""
        results: dict[str, Interval] = {}
        for left, right in products:
            lv = self.read(left)
            rv = self.read(right)
            prod = lv.mul(rv)
            results[f"{left}*{right}"] = prod
        return results
