"""Deterministic hooks for model checking and invariant validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class TLASpecification:
    name: str
    init: Callable[[dict[str, float]], bool]
    next_state: Callable[[dict[str, float]], dict[str, float]]
    invariant: Callable[[dict[str, float]], bool]


class ModelChecker:
    """Simple bounded model checker for deterministic systems."""

    def __init__(self, specification: TLASpecification, bound: int = 5):
        self._spec = specification
        self._bound = bound

    def run(self, start: dict[str, float]) -> list[dict[str, float]]:
        if not self._spec.init(start):
            raise ValueError("initial state violates init predicate")
        states = [dict(start)]
        current = dict(start)
        for _ in range(self._bound):
            if not self._spec.invariant(current):
                raise AssertionError("invariant violated")
            current = self._spec.next_state(current)
            states.append(dict(current))
        if not self._spec.invariant(current):
            raise AssertionError("invariant violated at bound")
        return states

    @staticmethod
    def check_all(traces: Iterable[dict[str, float]], predicate: Callable[[dict[str, float]], bool]) -> bool:
        return all(predicate(state) for state in traces)
