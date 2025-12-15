"""Deterministic hooks for model checking and invariant validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List


@dataclass(frozen=True)
class TLASpecification:
    name: str
    init: Callable[[Dict[str, float]], bool]
    next_state: Callable[[Dict[str, float]], Dict[str, float]]
    invariant: Callable[[Dict[str, float]], bool]


class ModelChecker:
    """Simple bounded model checker for deterministic systems."""

    def __init__(self, specification: TLASpecification, bound: int = 5):
        self._spec = specification
        self._bound = bound

    def run(self, start: Dict[str, float]) -> List[Dict[str, float]]:
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
    def check_all(
        traces: Iterable[Dict[str, float]], predicate: Callable[[Dict[str, float]], bool]
    ) -> bool:
        return all(predicate(state) for state in traces)
