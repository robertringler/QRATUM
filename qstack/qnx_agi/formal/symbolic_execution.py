"""Deterministic symbolic execution stubs built atop interval domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from .abstract_interpretation import AbstractState
from .interval_arithmetic import Interval, IntervalEnvironment


@dataclass(frozen=True)
class PathConstraint:
    expression: str
    interval: Interval


class SymbolicExecutor:
    """Symbolic executor that propagates intervals as symbolic states."""

    def __init__(self, transition: Callable[[Dict[str, float]], Dict[str, float]]):
        self._transition = transition

    def execute(self, inputs: Dict[str, Interval], steps: int = 1) -> Dict[str, Interval]:
        env = IntervalEnvironment(inputs)
        for _ in range(steps):
            current = {k: v.midpoint() for k, v in env.intervals().items()}
            updated = self._transition(current)
            for key, value in updated.items():
                env.assign(key, Interval(value, value))
        return env.intervals()

    def explore_paths(self, guards: List[PathConstraint]) -> AbstractState:
        state = AbstractState()
        for guard in guards:
            state.assign(guard.expression, guard.interval)
        return state
