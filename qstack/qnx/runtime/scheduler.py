from __future__ import annotations

from typing import Any


class DeterministicScheduler:
    """Executes operators in lexicographic order of their registered names."""

    def __init__(self, operators):
        self._operators = operators

    def schedule(self, state: Any, goal: Any) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        for name, op in sorted(self._operators.available().items(), key=lambda kv: kv[0]):
            result = op.execute(state, goal)
            trace.append({"op": name, "result": result})
        return trace


class PriorityScheduler(DeterministicScheduler):
    """Deterministic priority scheduling based on integer priorities."""

    def __init__(self, operators, priorities: dict[str, int]):
        super().__init__(operators)
        self._priorities = priorities

    def schedule(self, state: Any, goal: Any) -> list[dict[str, Any]]:
        def sort_key(item):
            name, _op = item
            return (self._priorities.get(name, 0), name)

        trace: list[dict[str, Any]] = []
        for name, op in sorted(self._operators.available().items(), key=sort_key):
            result = op.execute(state, goal)
            trace.append({"op": name, "priority": self._priorities.get(name, 0), "result": result})
        return trace
