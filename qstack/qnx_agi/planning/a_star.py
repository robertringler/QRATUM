"""Advanced deterministic A* planner with safety constraints."""
from __future__ import annotations

from typing import Any, Callable

from ...qnx.runtime.safety import SafetyEnvelope
from .base import Planner, PlanStep


class ConstrainedAStarPlanner(Planner):
    def __init__(
        self,
        heuristic: Callable[[dict[str, Any], dict[str, Any]], float],
        envelope: SafetyEnvelope | None = None,
    ):
        super().__init__(heuristic)
        self._envelope = envelope

    def _neighbors(self, state: dict[str, Any], goal: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        neighbors: list[tuple[str, dict[str, Any]]] = []
        for key, value in goal.items():
            if state.get(key) != value:
                candidate = dict(state)
                candidate[key] = value
                if self._envelope:
                    pseudo_state = type("State", (), {"data": candidate})
                    if not self._envelope.inside(pseudo_state):
                        continue
                neighbors.append((f"set_{key}", candidate))
        return neighbors

    def plan(self, goal: dict[str, Any], state: dict[str, Any]) -> list[PlanStep]:
        open_set: list[PlanStep] = [PlanStep(action="start", parameters=state, cost=0.0, heuristic=self._heuristic(goal, state))]
        explored: dict[str, float] = {}
        while open_set:
            open_set.sort(key=lambda s: (s.total(), s.action))
            current = open_set.pop(0)
            if all(current.parameters.get(k) == v for k, v in goal.items()):
                path: list[PlanStep] = []
                cursor: PlanStep | None = current
                while cursor:
                    path.append(cursor)
                    cursor = cursor.parent
                return list(reversed(path))
            digest = str(sorted(current.parameters.items()))
            if digest in explored and explored[digest] <= current.total():
                continue
            explored[digest] = current.total()
            for action, neighbor in self._neighbors(current.parameters, goal):
                step_cost = current.cost + 1.0
                heuristic = self._heuristic(goal, neighbor)
                open_set.append(
                    PlanStep(action=action, parameters=neighbor, cost=step_cost, heuristic=heuristic, parent=current)
                )
        return []


__all__ = ["ConstrainedAStarPlanner"]
