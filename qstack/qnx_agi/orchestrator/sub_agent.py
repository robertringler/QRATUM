"""SubAgent orchestrating partial goals."""

from __future__ import annotations

from typing import Any

from ..planning.base import PlanningSystem
from ..planning.planners.greedy import GreedyPlanner


class SubAgent:
    def __init__(self, name: str):
        self.name = name
        self._planner = PlanningSystem(GreedyPlanner())

    def handle(self, goal: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
        return self._planner.evaluate(goal, state)
