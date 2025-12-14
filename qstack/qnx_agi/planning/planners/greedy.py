"""Greedy planner using deterministic scoring."""

from __future__ import annotations

from typing import Any, Dict, List

from ..base import Planner


def _score(goal: Dict[str, Any], state: Dict[str, Any]) -> float:
    overlap = 0
    for key, value in goal.items():
        if state.get(key) == value:
            overlap += 1
    return float(overlap)


class GreedyPlanner(Planner):
    def __init__(self):
        super().__init__(_score)

    def plan(self, goal: Dict[str, Any], state: Dict[str, Any]) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        for key, value in goal.items():
            steps.append(
                {"action": "set", "key": key, "value": value, "score": _score({key: value}, state)}
            )
        return steps
