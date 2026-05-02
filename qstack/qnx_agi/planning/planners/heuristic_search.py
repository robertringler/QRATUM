"""Heuristic search planner using deterministic expansion."""

from __future__ import annotations

from typing import Any

from ..base import Planner


class HeuristicSearchPlanner(Planner):
    def __init__(self):
        super().__init__(lambda goal, state: len(goal))

    def plan(self, goal: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
        ordered = sorted(goal.items(), key=lambda kv: kv[0])
        return [
            {"action": "set", "key": key, "value": value, "score": float(idx)}
            for idx, (key, value) in enumerate(ordered)
        ]
