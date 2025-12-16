"""Heuristic search planner using deterministic expansion."""

from __future__ import annotations

from typing import Any, Dict, List

from ..base import Planner


class HeuristicSearchPlanner(Planner):
    def __init__(self):
        super().__init__(lambda goal, state: len(goal))

    def plan(self, goal: Dict[str, Any], state: Dict[str, Any]) -> List[Dict[str, Any]]:
        ordered = sorted(goal.items(), key=lambda kv: kv[0])
        return [
            {"action": "set", "key": key, "value": value, "score": float(idx)}
            for idx, (key, value) in enumerate(ordered)
        ]
