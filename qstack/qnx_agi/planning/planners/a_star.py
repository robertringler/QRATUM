"""A* planner wrapper."""

from __future__ import annotations

from typing import Any, Callable, Dict

from ..base import AStarPlanner, Planner

__all__ = ["build_a_star"]


def build_a_star(heuristic: Callable[[Dict[str, Any], Dict[str, Any]], float]) -> Planner:
    return AStarPlanner(heuristic)
