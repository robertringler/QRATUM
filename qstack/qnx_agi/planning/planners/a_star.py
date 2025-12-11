"""A* planner wrapper."""
from __future__ import annotations

from typing import Any, Callable

from ..base import AStarPlanner, Planner

__all__ = ["build_a_star"]


def build_a_star(heuristic: Callable[[dict[str, Any], dict[str, Any]], float]) -> Planner:
    return AStarPlanner(heuristic)
