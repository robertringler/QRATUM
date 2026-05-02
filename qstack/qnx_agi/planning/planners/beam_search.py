"""Beam search planner wrapper."""

from __future__ import annotations

from typing import Any, Callable

from ..base import BeamSearchPlanner, Planner

__all__ = ["build_beam_search"]


def build_beam_search(
    heuristic: Callable[[dict[str, Any], dict[str, Any]], float], width: int = 3
) -> Planner:
    return BeamSearchPlanner(heuristic, width)
