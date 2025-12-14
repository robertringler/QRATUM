"""Hierarchical goal decomposition utilities."""

from __future__ import annotations

from typing import Any, Dict, List


def decompose(goal: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "subgoals" in goal:
        return list(goal["subgoals"])
    return [goal]
