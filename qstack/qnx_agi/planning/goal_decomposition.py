"""Hierarchical goal decomposition utilities."""
from __future__ import annotations

from typing import Any


def decompose(goal: dict[str, Any]) -> list[dict[str, Any]]:
    if "subgoals" in goal:
        return list(goal["subgoals"])
    return [goal]
