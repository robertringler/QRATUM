"""Deterministic compute budget allocator."""

from __future__ import annotations

from typing import Dict


def allocate(budget: int, tasks: Dict[str, int]) -> Dict[str, int]:
    if budget <= 0:
        return dict.fromkeys(tasks, 0)
    total_weight = sum(tasks.values()) or 1
    return {k: budget * v // total_weight for k, v in tasks.items()}
