"""Deterministic compute budget allocator."""
from __future__ import annotations



def allocate(budget: int, tasks: dict[str, int]) -> dict[str, int]:
    if budget <= 0:
        return {k: 0 for k in tasks}
    total_weight = sum(tasks.values()) or 1
    return {k: budget * v // total_weight for k, v in tasks.items()}
