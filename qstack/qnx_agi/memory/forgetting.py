"""Forgetting budget logic."""
from __future__ import annotations

from .base import MemorySystem


def enforce_budget(memory: MemorySystem, budget: int) -> list[tuple[str, object]]:
    entries = list(memory._buffer)
    if len(entries) <= budget:
        return entries
    to_remove = len(entries) - budget
    while to_remove > 0:
        memory._buffer.popleft()
        to_remove -= 1
    return list(memory._buffer)
