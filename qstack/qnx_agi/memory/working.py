"""Working memory that caches the latest percepts."""
from __future__ import annotations

from typing import Any

from .base import MemorySystem


class WorkingMemory(MemorySystem):
    def __init__(self, capacity: int = 8):
        super().__init__(capacity=capacity)

    def update_from_percepts(self, percepts: list[Any]) -> None:
        for percept in percepts:
            self.record(percept.modality, percept.value)
