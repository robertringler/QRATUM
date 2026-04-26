"""Episodic memory with deterministic decay."""

from __future__ import annotations

from typing import Any

from .base import MemorySystem


class EpisodicMemoryWithDecay(MemorySystem):
    def __init__(self, capacity: int = 16, decay: int = 4):
        super().__init__(capacity=capacity)
        self.decay = decay

    def record_episode(self, step: int, summary: dict[str, Any]) -> None:
        self.record(str(step), summary)
        self._apply_decay()

    def _apply_decay(self) -> None:
        if len(self._buffer) <= self.capacity:
            return
        while len(self._buffer) > self.capacity:
            self._buffer.popleft()
