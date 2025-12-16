"""Deterministic replay buffer for runtime traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class DeterministicReplayBuffer:
    capacity: int
    buffer: List[Any] = field(default_factory=list)

    def append(self, item: Any) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(item)

    def snapshot(self) -> List[Any]:
        return list(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()
