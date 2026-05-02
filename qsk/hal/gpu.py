"""GPU abstraction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPU:
    multiprocessors: int = 16
    memory_gb: float = 8.0

    def profile(self) -> dict:
        return {"multiprocessors": self.multiprocessors, "memory_gb": self.memory_gb}
