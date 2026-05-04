"""Counter-based deterministic tick generator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TickCounter:
    tick: int = 0

    def next(self) -> int:
        current = self.tick
        self.tick += 1
        return current

    def snapshot(self) -> int:
        return self.tick
