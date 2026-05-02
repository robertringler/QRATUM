"""Logical resource accounting."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ResourceBudget:
    cpu: int
    memory: int
    usage: dict[str, int] = field(default_factory=lambda: {"cpu": 0, "memory": 0})

    def consume(self, cpu: int = 0, memory: int = 0) -> bool:
        next_cpu = self.usage["cpu"] + cpu
        next_mem = self.usage["memory"] + memory
        if next_cpu > self.cpu or next_mem > self.memory:
            return False
        self.usage["cpu"] = next_cpu
        self.usage["memory"] = next_mem
        return True

    def release(self, cpu: int = 0, memory: int = 0) -> None:
        self.usage["cpu"] = max(0, self.usage["cpu"] - cpu)
        self.usage["memory"] = max(0, self.usage["memory"] - memory)

    def snapshot(self) -> dict[str, int]:
        return dict(self.usage)
