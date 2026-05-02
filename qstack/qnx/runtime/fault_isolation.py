"""Fault isolation zones for deterministic containment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FaultIsolationZones:
    zones: dict[str, list[str]] = field(default_factory=dict)

    def register(self, name: str, operators: list[str]) -> None:
        self.zones[name] = list(operators)

    def zone_for(self, operator: str) -> str:
        for name, ops in self.zones.items():
            if operator in ops:
                return name
        return "default"
