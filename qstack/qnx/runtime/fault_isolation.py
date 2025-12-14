"""Fault isolation zones for deterministic containment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FaultIsolationZones:
    zones: Dict[str, List[str]] = field(default_factory=dict)

    def register(self, name: str, operators: List[str]) -> None:
        self.zones[name] = list(operators)

    def zone_for(self, operator: str) -> str:
        for name, ops in self.zones.items():
            if operator in ops:
                return name
        return "default"
