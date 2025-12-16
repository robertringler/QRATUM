"""Typed events for scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Event:
    tick: int
    domain: str
    kind: str
    payload: Dict[str, object]

    def describe(self) -> Dict[str, object]:
        return {
            "tick": self.tick,
            "domain": self.domain,
            "kind": self.kind,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class SystemEvent(Event):
    pass


@dataclass(frozen=True)
class MarketEvent(Event):
    pass


@dataclass(frozen=True)
class MissionEvent(Event):
    pass


@dataclass(frozen=True)
class NodeEvent(Event):
    pass
