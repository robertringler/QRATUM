"""Cluster-level event log."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ClusterEvent:
    sequence: int
    kind: str
    payload: Dict[str, object]


class EventLog:
    def __init__(self) -> None:
        self._events: List[ClusterEvent] = []
        self._counter = 0

    def append(self, kind: str, payload: Dict[str, object]) -> ClusterEvent:
        self._counter += 1
        event = ClusterEvent(sequence=self._counter, kind=kind, payload=dict(payload))
        self._events.append(event)
        return event

    def events(self) -> List[ClusterEvent]:
        return list(self._events)
