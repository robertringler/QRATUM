"""Deterministic timeline implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from qscenario.events import Event


@dataclass(frozen=True)
class TimelineEntry:
    tick: int
    events: List[Event]


class Timeline:
    """Maintains an ordered list of events keyed by logical ticks."""

    def __init__(self, entries: Iterable[TimelineEntry]) -> None:
        ordered = sorted(entries, key=lambda e: e.tick)
        self._entries: List[TimelineEntry] = ordered

    def stream(self) -> Iterable[Tuple[int, List[Event]]]:
        for entry in self._entries:
            yield entry.tick, list(entry.events)

    def describe(self) -> List[Dict[str, object]]:
        description: List[Dict[str, object]] = []
        for tick, events in self.stream():
            description.append(
                {
                    "tick": tick,
                    "events": [event.describe() for event in events],
                }
            )
        return description

    def append(self, tick: int, events: List[Event]) -> None:
        combined = self._entries + [TimelineEntry(tick, events)]
        combined.sort(key=lambda e: e.tick)
        self._entries = combined
