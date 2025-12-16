"""Structured incident logging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Incident:
    category: str
    detail: Dict[str, object]
    sequence: int


class IncidentLog:
    def __init__(self) -> None:
        self._entries: List[Incident] = []
        self._counter = 0

    def record(self, category: str, detail: Dict[str, object]) -> Incident:
        self._counter += 1
        incident = Incident(category=category, detail=dict(detail), sequence=self._counter)
        self._entries.append(incident)
        return incident

    def recent(self, limit: int = 5) -> List[Incident]:
        return list(self._entries[-limit:])

    def all(self) -> List[Incident]:
        return list(self._entries)
