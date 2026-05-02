"""Bridge normalized QREAL data into AGI percepts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Percept:
    kind: str
    payload: dict[str, object]


def to_percept(kind: str, normalized: dict[str, object]) -> Percept:
    return Percept(kind=kind, payload=dict(normalized))
