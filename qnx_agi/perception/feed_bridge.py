"""Bridge normalized QREAL data into AGI percepts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Percept:
    kind: str
    payload: Dict[str, object]


def to_percept(kind: str, normalized: Dict[str, object]) -> Percept:
    return Percept(kind=kind, payload=dict(normalized))
