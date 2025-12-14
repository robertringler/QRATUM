"""Deterministic trace recorder for QNX runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceRecorder:
    records: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event: str, payload: dict[str, Any]) -> None:
        self.records.append({"event": event, "payload": payload})

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self.records)

    def clear(self) -> None:
        self.records.clear()
