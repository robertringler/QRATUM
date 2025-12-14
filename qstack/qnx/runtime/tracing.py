"""Deterministic trace recorder for QNX runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TraceRecorder:
    records: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, event: str, payload: Dict[str, Any]) -> None:
        self.records.append({"event": event, "payload": payload})

    def snapshot(self) -> List[Dict[str, Any]]:
        return list(self.records)

    def clear(self) -> None:
        self.records.clear()
