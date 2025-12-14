"""Deterministic audit log."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .provenance import hash_payload


@dataclass
class AuditLog:
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, category: str, payload: Dict[str, Any]) -> str:
        digest = hash_payload(payload)
        self.entries.append({"category": category, "payload": payload, "digest": digest})
        return digest

    def export(self) -> List[Dict[str, Any]]:
        return list(self.entries)
