"""Deterministic audit log."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .provenance import hash_payload


@dataclass
class AuditLog:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, category: str, payload: dict[str, Any]) -> str:
        digest = hash_payload(payload)
        self.entries.append({"category": category, "payload": payload, "digest": digest})
        return digest

    def export(self) -> list[dict[str, Any]]:
        return list(self.entries)
