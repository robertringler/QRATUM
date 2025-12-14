"""Ledger record definitions with deterministic hashing."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional


_DEF_PREV = "GENESIS"


def _canonical(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _hash_payload(payload: dict[str, Any]) -> str:
    canonical = _canonical(payload)
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class LedgerRecord:
    """Single append-only ledger entry."""

    tick: int
    record_type: str
    payload: dict[str, Any]
    node_id: str
    prev_hash: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "tick": int(self.tick),
            "record_type": self.record_type,
            "payload": self.payload,
            "node_id": self.node_id,
            "prev_hash": self.prev_hash or _DEF_PREV,
            "metadata": self.metadata,
        }

    def compute_hash(self) -> str:
        return _hash_payload(self.canonical_payload())

    def with_prev_hash(self, prev_hash: str) -> "LedgerRecord":
        return LedgerRecord(
            tick=self.tick,
            record_type=self.record_type,
            payload=self.payload,
            node_id=self.node_id,
            prev_hash=prev_hash,
            metadata=self.metadata,
        )

    @staticmethod
    def genesis(record_type: str = "genesis", payload: Optional[dict[str, Any]] = None) -> "LedgerRecord":
        payload = payload or {"message": "ledger genesis"}
        return LedgerRecord(tick=0, record_type=record_type, payload=payload, node_id="system", prev_hash=_DEF_PREV)
