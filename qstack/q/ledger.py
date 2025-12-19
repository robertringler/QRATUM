"""Deterministic append-only ledger with chained digests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LedgerEntry:
    index: int
    payload: Dict[str, str]
    prev_digest: str
    digest: str


@dataclass
class Ledger:
    entries: List[LedgerEntry] = field(default_factory=list)

    def _compute_digest(self, payload: Dict[str, str], prev_digest: str) -> str:
        material = f"{prev_digest}:{payload}".encode()
        return hashlib.sha256(material).hexdigest()

    def append(self, payload: Dict[str, str]) -> LedgerEntry:
        prev_digest = self.entries[-1].digest if self.entries else "genesis"
        digest = self._compute_digest(payload, prev_digest)
        entry = LedgerEntry(
            index=len(self.entries), payload=payload, prev_digest=prev_digest, digest=digest
        )
        self.entries.append(entry)
        return entry

    def verify(self) -> bool:
        prev_digest = "genesis"
        for idx, entry in enumerate(self.entries):
            expected = self._compute_digest(entry.payload, prev_digest)
            if entry.index != idx or entry.prev_digest != prev_digest or entry.digest != expected:
                return False
            prev_digest = entry.digest
        return True

    def head(self) -> Optional[LedgerEntry]:
        return self.entries[-1] if self.entries else None
