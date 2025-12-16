"""Deterministic provenance utilities for reality adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict


def _stable_dump(payload: Dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ProvenanceRecord:
    """Provenance attached to normalized data."""

    source: str
    tick: int
    digest: str
    raw_fingerprint: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "tick": self.tick,
            "digest": self.digest,
            "fingerprint": self.raw_fingerprint,
        }


def compute_provenance(source: str, normalized: Dict[str, object], tick: int) -> ProvenanceRecord:
    """Compute a deterministic provenance hash for normalized data."""

    serialized = _stable_dump(normalized)
    digest = hashlib.sha256(f"{source}:{tick}:{serialized}".encode()).hexdigest()
    raw_fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return ProvenanceRecord(
        source=source, tick=tick, digest=digest, raw_fingerprint=raw_fingerprint
    )
