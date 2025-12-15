"""Deterministic ledger with Merkle anchoring and attestation verification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..attestation import Attestor
from .deterministic_merkle import DeterministicMerkleTree


def _hash_payload(payload: Dict[str, str], prev: str) -> str:
    material = f"{prev}:{sorted(payload.items())}".encode()
    return hashlib.sha256(material).hexdigest()


@dataclass(frozen=True)
class DeterministicLedgerEntry:
    index: int
    payload: Dict[str, str]
    prev_digest: str
    digest: str
    attestation: Optional[Dict[str, str]] = None


@dataclass
class DeterministicLedger:
    """Append-only ledger with Merkle anchoring for integrity proofs."""

    attestor: Optional[Attestor] = None
    entries: List[DeterministicLedgerEntry] = field(default_factory=list)

    def append(self, payload: Dict[str, str]) -> DeterministicLedgerEntry:
        prev_digest = self.entries[-1].digest if self.entries else "genesis"
        digest = _hash_payload(payload, prev_digest)
        attestation = self.attestor.attest(payload) if self.attestor else None
        entry = DeterministicLedgerEntry(
            index=len(self.entries),
            payload=payload,
            prev_digest=prev_digest,
            digest=digest,
            attestation=attestation,
        )
        self.entries.append(entry)
        return entry

    def merkle_root(self) -> str:
        tree = DeterministicMerkleTree()
        for entry in self.entries:
            tree.add_leaf(entry.digest)
        return tree.root()

    def verify(self) -> bool:
        prev_digest = "genesis"
        for idx, entry in enumerate(self.entries):
            recomputed = _hash_payload(entry.payload, prev_digest)
            if entry.index != idx or entry.prev_digest != prev_digest or recomputed != entry.digest:
                return False
            if entry.attestation and self.attestor:
                if not self.attestor.verify(entry.attestation):
                    return False
            prev_digest = entry.digest
        return True

    def head(self) -> Optional[DeterministicLedgerEntry]:
        return self.entries[-1] if self.entries else None

    def export_chain(self) -> List[Dict[str, str]]:
        return [entry.__dict__ for entry in self.entries]
