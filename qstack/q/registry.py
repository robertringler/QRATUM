"""Identity registry binding identities to attestations and ledger entries."""

from __future__ import annotations

from dataclasses import dataclass, field

from .attestation import Attestor
from .identity import QIdentity
from .ledger import Ledger


@dataclass
class IdentityRecord:
    identity: QIdentity
    attestation: dict[str, str]
    ledger_index: int


@dataclass
class IdentityRegistry:
    attestor: Attestor
    ledger: Ledger
    records: dict[str, IdentityRecord] = field(default_factory=dict)

    def register(self, identity: QIdentity, claims: dict[str, str]) -> IdentityRecord:
        attested = self.attestor.attest({"identity": identity.to_dict(), "claims": claims})
        entry = self.ledger.append(attested)
        record = IdentityRecord(identity=identity, attestation=attested, ledger_index=entry.index)
        self.records[identity.name] = record
        return record

    def lookup(self, name: str) -> IdentityRecord | None:
        return self.records.get(name)

    def verify(self, name: str) -> bool:
        record = self.records.get(name)
        if not record:
            return False
        return self.attestor.verify(record.attestation)
