"""Deterministic revocation list tracking."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict

from ..identity import QIdentity


def _reason_hash(reason: str) -> str:
    return hashlib.sha256(reason.encode("utf-8")).hexdigest()


@dataclass
class DeterministicRevocationList:
    revoked: Dict[str, str] = field(default_factory=dict)

    def revoke(self, identity: QIdentity, reason: str) -> None:
        self.revoked[identity.name] = _reason_hash(reason)

    def is_revoked(self, identity: QIdentity) -> bool:
        return identity.name in self.revoked

    def verify_reason(self, identity: QIdentity, reason: str) -> bool:
        return self.revoked.get(identity.name) == _reason_hash(reason)

    def export(self) -> Dict[str, str]:
        return dict(self.revoked)
