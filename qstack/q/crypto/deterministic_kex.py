"""Deterministic key exchange using shared identifiers."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..identity import QIdentity


@dataclass(frozen=True)
class DeterministicKeyExchange:
    """Derives a shared secret from two identities without randomness."""

    def derive_shared(self, a: QIdentity, b: QIdentity) -> str:
        ordered = sorted([a.key, b.key])
        material = f"{ordered[0]}:{ordered[1]}".encode()
        return hashlib.sha256(material).hexdigest()
