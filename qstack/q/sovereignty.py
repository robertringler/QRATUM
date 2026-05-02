import hashlib
from dataclasses import dataclass
from typing import Dict

from .identity import QIdentity


@dataclass(frozen=True)
class SovereignObject:
    """

    Bundles an identity with a set of claims and produces a deterministic digest.
    """

    identity: QIdentity
    claims: Dict[str, str]

    def digest(self) -> str:
        payload = repr({"id": self.identity.to_dict(), "claims": self.claims})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def describe(self) -> dict:
        return {
            "identity": self.identity.to_dict(),
            "claims": self.claims,
            "digest": self.digest(),
        }
