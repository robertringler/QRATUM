from dataclasses import dataclass
from typing import Any, Dict

from .signing import Signer


@dataclass
class Attestor:
    """Creates and validates deterministic attestations."""

    signer: Signer

    def attest(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        signature = self.signer.sign(payload)
        return {"payload": payload, "signature": signature}

    def verify(self, attestation: Dict[str, Any]) -> bool:
        return self.signer.verify(attestation["payload"], attestation["signature"])
