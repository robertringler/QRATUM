"""Bind node identity to Q identity layer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeIdentity:
    node_id: str
    public_key: str
    trust_level: str = "baseline"

    def attest(self) -> dict[str, str]:
        return {"node_id": self.node_id, "public_key": self.public_key, "trust_level": self.trust_level}

    def verify_signature(self, payload: str, signature: str) -> bool:
        expected = f"{self.node_id}:{payload}"
        return signature == expected
