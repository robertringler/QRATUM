"""Deterministic capability token issuance."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..identity import QIdentity


def _hash_token(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DeterministicCapabilityToken:
    subject: str
    permission: str
    issuer: str
    token: str


class CapabilityAuthority:
    """Issues deterministic capability tokens."""

    def issue(self, issuer: QIdentity, subject: QIdentity, permission: str) -> DeterministicCapabilityToken:
        material = f"{issuer.key}:{subject.key}:{permission}"
        return DeterministicCapabilityToken(subject=subject.name, permission=permission, issuer=issuer.name, token=_hash_token(material))

    def validate(self, token: DeterministicCapabilityToken, subject: QIdentity, permission: str, issuer: QIdentity) -> bool:
        expected = self.issue(issuer, subject, permission)
        return expected.token == token.token and token.subject == subject.name and token.permission == permission and token.issuer == issuer.name

    def to_dict(self, token: DeterministicCapabilityToken) -> dict[str, str]:
        return {
            "subject": token.subject,
            "permission": token.permission,
            "issuer": token.issuer,
            "token": token.token,
        }
