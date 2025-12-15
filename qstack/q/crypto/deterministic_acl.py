"""Deterministic ACL management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set

from ..identity import QIdentity


@dataclass
class DeterministicAccessControlList:
    permissions: Dict[str, Set[str]] = field(default_factory=dict)

    def grant(self, identity: QIdentity, permission: str) -> None:
        perms = set(self.permissions.get(identity.name, set()))
        perms.add(permission)
        self.permissions[identity.name] = perms

    def revoke(self, identity: QIdentity, permission: str) -> None:
        perms = set(self.permissions.get(identity.name, set()))
        perms.discard(permission)
        self.permissions[identity.name] = perms

    def allowed(self, identity: QIdentity, permission: str) -> bool:
        return permission in self.permissions.get(identity.name, set())

    def snapshot(self) -> Dict[str, Set[str]]:
        return {k: set(v) for k, v in self.permissions.items()}
