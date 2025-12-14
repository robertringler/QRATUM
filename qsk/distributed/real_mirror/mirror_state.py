"""Deterministic external mirror state tracking."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MirrorState:
    """Snapshot of mirrored external state keyed by domain."""

    domains: dict[str, dict[str, object]] = field(default_factory=dict)
    tick: int = 0

    def apply(self, domain: str, payload: dict[str, object], tick: int) -> None:
        self.domains[domain] = dict(payload)
        self.tick = max(self.tick, tick)

    def view(self, domain: str) -> dict[str, object]:
        return dict(self.domains.get(domain, {}))
