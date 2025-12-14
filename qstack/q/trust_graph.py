"""Minimal deterministic trust graph for Q identities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from .attestation import Attestor
from .identity import QIdentity


@dataclass
class TrustGraph:
    """Tracks directed trust edges between identities."""

    edges: Dict[str, Set[str]] = field(default_factory=dict)

    def add_trust(self, source: QIdentity, target: QIdentity) -> None:
        followers = set(self.edges.get(source.name, set()))
        followers.add(target.name)
        self.edges[source.name] = followers

    def trusted(self, source: QIdentity) -> Set[str]:
        return set(self.edges.get(source.name, set()))

    def verify_path(self, start: QIdentity, end: QIdentity) -> bool:
        """Deterministically checks reachability using DFS without randomness."""
        visited: Set[str] = set()
        stack: List[str] = [start.name]
        while stack:
            current = stack.pop()
            if current == end.name:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(sorted(self.edges.get(current, set())))
        return False

    def attest_edge(
        self, attestor: Attestor, source: QIdentity, target: QIdentity
    ) -> Dict[str, str]:
        payload = {"source": source.to_dict(), "target": target.to_dict()}
        return attestor.attest(payload)
