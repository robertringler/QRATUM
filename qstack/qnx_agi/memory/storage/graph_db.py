"""Minimal knowledge graph backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class GraphDB:
    edges: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)

    def add_edge(self, subject: str, predicate: str, obj: str) -> None:
        neighbors = list(self.edges.get(subject, []))
        neighbors.append((predicate, obj))
        self.edges[subject] = neighbors

    def neighbors(self, subject: str) -> List[Tuple[str, str]]:
        return list(self.edges.get(subject, []))
