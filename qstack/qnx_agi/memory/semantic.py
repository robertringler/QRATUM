"""Semantic memory implemented as a simple knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass, field

from .storage.graph_db import GraphDB


@dataclass
class SemanticMemory:
    graph: GraphDB = field(default_factory=GraphDB)

    def upsert_fact(self, subject: str, predicate: str, obj: str) -> None:
        self.graph.add_edge(subject, predicate, obj)

    def query(self, subject: str) -> list[tuple[str, str]]:
        return self.graph.neighbors(subject)
