"""Deterministic vector storage."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VectorDB:
    vectors: dict[str, list[float]] = field(default_factory=dict)

    def upsert(self, key: str, vector: list[float]) -> None:
        self.vectors[key] = list(vector)

    def query(self, key: str) -> list[float]:
        return list(self.vectors.get(key, []))
