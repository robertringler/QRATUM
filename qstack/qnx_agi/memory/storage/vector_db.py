"""Deterministic vector storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class VectorDB:
    vectors: Dict[str, List[float]] = field(default_factory=dict)

    def upsert(self, key: str, vector: List[float]) -> None:
        self.vectors[key] = list(vector)

    def query(self, key: str) -> List[float]:
        return list(self.vectors.get(key, []))
