"""Deterministic Merkle tree implementation."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class DeterministicMerkleTree:
    """Constructs a Merkle tree with deterministic hashing order."""

    leaves: list[str] = field(default_factory=list)

    def add_leaf(self, value: str) -> None:
        self.leaves.append(value)

    def _pairwise(self, nodes: list[str]) -> list[str]:
        paired: list[str] = []
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i + 1] if i + 1 < len(nodes) else left
            combined = f"{left}:{right}".encode("utf-8")
            paired.append(_hash(combined))
        return paired

    def root(self) -> str:
        if not self.leaves:
            return _hash(b"empty")
        level = [ _hash(v.encode("utf-8")) for v in self.leaves ]
        while len(level) > 1:
            level = self._pairwise(level)
        return level[0]

    def prove(self, index: int) -> tuple[str, list[tuple[str, str]]]:
        if index < 0 or index >= len(self.leaves):
            raise IndexError("leaf index out of range")
        hashes = [ _hash(v.encode("utf-8")) for v in self.leaves ]
        path: list[tuple[str, str]] = []
        idx = index
        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])
            pair_index = idx ^ 1
            sibling = hashes[pair_index]
            direction = "left" if pair_index < idx else "right"
            path.append((direction, sibling))
            idx //= 2
            hashes = self._pairwise(hashes)
        return _hash(self.leaves[index].encode("utf-8")), path

    def verify_proof(self, leaf_hash: str, path: list[tuple[str, str]], expected_root: str) -> bool:
        computed = leaf_hash
        for direction, sibling in path:
            if direction == "left":
                combined = f"{sibling}:{computed}".encode("utf-8")
            else:
                combined = f"{computed}:{sibling}".encode("utf-8")
            computed = _hash(combined)
        return computed == expected_root
