"""Snapshot definitions for deterministic state capture."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


@dataclass(frozen=True)
class Snapshot:
    state: dict[str, Any]
    tick: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def serialize(self) -> dict[str, Any]:
        return {"state": self.state, "tick": self.tick, "metadata": self.metadata}

    def snapshot_id(self) -> str:
        return _hash(self.serialize())

    def diff(self, other: Snapshot) -> dict[str, Any]:
        differ = {}
        for key in set(self.state.keys()).union(other.state.keys()):
            left = self.state.get(key)
            right = other.state.get(key)
            if left != right:
                differ[key] = {"from": left, "to": right}
        return differ

    @staticmethod
    def restore(serialized: dict[str, Any]) -> Snapshot:
        return Snapshot(
            state=serialized["state"],
            tick=int(serialized["tick"]),
            metadata=serialized.get("metadata", {}),
        )
