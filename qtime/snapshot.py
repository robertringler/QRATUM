"""Snapshot definitions for deterministic state capture."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict


def _canonical(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hash(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


@dataclass(frozen=True)
class Snapshot:
    state: Dict[str, Any]
    tick: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def serialize(self) -> Dict[str, Any]:
        return {"state": self.state, "tick": self.tick, "metadata": self.metadata}

    def snapshot_id(self) -> str:
        return _hash(self.serialize())

    def diff(self, other: Snapshot) -> Dict[str, Any]:
        differ = {}
        for key in set(self.state.keys()).union(other.state.keys()):
            left = self.state.get(key)
            right = other.state.get(key)
            if left != right:
                differ[key] = {"from": left, "to": right}
        return differ

    @staticmethod
    def restore(serialized: Dict[str, Any]) -> Snapshot:
        return Snapshot(
            state=serialized["state"],
            tick=int(serialized["tick"]),
            metadata=serialized.get("metadata", {}),
        )
