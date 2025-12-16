"""Replica synchronization."""

from __future__ import annotations

from typing import List


class ReplicaSync:
    def apply(self, ordered: List[dict]) -> List[dict]:
        return [dict(msg, state="applied") for msg in ordered]
