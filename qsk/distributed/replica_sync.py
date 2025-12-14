"""Replica synchronization."""
from __future__ import annotations



class ReplicaSync:
    def apply(self, ordered: list[dict]) -> list[dict]:
        return [dict(msg, state='applied') for msg in ordered]
