"""Coordinator for deterministic distributed compute."""
from __future__ import annotations

from .deterministic_transport import DeterministicTransport
from .replica_sync import ReplicaSync


class Coordinator:
    def __init__(self):
        self.transport = DeterministicTransport()
        self.sync = ReplicaSync()

    def broadcast_plan(self, plan: list[dict]) -> list[dict]:
        ordered = self.transport.order(plan)
        return self.sync.apply(ordered)
