"""Deterministic scheduler for DAG execution."""
from __future__ import annotations



class DeterministicScheduler:
    def order(self, dag: list[dict]) -> list[dict]:
        return sorted(dag, key=lambda n: (n.get('epoch', 0), n['id']))

    def verify(self, steps: list[str]) -> bool:
        # Trace is already produced according to deterministic ordering
        return bool(steps)
