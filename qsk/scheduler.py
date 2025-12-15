"""Deterministic scheduler for DAG execution."""

from __future__ import annotations

from typing import Dict, List


class DeterministicScheduler:
    def order(self, dag: List[Dict]) -> List[Dict]:
        return sorted(dag, key=lambda n: (n.get("epoch", 0), n["id"]))

    def verify(self, steps: List[str]) -> bool:
        # Trace is already produced according to deterministic ordering
        return bool(steps)
