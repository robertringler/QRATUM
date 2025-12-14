"""Deterministic pseudo-consensus protocol."""
from __future__ import annotations



def propose(values: dict[str, int]) -> int:
    # Deterministic choose minimum value, breaking ties by node id ordering
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    return ordered[0][1]


def commit(participants: list[str], value: int) -> dict[str, int]:
    return {node: value for node in sorted(participants)}
