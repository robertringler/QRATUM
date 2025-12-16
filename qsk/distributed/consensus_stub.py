"""Deterministic pseudo-consensus protocol."""

from __future__ import annotations

from typing import Dict, List


def propose(values: Dict[str, int]) -> int:
    # Deterministic choose minimum value, breaking ties by node id ordering
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    return ordered[0][1]


def commit(participants: List[str], value: int) -> Dict[str, int]:
    return dict.fromkeys(sorted(participants), value)
