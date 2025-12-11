"""Simple critic scoring logic."""
from __future__ import annotations

from ..utils.provenance import hash_payload


def critic_score(plan: list[dict[str, object]]) -> float:
    checksum = hash_payload({"plan": plan})
    return float(int(checksum, 16) % 1000) / 1000.0
