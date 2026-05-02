"""Encode market adapter output into AGI percepts."""

from __future__ import annotations

from qnx_agi.perception.feed_bridge import Percept, to_percept


def encode(normalized: dict[str, object]) -> Percept:
    return to_percept("market", normalized)
