"""Encode transport feed data into AGI percepts."""

from __future__ import annotations

from typing import Dict

from qnx_agi.perception.feed_bridge import Percept, to_percept


def encode(normalized: Dict[str, object]) -> Percept:
    return to_percept("transport", normalized)
