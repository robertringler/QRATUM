"""Deterministic transport ensures ordering."""

from __future__ import annotations

from typing import List


class DeterministicTransport:
    def order(self, messages: List[dict]) -> List[dict]:
        return sorted(messages, key=lambda m: (m.get("epoch", 0), m.get("id", 0)))
