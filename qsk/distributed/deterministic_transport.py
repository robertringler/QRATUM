"""Deterministic transport ensures ordering."""
from __future__ import annotations



class DeterministicTransport:
    def order(self, messages: list[dict]) -> list[dict]:
        return sorted(messages, key=lambda m: (m.get('epoch', 0), m.get('id', 0)))
