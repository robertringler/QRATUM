"""Fault domain partitioning for deterministic recovery."""

from __future__ import annotations

from typing import List


class FaultDomains:
    def __init__(self, domains: List[str] | None = None):
        self.domains = domains or ["primary"]

    def classify(self, event: str) -> str:
        for domain in self.domains:
            if domain in event:
                return domain
        return self.domains[0]
