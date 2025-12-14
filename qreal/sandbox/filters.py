"""Whitelist/blacklist filters for feed sandboxing."""
from __future__ import annotations

from typing import Iterable


class FilterSet:
    def __init__(self, whitelist: Iterable[str] | None = None, blacklist: Iterable[str] | None = None) -> None:
        self.whitelist = set(whitelist or [])
        self.blacklist = set(blacklist or [])

    def allow(self, payload: dict[str, object]) -> bool:
        keys = set(payload.keys())
        if self.whitelist and not keys.issuperset(self.whitelist):
            return False
        return not self.blacklist.intersection(keys)
