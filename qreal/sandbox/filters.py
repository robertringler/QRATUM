"""Whitelist/blacklist filters for feed sandboxing."""

from __future__ import annotations

from typing import Dict, Iterable


class FilterSet:
    def __init__(
        self, whitelist: Iterable[str] | None = None, blacklist: Iterable[str] | None = None
    ) -> None:
        self.whitelist = set(whitelist or [])
        self.blacklist = set(blacklist or [])

    def allow(self, payload: Dict[str, object]) -> bool:
        keys = set(payload.keys())
        if self.whitelist and not keys.issuperset(self.whitelist):
            return False
        if self.blacklist.intersection(keys):
            return False
        return True
