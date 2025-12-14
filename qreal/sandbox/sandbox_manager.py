"""Coordinate deterministic ingestion sessions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from qreal.sandbox.cache import DeterministicCache
from qreal.sandbox.rate_limiter import RateLimiter
from qreal.sandbox.filters import FilterSet


@dataclass
class SandboxSession:
    name: str
    limiter: RateLimiter
    filter_set: FilterSet
    history: list[tuple[int, dict[str, object]]] = field(default_factory=list)

    def record(self, tick: int, payload: dict[str, object]) -> None:
        self.history.append((tick, payload))


class SandboxManager:
    def __init__(self) -> None:
        self.cache = DeterministicCache()
        self.sessions: dict[str, SandboxSession] = {}

    def create_session(self, name: str, limiter: RateLimiter, filter_set: FilterSet) -> SandboxSession:
        session = SandboxSession(name=name, limiter=limiter, filter_set=filter_set)
        self.sessions[name] = session
        return session

    def ingest(self, session_name: str, tick: int, payload: dict[str, object], handler: Callable[[dict[str, object]], object]) -> object:
        session = self.sessions[session_name]
        if not session.limiter.allow(tick):
            return None
        if not session.filter_set.allow(payload):
            raise ValueError("Payload failed filter checks")
        session.record(tick, payload)
        cached = self.cache.get(session_name, tick)
        if cached is not None:
            return cached
        result = handler(payload)
        self.cache.set(session_name, tick, result)
        return result

    def replay(self, session_name: str) -> list[tuple[int, dict[str, object]]]:
        session = self.sessions[session_name]
        return list(session.history)
