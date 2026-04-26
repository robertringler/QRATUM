"""Deterministic replay of cluster events."""

from __future__ import annotations

from qsk.distributed.event_log import EventLog


def replay(log: EventLog) -> list[dict[str, object]]:
    return [{"sequence": e.sequence, "kind": e.kind, "payload": e.payload} for e in log.events()]
