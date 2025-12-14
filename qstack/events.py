"""Deterministic event bus for Q-Stack orchestration."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping


class EventType(str, Enum):
    """Supported event categories within the Q-Stack ecosystem."""

    SYSTEM_BOOT = "SYSTEM_BOOT"
    QNX_CYCLE_STARTED = "QNX_CYCLE_STARTED"
    QNX_CYCLE_COMPLETED = "QNX_CYCLE_COMPLETED"
    QUASIM_SIMULATION_RUN = "QUASIM_SIMULATION_RUN"
    QUNIMBUS_EVAL_COMPLETED = "QUNIMBUS_EVAL_COMPLETED"
    SCENARIO_STARTED = "SCENARIO_STARTED"
    SCENARIO_ENDED = "SCENARIO_ENDED"
    NODE_SCORED = "NODE_SCORED"
    ERROR_RAISED = "ERROR_RAISED"
    ALIGNMENT_PRE_CHECK_FAILED = "ALIGNMENT_PRE_CHECK_FAILED"
    ALIGNMENT_POST_CHECK_VIOLATION = "ALIGNMENT_POST_CHECK_VIOLATION"


@dataclass(frozen=True)
class Event:
    """Immutable event record with deterministic representation."""

    type: EventType
    payload: Mapping[str, Any]
    index: int
    event_id: str

    def __repr__(self) -> str:  # pragma: no cover - deterministic formatting
        return f"Event(type={self.type.value}, id={self.event_id}, index={self.index})"


@dataclass
class EventBus:
    """Pure, deterministic event bus with ordered subscribers."""

    timestamp_seed: str | int = "0"
    events: list[Event] = field(default_factory=list)
    subscribers: list[Callable[[Event], None]] = field(default_factory=list)

    def _normalize_payload(self, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        if payload is None:
            return {}
        return dict(payload)

    def _serialize_payload(self, payload: Mapping[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, default=str)

    def _compute_event_id(self, index: int, payload: Mapping[str, Any]) -> str:
        seed_str = str(self.timestamp_seed)
        payload_repr = self._serialize_payload(payload)
        hashed = hashlib.sha256(f"{seed_str}:{index}:{payload_repr}".encode("utf-8"))
        return hashed.hexdigest()

    def subscribe(self, subscriber: Callable[[Event], None]) -> None:
        if subscriber not in self.subscribers:
            self.subscribers.append(subscriber)

    def publish(self, event_type: EventType, payload: Mapping[str, Any] | None = None) -> Event:
        normalized = self._normalize_payload(payload)
        index = len(self.events)
        event_id = self._compute_event_id(index, normalized)
        event = Event(type=event_type, payload=normalized, index=index, event_id=event_id)
        self.events.append(event)
        for subscriber in list(self.subscribers):
            subscriber(event)
        return event

    def extend(self, events: Iterable[Event]) -> None:
        for event in events:
            # re-emit to keep index continuity deterministically
            self.publish(event.type, dict(event.payload))

    def __repr__(self) -> str:  # pragma: no cover - deterministic formatting
        return f"EventBus(seed={self.timestamp_seed}, events={len(self.events)})"
