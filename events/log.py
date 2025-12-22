"""Global Append-Only Event Log with Causal Chains.

This module implements the deterministic event fabric for QRATUM,
providing append-only event logging with cryptographic chaining.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Event:
    """Immutable event in the global event log.

    Attributes:
        event_id: Unique event identifier (hash)
        event_type: Type of event
        timestamp: ISO 8601 timestamp
        contract_id: Reference to contract
        previous_event_hash: Hash of previous event (causal chain)
        payload: Event payload data
        metadata: Additional metadata
    """

    event_id: str
    event_type: str
    timestamp: str
    contract_id: str
    previous_event_hash: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate event after initialization."""
        if not self.event_id:
            raise ValueError("event_id cannot be empty")
        if not self.event_type:
            raise ValueError("event_type cannot be empty")
        if not self.timestamp:
            raise ValueError("timestamp cannot be empty")
        if not self.contract_id:
            raise ValueError("contract_id cannot be empty")

    def serialize(self) -> dict[str, Any]:
        """Serialize event to deterministic dictionary.

        Returns:
            Dictionary representation with sorted keys
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "contract_id": self.contract_id,
            "previous_event_hash": self.previous_event_hash,
            "payload": self.payload,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize event to deterministic JSON.

        Returns:
            JSON string with sorted keys
        """
        return json.dumps(self.serialize(), sort_keys=True)

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of event.

        Returns:
            Hexadecimal SHA-256 hash
        """
        content = {
            k: v for k, v in self.serialize().items() if k != "event_id"
        }
        json_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def verify_hash(self) -> bool:
        """Verify event_id matches content hash.

        Returns:
            True if hash is valid, False otherwise
        """
        return self.compute_hash() == self.event_id


class EventLog:
    """Global append-only event log with causal chains."""

    def __init__(self) -> None:
        """Initialize empty event log."""
        self._events: list[Event] = []
        self._event_index: dict[str, Event] = {}
        self._contract_events: dict[str, list[Event]] = {}

    def append_event(
        self,
        event_type: str,
        contract_id: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> Event:
        """Append a new event to the log.

        Args:
            event_type: Type of event
            contract_id: Reference to contract
            payload: Event payload data
            metadata: Optional additional metadata

        Returns:
            Created Event

        Raises:
            ValueError: If event creation fails
        """
        # Get previous event hash (causal chain)
        previous_hash = self._get_last_event_hash(contract_id)

        # Create event
        from datetime import timezone
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        content = {
            "event_type": event_type,
            "timestamp": timestamp,
            "contract_id": contract_id,
            "previous_event_hash": previous_hash,
            "payload": payload,
            "metadata": metadata or {},
        }

        # Compute event ID
        json_str = json.dumps(content, sort_keys=True)
        event_id = hashlib.sha256(json_str.encode("utf-8")).hexdigest()

        # Create immutable event
        event = Event(
            event_id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            contract_id=contract_id,
            previous_event_hash=previous_hash,
            payload=payload,
            metadata=metadata or {},
        )

        # Append to log (append-only!)
        self._events.append(event)
        self._event_index[event_id] = event

        # Index by contract
        if contract_id not in self._contract_events:
            self._contract_events[contract_id] = []
        self._contract_events[contract_id].append(event)

        return event

    def get_event(self, event_id: str) -> Event | None:
        """Get event by ID.

        Args:
            event_id: Event identifier

        Returns:
            Event if found, None otherwise
        """
        return self._event_index.get(event_id)

    def get_contract_events(self, contract_id: str) -> list[Event]:
        """Get all events for a contract.

        Args:
            contract_id: Contract identifier

        Returns:
            List of events in chronological order
        """
        return self._contract_events.get(contract_id, [])

    def get_all_events(self) -> list[Event]:
        """Get all events in chronological order.

        Returns:
            List of all events
        """
        return self._events[:]

    def verify_causal_chain(self, contract_id: str) -> bool:
        """Verify causal chain integrity for a contract.

        Args:
            contract_id: Contract identifier

        Returns:
            True if chain is valid, False otherwise
        """
        events = self.get_contract_events(contract_id)
        if not events:
            return True

        # First event should have empty previous hash
        if events[0].previous_event_hash != "":
            return False

        # Verify each subsequent event
        for i in range(1, len(events)):
            prev_event = events[i - 1]
            curr_event = events[i]

            # Verify previous hash matches
            if curr_event.previous_event_hash != prev_event.event_id:
                return False

            # Verify event hash
            if not curr_event.verify_hash():
                return False

        return True

    def get_event_sequence(self, contract_id: str) -> list[str]:
        """Get event type sequence for a contract.

        Args:
            contract_id: Contract identifier

        Returns:
            List of event types in order
        """
        events = self.get_contract_events(contract_id)
        return [e.event_type for e in events]

    def _get_last_event_hash(self, contract_id: str) -> str:
        """Get hash of last event for a contract.

        Args:
            contract_id: Contract identifier

        Returns:
            Event hash, or empty string if no events
        """
        events = self._contract_events.get(contract_id, [])
        if events:
            return events[-1].event_id
        return ""

    def count_events(self, event_type: str | None = None) -> int:
        """Count events, optionally filtered by type.

        Args:
            event_type: Optional event type to filter

        Returns:
            Event count
        """
        if event_type is None:
            return len(self._events)
        return sum(1 for e in self._events if e.event_type == event_type)

    def export_log(self) -> list[dict[str, Any]]:
        """Export entire log as JSON-serializable list.

        Returns:
            List of event dictionaries
        """
        return [e.serialize() for e in self._events]


# Global event log instance
_global_event_log = EventLog()


def get_global_event_log() -> EventLog:
    """Get the global event log instance.

    Returns:
        Global EventLog instance
    """
    return _global_event_log


def log_event(
    event_type: str,
    contract_id: str,
    payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> Event:
    """Log an event to the global event log.

    Args:
        event_type: Type of event
        contract_id: Reference to contract
        payload: Event payload data
        metadata: Optional additional metadata

    Returns:
        Created Event
    """
    return _global_event_log.append_event(event_type, contract_id, payload, metadata)
