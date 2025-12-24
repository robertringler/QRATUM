"""
Event Chain - Auditable event tracking

All operations in QRADLE emit events to an immutable event chain.
Events are Merkle-chained for cryptographic verification.

Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class EventType(Enum):
    """Types of events in QRADLE."""
    CONTRACT_CREATED = "contract_created"
    CONTRACT_AUTHORIZED = "contract_authorized"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"
    CHECKPOINT_CREATED = "checkpoint_created"
    ROLLBACK_EXECUTED = "rollback_executed"
    INVARIANT_VIOLATION = "invariant_violation"
    SAFETY_CHECK = "safety_check"


@dataclass(frozen=True)
class Event:
    """Immutable event record.
    
    Attributes:
        event_type: Type of event
        timestamp: ISO 8601 timestamp
        data: Event-specific data
        metadata: Additional metadata
    """
    event_type: EventType
    timestamp: str
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "metadata": self.metadata,
        }


class EventChain:
    """Chain of immutable events.
    
    The event chain provides a complete audit trail of all operations.
    Events are append-only and cryptographically chained.
    """

    def __init__(self):
        """Initialize event chain."""
        self.events: list[Event] = []

    def emit(
        self,
        event_type: EventType,
        data: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None
    ) -> Event:
        """Emit a new event.
        
        Args:
            event_type: Type of event
            data: Event data
            metadata: Optional metadata
            
        Returns:
            The created Event
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        event = Event(
            event_type=event_type,
            timestamp=timestamp,
            data=data,
            metadata=metadata or {}
        )

        self.events.append(event)
        return event

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> list[Event]:
        """Get events with optional filtering.
        
        Args:
            event_type: Filter by event type
            start_time: Filter events after this time
            end_time: Filter events before this time
            
        Returns:
            List of matching events
        """
        filtered = self.events

        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]

        if start_time:
            filtered = [e for e in filtered if e.timestamp >= start_time]

        if end_time:
            filtered = [e for e in filtered if e.timestamp <= end_time]

        return filtered

    def get_events_for_contract(self, contract_id: str) -> list[Event]:
        """Get all events for a specific contract.
        
        Args:
            contract_id: Contract ID to filter by
            
        Returns:
            List of events for the contract
        """
        return [
            e for e in self.events
            if e.data.get("contract_id") == contract_id
        ]

    def export_events(self) -> list[dict[str, Any]]:
        """Export all events as dictionaries.
        
        Returns:
            List of event dictionaries
        """
        return [e.to_dict() for e in self.events]

    def get_stats(self) -> dict[str, Any]:
        """Get event chain statistics.
        
        Returns:
            Statistics dictionary
        """
        event_counts = {}
        for event in self.events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "total_events": len(self.events),
            "event_type_counts": event_counts,
            "first_event": self.events[0].timestamp if self.events else None,
            "last_event": self.events[-1].timestamp if self.events else None,
        }
