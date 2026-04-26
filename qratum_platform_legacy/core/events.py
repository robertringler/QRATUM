"""Execution Event Chain with Merkle Tree Structure.

Cryptographic audit trail for all platform operations using a Merkle tree
to ensure event integrity and enable efficient verification.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional


class EventType(Enum):
    """Types of execution events."""

    INTENT_CREATED = "intent_created"
    CONTRACT_AUTHORIZED = "contract_authorized"
    EXECUTION_STARTED = "execution_started"
    COMPUTATION_STEP = "computation_step"
    VALIDATION_CHECK = "validation_check"
    SAFETY_CHECK = "safety_check"
    RESULT_GENERATED = "result_generated"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"


@dataclass(frozen=True)
class ExecutionEvent:
    """Immutable execution event with cryptographic properties.

    Attributes:
        event_type: Type of event
        vertical: Vertical module that generated the event
        operation: Operation being executed
        timestamp: Event creation time (UTC)
        payload: Event-specific data
        previous_hash: Hash of previous event in chain
        event_hash: SHA-256 hash of this event
    """

    event_type: EventType
    vertical: str
    operation: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    payload: Dict[str, Any] = field(default_factory=dict)
    previous_hash: Optional[str] = None
    event_hash: str = ""

    def compute_hash(self, previous_hash: Optional[str] = None) -> str:
        """Compute SHA-256 hash of event.

        Args:
            previous_hash: Hash of previous event for chaining

        Returns:
            SHA-256 hash as hex string
        """
        data = {
            "event_type": self.event_type.value,
            "vertical": self.vertical,
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "previous_hash": previous_hash or self.previous_hash,
        }
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


class MerkleEventChain:
    """Thread-safe Merkle chain for execution events.

    Maintains a cryptographically linked chain of events with Merkle tree
    properties for efficient verification and tamper detection.

    Attributes:
        events: List of all events in the chain
        current_hash: Hash of the most recent event
    """

    def __init__(self):
        """Initialize empty event chain."""
        self.events: List[ExecutionEvent] = []
        self.current_hash: Optional[str] = None
        self._lock = Lock()
        self._root_hash: Optional[str] = None

    def append(self, event: ExecutionEvent) -> ExecutionEvent:
        """Append event to chain with cryptographic linking.

        Args:
            event: Event to append

        Returns:
            Event with computed hash

        Thread-safe operation.
        """
        with self._lock:
            # Compute hash with previous event's hash
            event_hash = event.compute_hash(self.current_hash)

            # Create new event with computed hash
            chained_event = ExecutionEvent(
                event_type=event.event_type,
                vertical=event.vertical,
                operation=event.operation,
                timestamp=event.timestamp,
                payload=event.payload,
                previous_hash=self.current_hash,
                event_hash=event_hash,
            )

            self.events.append(chained_event)
            self.current_hash = event_hash

            return chained_event

    def verify_chain(self) -> bool:
        """Verify integrity of the entire event chain.

        Returns:
            True if chain is valid, False if tampered

        Verifies that each event's hash correctly chains to the next.
        """
        if not self.events:
            return True

        previous_hash = None
        for event in self.events:
            expected_hash = event.compute_hash(previous_hash)
            if event.event_hash != expected_hash:
                return False
            previous_hash = event.event_hash

        return True

    def get_merkle_root(self) -> str:
        """Compute Merkle root of all events.

        Returns:
            Merkle root hash

        The root hash represents the entire event history in a single hash.
        """
        if not self.events:
            return hashlib.sha256(b"").hexdigest()

        # Build Merkle tree from event hashes
        hashes = [event.event_hash for event in self.events]
        return self._compute_merkle_root(hashes)

    def _compute_merkle_root(self, hashes: List[str]) -> str:
        """Recursively compute Merkle root.

        Args:
            hashes: List of hashes to combine

        Returns:
            Merkle root hash
        """
        if len(hashes) == 0:
            return hashlib.sha256(b"").hexdigest()
        if len(hashes) == 1:
            return hashes[0]

        # Pair up hashes and compute parent hashes
        new_hashes = []
        for i in range(0, len(hashes), 2):
            if i + 1 < len(hashes):
                combined = hashes[i] + hashes[i + 1]
            else:
                combined = hashes[i] + hashes[i]
            parent_hash = hashlib.sha256(combined.encode()).hexdigest()
            new_hashes.append(parent_hash)

        return self._compute_merkle_root(new_hashes)

    def get_events_by_type(self, event_type: EventType) -> List[ExecutionEvent]:
        """Get all events of a specific type.

        Args:
            event_type: Type of events to retrieve

        Returns:
            List of matching events
        """
        return [e for e in self.events if e.event_type == event_type]

    def get_events_by_operation(self, operation: str) -> List[ExecutionEvent]:
        """Get all events for a specific operation.

        Args:
            operation: Operation name to filter by

        Returns:
            List of matching events
        """
        return [e for e in self.events if e.operation == operation]

    def __len__(self) -> int:
        """Return number of events in chain."""
        return len(self.events)

    def __repr__(self) -> str:
        """String representation of chain."""
        return f"MerkleEventChain(events={len(self.events)}, root={self.get_merkle_root()[:8]}...)"
