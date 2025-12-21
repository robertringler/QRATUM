"""ASI Merkle chain with rollback support."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import hashlib
import json

from qratum_asi.core.events import ASIEvent


@dataclass
class RollbackPoint:
    """Represents a rollback point in the chain."""

    rollback_id: str
    chain_index: int
    timestamp: str
    description: str
    state_snapshot: Dict[str, Any]


@dataclass
class ASIMerkleChain:
    """Merkle chain for ASI events with rollback support.
    
    All ASI operations are logged to an immutable Merkle chain,
    providing full auditability and rollback capability.
    """

    events: List[ASIEvent] = field(default_factory=list)
    rollback_points: List[RollbackPoint] = field(default_factory=list)
    chain_hash: Optional[str] = None

    def append(self, event: ASIEvent) -> None:
        """Append event to chain and update chain hash."""
        self.events.append(event)
        self._update_chain_hash()

    def create_rollback_point(
        self,
        rollback_id: str,
        description: str,
        state_snapshot: Dict[str, Any],
    ) -> RollbackPoint:
        """Create a rollback point at current chain position."""
        from datetime import datetime

        rollback_point = RollbackPoint(
            rollback_id=rollback_id,
            chain_index=len(self.events),
            timestamp=datetime.utcnow().isoformat(),
            description=description,
            state_snapshot=state_snapshot,
        )
        self.rollback_points.append(rollback_point)
        return rollback_point

    def rollback_to(self, rollback_id: str) -> bool:
        """Rollback chain to specified rollback point.
        
        Returns:
            True if rollback successful, False if rollback point not found
        """
        # Find rollback point
        rollback_point = None
        for rp in self.rollback_points:
            if rp.rollback_id == rollback_id:
                rollback_point = rp
                break

        if rollback_point is None:
            return False

        # Truncate events to rollback point
        self.events = self.events[: rollback_point.chain_index]
        self._update_chain_hash()
        return True

    def get_events_since(self, index: int) -> List[ASIEvent]:
        """Get events since specified index."""
        return self.events[index:]

    def verify_integrity(self) -> bool:
        """Verify chain integrity by recomputing hash."""
        # Empty chain is valid
        if not self.events and self.chain_hash is None:
            return True
        expected_hash = self._compute_chain_hash()
        return expected_hash == self.chain_hash

    def _update_chain_hash(self) -> None:
        """Update chain hash based on current events."""
        self.chain_hash = self._compute_chain_hash()

    def _compute_chain_hash(self) -> str:
        """Compute hash of entire chain."""
        if not self.events:
            return hashlib.sha256(b"").hexdigest()

        chain_data = [event.to_dict() for event in self.events]
        serialized = json.dumps(chain_data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def get_chain_length(self) -> int:
        """Get current chain length."""
        return len(self.events)

    def get_latest_event(self) -> Optional[ASIEvent]:
        """Get the latest event in the chain."""
        if not self.events:
            return None
        return self.events[-1]
