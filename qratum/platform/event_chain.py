"""
QRATUM Merkle Event Chain

Thread-safe, cryptographically-verified append-only event log.
Maintains integrity through Merkle tree structure where each event
is linked to the previous via hash chaining.
"""

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .core import Event


@dataclass
class MerkleNode:
    """
    Node in the Merkle event chain.

    Each node contains an event and a hash that depends on:
    1. The event content
    2. The previous node's hash (chaining)

    This creates a tamper-evident chain where any modification
    to a past event invalidates all subsequent hashes.
    """

    event: Event
    node_hash: str
    previous_hash: str
    index: int

    def verify(self, previous_hash: str) -> bool:
        """Verify node integrity"""
        expected_hash = self._compute_hash(self.event, previous_hash, self.index)
        return self.node_hash == expected_hash and self.previous_hash == previous_hash

    @staticmethod
    def _compute_hash(event: Event, previous_hash: str, index: int) -> str:
        """Compute deterministic hash for a node"""
        content = {
            "event": event.to_dict(),
            "previous_hash": previous_hash,
            "index": index,
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()


class MerkleEventChain:
    """
    Thread-safe Merkle chain for storing execution events.

    Features:
    - Append-only (no deletions or modifications)
    - Cryptographic integrity (hash chaining)
    - Thread-safe (lock-protected operations)
    - Deterministic replay (ordered event sequence)
    - Tamper detection (verify_integrity)

    This is the foundation for QRATUM's auditability and determinism.
    """

    def __init__(self):
        """Initialize empty event chain"""
        self._chain: List[MerkleNode] = []
        self._lock = threading.RLock()
        self._genesis_hash = "0" * 64  # Genesis hash for first node

    def append(self, event: Event) -> str:
        """
        Append an event to the chain.

        Thread-safe operation that creates a new MerkleNode linked
        to the previous node via hash chaining.

        Args:
            event: Event to append

        Returns:
            Hash of the newly created node

        Raises:
            RuntimeError: If chain integrity is compromised
        """
        with self._lock:
            index = len(self._chain)
            previous_hash = self._genesis_hash if index == 0 else self._chain[-1].node_hash

            node_hash = MerkleNode._compute_hash(event, previous_hash, index)
            node = MerkleNode(
                event=event,
                node_hash=node_hash,
                previous_hash=previous_hash,
                index=index,
            )

            self._chain.append(node)
            return node_hash

    def get_events(self, contract_id: Optional[str] = None) -> List[Event]:
        """
        Retrieve events from the chain.

        Args:
            contract_id: If provided, filter events by contract_id

        Returns:
            List of events (chronologically ordered)
        """
        with self._lock:
            events = [node.event for node in self._chain]

            if contract_id:
                events = [e for e in events if e.contract_id == contract_id]

            return events

    def verify_integrity(self) -> bool:
        """
        Verify the cryptographic integrity of the entire chain.

        This checks that:
        1. Each node's hash is correctly computed
        2. Each node is properly linked to its predecessor
        3. No tampering has occurred

        Returns:
            True if chain is valid, False if compromised
        """
        with self._lock:
            if not self._chain:
                return True

            # Verify genesis node
            if self._chain[0].previous_hash != self._genesis_hash:
                return False

            if not self._chain[0].verify(self._genesis_hash):
                return False

            # Verify all subsequent nodes
            for i in range(1, len(self._chain)):
                prev_hash = self._chain[i - 1].node_hash
                if not self._chain[i].verify(prev_hash):
                    return False

            return True

    def get_chain_state(self) -> Dict[str, Any]:
        """
        Get current state of the chain for inspection.

        Returns:
            Dictionary with chain metadata
        """
        with self._lock:
            return {
                "length": len(self._chain),
                "genesis_hash": self._genesis_hash,
                "latest_hash": self._chain[-1].node_hash if self._chain else None,
                "integrity_verified": self.verify_integrity(),
            }

    def replay_events(self, contract_id: str) -> List[Dict[str, Any]]:
        """
        Replay all events for a specific contract in chronological order.

        This supports deterministic replay - the ability to reconstruct
        the exact execution path from the event log.

        Args:
            contract_id: Contract to replay

        Returns:
            List of event dictionaries in execution order
        """
        events = self.get_events(contract_id)
        return [event.to_dict() for event in events]

    def __len__(self) -> int:
        """Return number of events in chain"""
        with self._lock:
            return len(self._chain)

    def __repr__(self) -> str:
        """String representation"""
        with self._lock:
            return (
                f"MerkleEventChain(length={len(self._chain)}, verified={self.verify_integrity()})"
            )
