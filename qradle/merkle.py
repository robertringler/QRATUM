"""
Merkle chain implementation for QRADLE audit trails.

Provides tamper-evident logging of all operations with
cryptographic verification of event integrity.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MerkleNode:
    """Single node in the Merkle chain."""

    event_type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=lambda: time.time())
    previous_hash: str = "0" * 64  # Genesis node

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of this node."""
        payload = {
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
        }
        json_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "hash": self.compute_hash(),
        }


class MerkleChain:
    """Append-only Merkle chain for audit trails.
    
    Maintains cryptographic integrity of all events.
    Supports verification of chain integrity.
    """

    def __init__(self):
        self.chain: List[MerkleNode] = []
        self._add_genesis_block()

    def _add_genesis_block(self):
        """Add genesis block to initialize chain."""
        genesis = MerkleNode(
            event_type="genesis",
            data={"message": "QRADLE chain initialized"},
            previous_hash="0" * 64
        )
        self.chain.append(genesis)

    def add_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """Add new event to chain.
        
        Args:
            event_type: Type of event
            data: Event data
            
        Returns:
            Hash of the new node
        """
        previous_hash = self.chain[-1].compute_hash() if self.chain else "0" * 64
        node = MerkleNode(
            event_type=event_type,
            data=data,
            previous_hash=previous_hash
        )
        self.chain.append(node)
        return node.compute_hash()

    def verify_integrity(self) -> bool:
        """Verify integrity of the entire chain.
        
        Returns:
            True if chain is valid, False otherwise
        """
        if not self.chain:
            return False

        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # Verify previous hash matches
            if current.previous_hash != previous.compute_hash():
                return False

        return True

    def get_events(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all events, optionally filtered by type.
        
        Args:
            event_type: Optional filter by event type
            
        Returns:
            List of event dictionaries
        """
        events = [node.to_dict() for node in self.chain]
        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        return events

    def get_chain_proof(self) -> str:
        """Get cryptographic proof of current chain state.
        
        Returns:
            Hash representing the entire chain
        """
        if not self.chain:
            return "0" * 64
        return self.chain[-1].compute_hash()
