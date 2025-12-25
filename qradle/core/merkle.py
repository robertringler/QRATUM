"""
Merkle Chain Implementation for QRADLE

Provides cryptographic chaining of events for complete auditability.
Every event is hashed and linked to previous events, creating a tamper-evident chain.

Version: 1.0.0
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass(frozen=True)
class MerkleNode:
    """A node in the Merkle chain.
    
    Attributes:
        data: Event data
        timestamp: ISO 8601 timestamp
        previous_hash: Hash of previous node
        node_hash: Hash of this node
    """
    data: dict[str, Any]
    timestamp: str
    previous_hash: str
    node_hash: str = field(default="", init=False)

    def __post_init__(self):
        """Compute node hash after initialization."""
        if not self.node_hash:
            computed_hash = self._compute_hash()
            object.__setattr__(self, "node_hash", computed_hash)

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of node content."""
        content = {
            "data": self.data,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
        }
        serialized = json.dumps(content, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def verify(self) -> bool:
        """Verify node hash matches content."""
        expected_hash = self._compute_hash()
        return self.node_hash == expected_hash


@dataclass
class MerkleProof:
    """Cryptographic proof of event in Merkle chain.
    
    Attributes:
        event_id: ID of the event
        event_hash: Hash of the event
        chain_position: Position in chain
        proof_path: Hashes needed to verify inclusion
        root_hash: Root hash of the chain
    """
    event_id: str
    event_hash: str
    chain_position: int
    proof_path: list[str]
    root_hash: str

    def verify(self, claimed_root: str) -> bool:
        """Verify this proof against a claimed root hash."""
        return self.root_hash == claimed_root

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_hash": self.event_hash,
            "chain_position": self.chain_position,
            "proof_path": self.proof_path,
            "root_hash": self.root_hash,
        }


class MerkleChain:
    """Merkle chain for cryptographically linking events.
    
    All events are appended to the chain and cryptographically linked.
    The chain provides tamper-evident audit trails with cryptographic proofs.
    """

    def __init__(self, genesis_data: Optional[dict[str, Any]] = None):
        """Initialize Merkle chain with genesis block.
        
        Args:
            genesis_data: Optional data for genesis block
        """
        self.nodes: list[MerkleNode] = []
        self._genesis_hash = self._create_genesis_hash()

        # Create genesis block
        genesis = genesis_data or {"type": "genesis", "version": "1.0.0"}
        self.append(genesis)

    def _create_genesis_hash(self) -> str:
        """Create hash for genesis block."""
        return hashlib.sha256(b"QRADLE_GENESIS_1.0.0").hexdigest()

    def append(self, data: dict[str, Any]) -> MerkleNode:
        """Append data to the chain.
        
        Args:
            data: Event data to append
            
        Returns:
            The created MerkleNode
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        previous_hash = self.nodes[-1].node_hash if self.nodes else self._genesis_hash

        node = MerkleNode(
            data=data,
            timestamp=timestamp,
            previous_hash=previous_hash
        )

        self.nodes.append(node)
        return node

    def get_chain_head(self) -> Optional[MerkleNode]:
        """Get the most recent node in the chain."""
        return self.nodes[-1] if self.nodes else None

    def get_root_hash(self) -> str:
        """Get the root hash (hash of most recent node)."""
        head = self.get_chain_head()
        return head.node_hash if head else self._genesis_hash

    def verify_chain_integrity(self) -> bool:
        """Verify the integrity of the entire chain.
        
        Returns:
            True if chain is valid, False otherwise
        """
        if not self.nodes:
            return True

        # Verify each node
        for node in self.nodes:
            if not node.verify():
                return False

        # Verify linkage
        expected_prev = self._genesis_hash
        for node in self.nodes:
            if node.previous_hash != expected_prev:
                return False
            expected_prev = node.node_hash

        return True

    def get_proof(self, node_index: int) -> MerkleProof:
        """Generate a Merkle proof for a node at given index.
        
        Args:
            node_index: Index of node to generate proof for
            
        Returns:
            MerkleProof for the node
        """
        if node_index < 0 or node_index >= len(self.nodes):
            raise ValueError(f"Invalid node index: {node_index}")

        node = self.nodes[node_index]

        # For a linear chain, the proof path includes all subsequent hashes
        proof_path = [n.node_hash for n in self.nodes[node_index + 1:]]

        return MerkleProof(
            event_id=node.data.get("event_id", str(node_index)),
            event_hash=node.node_hash,
            chain_position=node_index,
            proof_path=proof_path,
            root_hash=self.get_root_hash()
        )

    def verify_proof(self, proof: MerkleProof) -> bool:
        """Verify a Merkle proof against the current chain.
        
        Args:
            proof: The proof to verify
            
        Returns:
            True if proof is valid, False otherwise
        """
        current_root = self.get_root_hash()
        return proof.verify(current_root)

    def get_node(self, index: int) -> Optional[MerkleNode]:
        """Get node at index.
        
        Args:
            index: Index of node to retrieve
            
        Returns:
            MerkleNode or None if index invalid
        """
        if 0 <= index < len(self.nodes):
            return self.nodes[index]
        return None

    def find_node_by_hash(self, node_hash: str) -> Optional[tuple[int, MerkleNode]]:
        """Find node by its hash.
        
        Args:
            node_hash: Hash to search for
            
        Returns:
            Tuple of (index, node) or None if not found
        """
        for idx, node in enumerate(self.nodes):
            if node.node_hash == node_hash:
                return (idx, node)
        return None

    def get_chain_stats(self) -> dict[str, Any]:
        """Get statistics about the chain.
        
        Returns:
            Dictionary with chain statistics
        """
        return {
            "length": len(self.nodes),
            "root_hash": self.get_root_hash(),
            "genesis_hash": self._genesis_hash,
            "chain_valid": self.verify_chain_integrity(),
            "first_timestamp": self.nodes[0].timestamp if self.nodes else None,
            "last_timestamp": self.nodes[-1].timestamp if self.nodes else None,
        }

    def export_chain(self) -> list[dict[str, Any]]:
        """Export the entire chain as a list of dictionaries.
        
        Returns:
            List of node data
        """
        return [
            {
                "data": node.data,
                "timestamp": node.timestamp,
                "previous_hash": node.previous_hash,
                "node_hash": node.node_hash,
            }
            for node in self.nodes
        ]
