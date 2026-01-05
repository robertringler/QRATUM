"""
Tests for QRADLE Merkle Chain

Tests cryptographic chaining and proof generation.
"""

import pytest

from qradle.core.merkle import MerkleChain


class TestMerkleChain:
    """Test suite for Merkle chain functionality."""

    def test_chain_initialization(self):
        """Test chain initializes with genesis block."""
        chain = MerkleChain()
        assert len(chain.nodes) == 1
        assert chain.nodes[0].data.get("type") == "genesis"

    def test_append_to_chain(self):
        """Test appending events to chain."""
        chain = MerkleChain()

        # Append first event
        data1 = {"event": "test1", "value": 42}
        node1 = chain.append(data1)
        assert len(chain.nodes) == 2
        assert node1.data == data1

        # Append second event
        data2 = {"event": "test2", "value": 43}
        node2 = chain.append(data2)
        assert len(chain.nodes) == 3
        assert node2.previous_hash == node1.node_hash

    def test_chain_integrity(self):
        """Test chain integrity verification."""
        chain = MerkleChain()

        # Empty chain should be valid
        assert chain.verify_chain_integrity()

        # Add events
        chain.append({"event": "test1"})
        chain.append({"event": "test2"})
        chain.append({"event": "test3"})

        # Chain should still be valid
        assert chain.verify_chain_integrity()

    def test_node_hash_verification(self):
        """Test node hash verification."""
        chain = MerkleChain()
        node = chain.append({"event": "test", "data": "value"})

        # Node should verify correctly
        assert node.verify()

    def test_proof_generation(self):
        """Test Merkle proof generation."""
        chain = MerkleChain()

        # Add some events
        chain.append({"event": "event1"})
        chain.append({"event": "event2"})
        node3 = chain.append({"event": "event3"})

        # Generate proof for middle node (index 2, since 0 is genesis)
        proof = chain.get_proof(2)
        assert proof.chain_position == 2
        assert proof.event_hash == node3.node_hash
        assert proof.root_hash == chain.get_root_hash()

    def test_proof_verification(self):
        """Test proof verification."""
        chain = MerkleChain()

        chain.append({"event": "event1"})
        chain.append({"event": "event2"})

        # Generate proof
        proof = chain.get_proof(1)

        # Verify proof
        assert chain.verify_proof(proof)

    def test_invalid_proof_index(self):
        """Test proof generation with invalid index."""
        chain = MerkleChain()

        with pytest.raises(ValueError):
            chain.get_proof(10)  # Index out of bounds

    def test_find_node_by_hash(self):
        """Test finding node by hash."""
        chain = MerkleChain()

        node1 = chain.append({"event": "test1"})
        node2 = chain.append({"event": "test2"})

        # Find node by hash
        result = chain.find_node_by_hash(node2.node_hash)
        assert result is not None
        idx, found_node = result
        assert found_node.node_hash == node2.node_hash

    def test_chain_stats(self):
        """Test chain statistics."""
        chain = MerkleChain()
        chain.append({"event": "test1"})
        chain.append({"event": "test2"})

        stats = chain.get_chain_stats()
        assert stats["length"] == 3  # genesis + 2 events
        assert stats["chain_valid"] is True
        assert "root_hash" in stats
        assert "genesis_hash" in stats

    def test_export_chain(self):
        """Test chain export."""
        chain = MerkleChain()
        chain.append({"event": "test1"})
        chain.append({"event": "test2"})

        exported = chain.export_chain()
        assert len(exported) == 3
        assert all("node_hash" in node for node in exported)
        assert all("previous_hash" in node for node in exported)
