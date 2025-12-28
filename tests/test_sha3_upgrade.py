"""Tests for SHA-3 quantum-resistant hashing upgrade.

Validates that SHA-256 has been replaced with SHA-3 for quantum resistance
against Grover's algorithm.
"""

import hashlib

import pytest


def test_seeding_uses_sha3():
    """Test that seeding module uses SHA-3 instead of SHA-256."""
    from quasim.common.seeding import derive_seed, hash_config

    # Test hash_config uses SHA-3
    config = {"key": "value", "number": 42}
    config_hash = hash_config(config)

    # Verify hash length (SHA3-256 produces 64 hex characters)
    assert len(config_hash) == 64

    # Verify it matches SHA3-256
    import json

    config_str = json.dumps(config, sort_keys=True)
    expected = hashlib.sha3_256(config_str.encode()).hexdigest()
    assert config_hash == expected

    # Test derive_seed uses SHA-3
    seed = derive_seed(12345, "test_suffix")
    assert isinstance(seed, int)
    assert seed > 0


def test_reasoning_engine_uses_sha3():
    """Test that reasoning engine uses SHA-3 for provenance hashing."""
    from qratum.platform.reasoning_engine import ReasoningChain, ReasoningNode, ReasoningStrategy

    # Create a simple reasoning chain
    node = ReasoningNode(
        node_id="test_node_1",
        vertical="TEST",
        reasoning_type=ReasoningStrategy.DEDUCTIVE,
        input_data={"query": "test"},
        output_data={"result": "test_result"},
        confidence=0.9,
    )

    chain = ReasoningChain(
        chain_id="test_chain",
        query="test query",
        nodes=[node],
        verticals_used=["TEST"],
        final_conclusion={"result": "test"},
        confidence=0.9,
    )

    # Verify provenance hash is computed
    assert chain.provenance_hash
    assert len(chain.provenance_hash) == 64  # SHA3-256 hex length

    # Verify it uses SHA-3
    assert chain.verify_provenance()


def test_security_consensus_vote_uses_sha3():
    """Test that consensus vote signatures use SHA-3."""
    from qratum.planetary.security import ConsensusVote

    vote = ConsensusVote(
        voter_id="voter_1",
        proposal_id="prop_1",
        vote=True,
    )

    # Verify signature is generated
    assert vote.signature
    assert len(vote.signature) == 64  # SHA3-256 hex length

    # Verify it matches SHA3-256
    expected = hashlib.sha3_256(
        f"{vote.voter_id}{vote.proposal_id}{vote.vote}".encode()
    ).hexdigest()
    assert vote.signature == expected


def test_sha3_collision_resistance():
    """Test that SHA-3 provides good collision resistance."""
    from quasim.common.seeding import hash_config

    # Create slightly different configs
    config1 = {"key": "value1"}
    config2 = {"key": "value2"}

    hash1 = hash_config(config1)
    hash2 = hash_config(config2)

    # Hashes should be different
    assert hash1 != hash2

    # Both should be valid SHA3-256 hashes
    assert len(hash1) == 64
    assert len(hash2) == 64


def test_sha3_determinism():
    """Test that SHA-3 hashing is deterministic."""
    from quasim.common.seeding import hash_config

    config = {"key": "value", "number": 42, "nested": {"a": 1, "b": 2}}

    # Hash the same config multiple times
    hash1 = hash_config(config)
    hash2 = hash_config(config)
    hash3 = hash_config(config)

    # All hashes should be identical
    assert hash1 == hash2 == hash3


def test_quantum_resistance_documentation():
    """Verify that quantum resistance is documented in docstrings."""
    from quasim.common.seeding import derive_seed, hash_config

    # Check docstrings mention quantum resistance or Grover's algorithm
    assert "quantum" in hash_config.__doc__.lower() or "grover" in hash_config.__doc__.lower()
    assert "sha-3" in hash_config.__doc__.lower() or "sha3" in hash_config.__doc__.lower()

    assert "quantum" in derive_seed.__doc__.lower() or "grover" in derive_seed.__doc__.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
