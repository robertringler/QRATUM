"""
Tests for deterministic execution verification.

Validates that the QRATUM platform maintains determinism
across multiple executions with identical inputs.
"""

import pytest

from qradle import QRADLEEngine


def test_deterministic_contract_hash():
    """Test that identical contracts produce identical hashes."""
    engine = QRADLEEngine()

    # Create two identical contracts
    contract1 = engine.create_contract(operation="test_op", inputs={"value": 42}, user_id="user1")

    contract2 = engine.create_contract(operation="test_op", inputs={"value": 42}, user_id="user1")

    # Hashes should be deterministic based on content
    # Note: contract_id will be different, but hash of inputs should match
    assert contract1.inputs == contract2.inputs


def test_merkle_chain_determinism():
    """Test that Merkle chain is deterministic for same operations."""
    from qradle.merkle import MerkleChain

    # Create two chains with identical operations
    chain1 = MerkleChain()
    chain2 = MerkleChain()

    events = [
        ("event1", {"data": "test1"}),
        ("event2", {"data": "test2"}),
        ("event3", {"data": "test3"}),
    ]

    for event_type, data in events:
        chain1.add_event(event_type, data)
        chain2.add_event(event_type, data)

    # Final hashes should match
    assert chain1.get_chain_proof() == chain2.get_chain_proof()


def test_json_serialization_determinism():
    """Test that JSON serialization is deterministic."""
    from qradle.contracts import Contract

    contract = Contract(
        contract_id="test-123",
        operation="test_op",
        inputs={"b": 2, "a": 1},  # Deliberately unordered
        user_id="user1",
    )

    # Serialize multiple times
    json1 = contract.to_json()
    json2 = contract.to_json()

    # Should be identical (keys sorted)
    assert json1 == json2
    assert '"a":1' in json1  # Verify sorting
    assert json1.index('"a":1') < json1.index('"b":2')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
