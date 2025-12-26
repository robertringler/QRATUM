#!/usr/bin/env python3
"""
Simple test runner for QRADLE

Runs basic smoke tests to verify QRADLE functionality.
"""

import sys

sys.path.insert(0, ".")

from qradle.core.engine import DeterministicEngine, ExecutionContext
from qradle.core.invariants import FatalInvariants, InvariantViolation
from qradle.core.merkle import MerkleChain
from qradle.core.rollback import RollbackManager


def test_invariants():
    """Test fatal invariants enforcement."""
    print("Testing Fatal Invariants...")

    # Test 1: Human oversight
    try:
        FatalInvariants.enforce_human_oversight(
            operation="test", safety_level="SENSITIVE", authorized=False
        )
        print("  ❌ FAILED: Should have raised InvariantViolation")
        return False
    except InvariantViolation:
        print("  ✓ Human oversight enforcement works")

    # Test 2: Merkle integrity
    try:
        FatalInvariants.enforce_merkle_integrity(chain_valid=False, last_hash="test")
        print("  ❌ FAILED: Should have raised InvariantViolation")
        return False
    except InvariantViolation:
        print("  ✓ Merkle integrity enforcement works")

    print("✓ All invariant tests passed\n")
    return True


def test_merkle_chain():
    """Test Merkle chain functionality."""
    print("Testing Merkle Chain...")

    chain = MerkleChain()

    # Add events
    node1 = chain.append({"event": "test1"})
    node2 = chain.append({"event": "test2"})

    # Verify chain
    if not chain.verify_chain_integrity():
        print("  ❌ FAILED: Chain integrity check failed")
        return False
    print("  ✓ Chain integrity verified")

    # Generate proof
    proof = chain.get_proof(1)
    if not chain.verify_proof(proof):
        print("  ❌ FAILED: Proof verification failed")
        return False
    print("  ✓ Proof generation and verification works")

    print("✓ All Merkle chain tests passed\n")
    return True


def test_rollback():
    """Test rollback functionality."""
    print("Testing Rollback Manager...")

    manager = RollbackManager()

    # Create checkpoints
    state1 = {"counter": 1}
    cp1 = manager.create_checkpoint(state1, checkpoint_id="cp1")

    state2 = {"counter": 2}
    cp2 = manager.create_checkpoint(state2, checkpoint_id="cp2")

    # Rollback
    restored = manager.rollback_to("cp1")
    if restored != state1:
        print("  ❌ FAILED: Rollback returned incorrect state")
        return False
    print("  ✓ Checkpoint creation and rollback works")

    # Verify checkpoints
    failed = manager.verify_all_checkpoints()
    if len(failed) > 0:
        print(f"  ❌ FAILED: {len(failed)} checkpoints failed verification")
        return False
    print("  ✓ Checkpoint verification works")

    print("✓ All rollback tests passed\n")
    return True


def test_deterministic_engine():
    """Test deterministic execution engine."""
    print("Testing Deterministic Engine...")

    engine = DeterministicEngine()

    # Execute contract
    context = ExecutionContext(
        contract_id="test_contract",
        parameters={"x": 10, "y": 20},
        timestamp="2025-01-01T00:00:00Z",
        safety_level="ROUTINE",
        authorized=True,
    )

    def add_func(params):
        return params["x"] + params["y"]

    result = engine.execute_contract(context, add_func)

    if not result.success:
        print("  ❌ FAILED: Execution failed")
        return False
    print("  ✓ Contract execution works")

    if result.output != 30:
        print(f"  ❌ FAILED: Expected 30, got {result.output}")
        return False
    print("  ✓ Deterministic output verified")

    if not result.output_hash:
        print("  ❌ FAILED: No output hash generated")
        return False
    print("  ✓ Cryptographic proof generated")

    # Verify execution
    proof = engine.get_execution_proof("test_contract")
    if not proof:
        print("  ❌ FAILED: Could not get execution proof")
        return False
    print("  ✓ Execution proof retrieval works")

    # Check stats
    stats = engine.get_stats()
    if stats["total_executions"] != 1:
        print("  ❌ FAILED: Execution count incorrect")
        return False
    print("  ✓ Engine statistics correct")

    print("✓ All deterministic engine tests passed\n")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("QRADLE Test Suite")
    print("=" * 60 + "\n")

    results = []

    results.append(("Fatal Invariants", test_invariants()))
    results.append(("Merkle Chain", test_merkle_chain()))
    results.append(("Rollback Manager", test_rollback()))
    results.append(("Deterministic Engine", test_deterministic_engine()))

    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} test suites passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test suite(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
