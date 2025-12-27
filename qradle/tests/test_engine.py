"""
Tests for QRADLE Deterministic Engine

Tests deterministic execution with all invariants enforced.
"""

import pytest

from qradle.core.engine import DeterministicEngine, ExecutionContext
from qradle.core.invariants import InvariantViolation


class TestDeterministicEngine:
    """Test suite for deterministic execution engine."""

    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        engine = DeterministicEngine()
        assert engine.merkle_chain is not None
        assert engine.rollback_manager is not None
        assert engine._execution_count == 0

    def test_successful_execution(self):
        """Test successful contract execution."""
        engine = DeterministicEngine()

        context = ExecutionContext(
            contract_id="test_contract_1",
            parameters={"x": 10, "y": 20},
            timestamp="2025-01-01T00:00:00Z",
            safety_level="ROUTINE",
            authorized=True,
        )

        # Simple deterministic function
        def executor(params):
            return params["x"] + params["y"]

        result = engine.execute_contract(context, executor)

        assert result.success is True
        assert result.output == 30
        assert result.output_hash is not None
        assert result.events_emitted > 0
        assert result.checkpoint_id is not None

    def test_deterministic_output(self):
        """Test that same inputs produce same output hash."""
        engine = DeterministicEngine()

        context = ExecutionContext(
            contract_id="test_determinism",
            parameters={"value": 42},
            timestamp="2025-01-01T00:00:00Z",
            safety_level="ROUTINE",
            authorized=True,
        )

        def executor(params):
            return params["value"] * 2

        result1 = engine.execute_contract(context, executor)

        # Execute same contract again (new engine for clean state)
        engine2 = DeterministicEngine()
        result2 = engine2.execute_contract(context, executor)

        # Output hashes should match (determinism)
        assert result1.output_hash == result2.output_hash
        assert result1.output == result2.output

    def test_human_oversight_enforcement(self):
        """Test that sensitive operations require authorization."""
        engine = DeterministicEngine()

        context = ExecutionContext(
            contract_id="test_sensitive",
            parameters={},
            timestamp="2025-01-01T00:00:00Z",
            safety_level="SENSITIVE",
            authorized=False,  # Not authorized!
        )

        def executor(params):
            return "should not execute"

        with pytest.raises(InvariantViolation):
            engine.execute_contract(context, executor)

    def test_checkpoint_creation(self):
        """Test that checkpoints are created before execution."""
        engine = DeterministicEngine()

        context = ExecutionContext(
            contract_id="test_checkpoint",
            parameters={"data": "test"},
            timestamp="2025-01-01T00:00:00Z",
            safety_level="ROUTINE",
            authorized=True,
        )

        def executor(params):
            return "result"

        result = engine.execute_contract(context, executor, create_checkpoint=True)

        assert result.checkpoint_id is not None
        assert engine.rollback_manager.has_checkpoint(result.checkpoint_id)

    def test_rollback_to_checkpoint(self):
        """Test rolling back to a checkpoint."""
        engine = DeterministicEngine()

        context1 = ExecutionContext(
            contract_id="test_rollback_1",
            parameters={"step": 1},
            timestamp="2025-01-01T00:00:00Z",
            safety_level="ROUTINE",
            authorized=True,
        )

        result1 = engine.execute_contract(context1, lambda p: p["step"])
        checkpoint_id = result1.checkpoint_id

        # Execute another contract
        context2 = ExecutionContext(
            contract_id="test_rollback_2",
            parameters={"step": 2},
            timestamp="2025-01-01T00:01:00Z",
            safety_level="ROUTINE",
            authorized=True,
        )
        engine.execute_contract(context2, lambda p: p["step"])

        # Rollback to first checkpoint
        success = engine.rollback_to_checkpoint(checkpoint_id)
        assert success is True

    def test_execution_proof_generation(self):
        """Test generating execution proof."""
        engine = DeterministicEngine()

        context = ExecutionContext(
            contract_id="test_proof",
            parameters={"value": 100},
            timestamp="2025-01-01T00:00:00Z",
            safety_level="ROUTINE",
            authorized=True,
        )

        result = engine.execute_contract(context, lambda p: p["value"] * 2)

        proof = engine.get_execution_proof("test_proof")
        assert proof is not None
        assert proof["contract_id"] == "test_proof"
        assert "merkle_proof" in proof
        assert "chain_root" in proof

    def test_execution_verification(self):
        """Test verifying an execution was deterministic."""
        engine = DeterministicEngine()

        context = ExecutionContext(
            contract_id="test_verify",
            parameters={"x": 5},
            timestamp="2025-01-01T00:00:00Z",
            safety_level="ROUTINE",
            authorized=True,
        )

        result = engine.execute_contract(context, lambda p: p["x"] ** 2)

        # Verify with correct hash
        verified = engine.verify_execution("test_verify", result.output_hash)
        assert verified is True

        # Verify with wrong hash should fail
        verified_wrong = engine.verify_execution("test_verify", "wrong_hash")
        assert verified_wrong is False

    def test_execution_failure_handling(self):
        """Test that execution failures are handled gracefully."""
        engine = DeterministicEngine()

        context = ExecutionContext(
            contract_id="test_failure",
            parameters={},
            timestamp="2025-01-01T00:00:00Z",
            safety_level="ROUTINE",
            authorized=True,
        )

        def failing_executor(params):
            raise RuntimeError("Intentional failure")

        result = engine.execute_contract(context, failing_executor)

        assert result.success is False
        assert result.error is not None
        assert "Intentional failure" in result.error

    def test_engine_stats(self):
        """Test engine statistics."""
        engine = DeterministicEngine()

        # Execute a few contracts
        for i in range(3):
            context = ExecutionContext(
                contract_id=f"test_stats_{i}",
                parameters={"n": i},
                timestamp="2025-01-01T00:00:00Z",
                safety_level="ROUTINE",
                authorized=True,
            )
            engine.execute_contract(context, lambda p: p["n"])

        stats = engine.get_stats()
        assert stats["total_executions"] == 3
        assert stats["chain_length"] > 0
        assert stats["chain_valid"] is True
        assert "checkpoints" in stats
