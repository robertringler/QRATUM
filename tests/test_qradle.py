"""
Unit tests for QRADLE engine.

Tests deterministic execution, Merkle chain integrity,
rollback capability, and invariants enforcement.
"""

import pytest

from qradle import (
    ContractStatus,
    FatalInvariants,
    QRADLEEngine,
)


class TestQRADLEEngine:
    """Test QRADLE engine functionality."""

    def setup_method(self):
        """Setup test engine."""
        self.engine = QRADLEEngine()
        # Register test operations
        self.engine.register_operation(
            "add", lambda inputs: {"result": inputs.get("a", 0) + inputs.get("b", 0)}
        )
        self.engine.register_operation(
            "multiply", lambda inputs: {"result": inputs.get("a", 1) * inputs.get("b", 1)}
        )

    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        assert self.engine is not None
        assert len(self.engine.merkle_chain.chain) >= 1  # Genesis + init event
        assert self.engine.rollback_manager is not None

    def test_contract_creation(self):
        """Test contract creation."""
        contract = self.engine.create_contract(
            operation="add", inputs={"a": 5, "b": 3}, user_id="test_user"
        )

        assert contract is not None
        assert contract.operation == "add"
        assert contract.inputs == {"a": 5, "b": 3}
        assert contract.user_id == "test_user"

    def test_contract_execution(self):
        """Test contract execution."""
        contract = self.engine.create_contract(
            operation="add", inputs={"a": 5, "b": 3}, user_id="test_user"
        )

        execution = self.engine.execute_contract(contract)

        assert execution.status == ContractStatus.COMPLETED
        assert execution.outputs == {"result": 8}
        assert execution.execution_time >= 0

    def test_deterministic_execution(self):
        """Test that same inputs produce same outputs."""
        contract1 = self.engine.create_contract(
            operation="multiply", inputs={"a": 7, "b": 6}, user_id="test_user"
        )

        contract2 = self.engine.create_contract(
            operation="multiply", inputs={"a": 7, "b": 6}, user_id="test_user"
        )

        result1 = self.engine.execute_contract(contract1)
        result2 = self.engine.execute_contract(contract2)

        assert result1.outputs == result2.outputs
        assert result1.outputs == {"result": 42}

    def test_merkle_chain_integrity(self):
        """Test Merkle chain maintains integrity."""
        # Execute some contracts
        for i in range(5):
            contract = self.engine.create_contract(
                operation="add", inputs={"a": i, "b": i + 1}, user_id="test_user"
            )
            self.engine.execute_contract(contract)

        # Verify integrity
        assert self.engine.merkle_chain.verify_integrity()

    def test_checkpoint_creation(self):
        """Test checkpoint creation."""
        checkpoint = self.engine.create_checkpoint("Test checkpoint")

        assert checkpoint is not None
        assert checkpoint.checkpoint_id is not None
        assert checkpoint.description == "Test checkpoint"
        assert checkpoint.merkle_proof != ""

    def test_rollback_capability(self):
        """Test rollback to checkpoint."""
        # Create checkpoint
        checkpoint = self.engine.create_checkpoint("Before changes")

        # Execute some operations
        contract = self.engine.create_contract(
            operation="add", inputs={"a": 10, "b": 20}, user_id="test_user"
        )
        self.engine.execute_contract(contract)

        # Rollback
        success = self.engine.rollback_to_checkpoint(checkpoint.checkpoint_id)
        assert success

    def test_audit_trail(self):
        """Test audit trail generation."""
        contract = self.engine.create_contract(
            operation="add", inputs={"a": 1, "b": 2}, user_id="test_user"
        )
        self.engine.execute_contract(contract)

        # Get audit trail
        events = self.engine.get_audit_trail()
        assert len(events) > 0

        # Filter by contract
        contract_events = self.engine.get_audit_trail(contract.contract_id)
        assert len(contract_events) >= 2  # Creation + execution

    def test_system_proof(self):
        """Test system proof generation."""
        proof = self.engine.get_system_proof()

        assert "merkle_root" in proof
        assert "chain_length" in proof
        assert "integrity_verified" in proof
        assert proof["integrity_verified"] is True

    def test_invariants_enforcement(self):
        """Test that invariants are enforced."""
        context = {
            "chain": self.engine.merkle_chain,
            "rollback_manager": self.engine.rollback_manager,
        }

        # Should not raise exception
        FatalInvariants.verify_all(context)

    def test_contract_immutability(self):
        """Test that contracts are immutable."""
        contract = self.engine.create_contract(
            operation="add", inputs={"a": 1, "b": 2}, user_id="test_user"
        )

        # Try to modify (should fail with frozen dataclass)
        with pytest.raises(Exception):
            contract.operation = "multiply"

    def test_failed_contract_execution(self):
        """Test handling of failed contract execution."""
        contract = self.engine.create_contract(
            operation="unknown_operation", inputs={}, user_id="test_user"
        )

        execution = self.engine.execute_contract(contract)

        assert execution.status == ContractStatus.FAILED
        assert execution.error is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
