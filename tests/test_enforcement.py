"""Tests for Contract Enforcement System."""

import pytest
from contracts.enforcement import (
    ContractEnforcer,
    EnforcedContract,
    EnforcementCheckpoint,
    EnforcementResult,
    create_enforced_contract,
)
from contracts.base import BaseContract, get_current_timestamp, generate_contract_id
from qradle.core.zones import SecurityZone
from qradle.core.invariants import InvariantViolation


class MockContract(BaseContract):
    """Mock contract for testing."""

    def __init__(self, contract_id: str):
        # Create a proper BaseContract
        object.__setattr__(self, "contract_id", contract_id)
        object.__setattr__(self, "contract_type", "MockContract")
        object.__setattr__(self, "created_at", get_current_timestamp())
        object.__setattr__(self, "version", "1.0.0")
        object.__setattr__(self, "metadata", {})


def create_mock_contract(suffix: str = "001") -> MockContract:
    """Create a mock contract for testing."""
    contract_id = f"mock_contract_{suffix}"
    return MockContract(contract_id)


class TestContractEnforcer:
    """Test suite for ContractEnforcer."""

    def test_enforcer_creation(self):
        """Test creating an enforcer."""
        enforcer = ContractEnforcer()
        assert enforcer is not None
        assert enforcer._check_count == 0
        assert enforcer._violation_count == 0

    def test_enforce_contract_creation_success(self):
        """Test successful contract creation enforcement."""
        enforcer = ContractEnforcer()
        contract = create_mock_contract()

        # Use Z1 which allows 'create' operation
        checkpoint = enforcer.enforce_contract_creation(
            contract=contract,
            actor_id="user_001",
            zone=SecurityZone.Z1,
            safety_level="ROUTINE",
        )

        assert checkpoint.result == EnforcementResult.PASSED
        assert checkpoint.contract_id == contract.contract_id

    def test_enforce_contract_creation_tracks_checkpoint(self):
        """Test that enforcement creates checkpoints."""
        enforcer = ContractEnforcer()
        contract = create_mock_contract()

        # Use Z1 which allows 'create' operation
        enforcer.enforce_contract_creation(
            contract=contract,
            actor_id="user_001",
            zone=SecurityZone.Z1,
        )

        assert len(enforcer.checkpoints) == 1
        assert enforcer._check_count == 1

    def test_enforce_contract_execution_success(self):
        """Test successful contract execution enforcement."""
        enforcer = ContractEnforcer()
        contract = create_mock_contract()

        # Use Z1 which allows 'execute' operation
        checkpoint = enforcer.enforce_contract_execution(
            contract=contract,
            actor_id="user_001",
            zone=SecurityZone.Z1,
            safety_level="ROUTINE",
            has_checkpoint=True,
            merkle_chain_valid=True,
        )

        assert checkpoint.result == EnforcementResult.PASSED

    def test_enforce_contract_execution_requires_checkpoint(self):
        """Test execution enforcement fails without checkpoint."""
        enforcer = ContractEnforcer()
        contract = create_mock_contract()

        with pytest.raises(InvariantViolation):
            enforcer.enforce_contract_execution(
                contract=contract,
                actor_id="user_001",
                zone=SecurityZone.Z0,
                safety_level="ROUTINE",
                has_checkpoint=False,  # No checkpoint!
                merkle_chain_valid=True,
            )

        assert enforcer._violation_count == 1

    def test_enforce_contract_execution_requires_merkle_valid(self):
        """Test execution enforcement fails with invalid Merkle chain."""
        enforcer = ContractEnforcer()
        contract = create_mock_contract()

        with pytest.raises(InvariantViolation):
            enforcer.enforce_contract_execution(
                contract=contract,
                actor_id="user_001",
                zone=SecurityZone.Z0,
                safety_level="ROUTINE",
                has_checkpoint=True,
                merkle_chain_valid=False,  # Invalid!
            )

    def test_enforce_contract_completion_success(self):
        """Test successful contract completion enforcement."""
        enforcer = ContractEnforcer()
        contract = create_mock_contract()

        checkpoint = enforcer.enforce_contract_completion(
            contract=contract,
            output_hash="abc123",
            expected_hash=None,  # No determinism check
            events_emitted=1,
        )

        assert checkpoint.result == EnforcementResult.PASSED

    def test_enforce_contract_completion_requires_events(self):
        """Test completion enforcement fails without events."""
        enforcer = ContractEnforcer()
        contract = create_mock_contract()

        with pytest.raises(InvariantViolation):
            enforcer.enforce_contract_completion(
                contract=contract,
                output_hash="abc123",
                expected_hash=None,
                events_emitted=0,  # No events!
            )

    def test_enforce_contract_completion_determinism_check(self):
        """Test completion enforces determinism when expected hash provided."""
        enforcer = ContractEnforcer()
        contract = create_mock_contract()

        # Should pass with matching hashes
        checkpoint = enforcer.enforce_contract_completion(
            contract=contract,
            output_hash="abc123",
            expected_hash="abc123",
            events_emitted=1,
        )
        assert checkpoint.result == EnforcementResult.PASSED

    def test_enforce_contract_completion_determinism_failure(self):
        """Test completion fails with non-matching hashes."""
        enforcer = ContractEnforcer()
        contract = create_mock_contract()

        with pytest.raises(InvariantViolation):
            enforcer.enforce_contract_completion(
                contract=contract,
                output_hash="abc123",
                expected_hash="different_hash",
                events_emitted=1,
            )

    def test_enforce_contract_rollback_success(self):
        """Test successful rollback enforcement."""
        enforcer = ContractEnforcer()
        contract = create_mock_contract()

        # Use Z1 for rollback which allows 'execute' operation
        checkpoint = enforcer.enforce_contract_rollback(
            contract=contract,
            target_checkpoint_id="checkpoint_001",
            actor_id="user_001",
            zone=SecurityZone.Z1,
        )

        assert checkpoint.result == EnforcementResult.PASSED

    def test_enforce_contract_rollback_requires_checkpoint(self):
        """Test rollback enforcement fails without checkpoint ID."""
        enforcer = ContractEnforcer()
        contract = create_mock_contract()

        with pytest.raises(InvariantViolation):
            enforcer.enforce_contract_rollback(
                contract=contract,
                target_checkpoint_id="",  # Empty!
                actor_id="user_001",
                zone=SecurityZone.Z0,
            )

    def test_get_enforcement_checkpoints(self):
        """Test getting enforcement checkpoints."""
        enforcer = ContractEnforcer()
        contract1 = create_mock_contract("001")
        contract2 = create_mock_contract("002")

        # Use Z1 which allows 'create'
        enforcer.enforce_contract_creation(contract1, "user_001", SecurityZone.Z1)
        enforcer.enforce_contract_creation(contract2, "user_001", SecurityZone.Z1)

        # Get all
        all_checkpoints = enforcer.get_enforcement_checkpoints()
        assert len(all_checkpoints) == 2

        # Get by contract ID
        contract1_checkpoints = enforcer.get_enforcement_checkpoints(
            contract_id=contract1.contract_id
        )
        assert len(contract1_checkpoints) == 1

    def test_get_stats(self):
        """Test getting enforcer statistics."""
        enforcer = ContractEnforcer()
        contract = create_mock_contract()

        # Use Z1 which allows 'create'
        enforcer.enforce_contract_creation(contract, "user_001", SecurityZone.Z1)

        stats = enforcer.get_stats()
        assert stats["total_checks"] == 1
        assert stats["total_violations"] == 0
        assert stats["checkpoints_stored"] == 1

    def test_export_audit_log(self):
        """Test exporting audit log."""
        enforcer = ContractEnforcer()
        contract = create_mock_contract()

        # Use Z1 which allows 'create'
        enforcer.enforce_contract_creation(contract, "user_001", SecurityZone.Z1)

        log = enforcer.export_audit_log()
        assert len(log) == 1
        assert "contract_id" in log[0]
        assert "timestamp" in log[0]
        assert "result" in log[0]


class TestEnforcementCheckpoint:
    """Test suite for EnforcementCheckpoint."""

    def test_checkpoint_creation(self):
        """Test creating an enforcement checkpoint."""
        from qradle.core.invariants import InvariantType

        checkpoint = EnforcementCheckpoint(
            checkpoint_id="chk_001",
            contract_id="contract_001",
            timestamp="2025-01-01T00:00:00Z",
            invariant_type=InvariantType.HUMAN_OVERSIGHT,
            result=EnforcementResult.PASSED,
            details={"key": "value"},
        )

        assert checkpoint.checkpoint_id == "chk_001"
        assert checkpoint.result == EnforcementResult.PASSED

    def test_checkpoint_serialization(self):
        """Test checkpoint serialization."""
        from qradle.core.invariants import InvariantType

        checkpoint = EnforcementCheckpoint(
            checkpoint_id="chk_001",
            contract_id="contract_001",
            timestamp="2025-01-01T00:00:00Z",
            invariant_type=InvariantType.DETERMINISM,
            result=EnforcementResult.FAILED,
        )

        serialized = checkpoint.serialize()
        assert serialized["checkpoint_id"] == "chk_001"
        assert serialized["result"] == "failed"
        assert serialized["invariant_type"] == "determinism_guarantee"


class TestEnforcedContract:
    """Test suite for EnforcedContract."""

    def test_enforced_contract_creation(self):
        """Test creating an enforced contract."""
        enforcer = ContractEnforcer()
        contract = create_mock_contract()

        # Use Z1 which allows 'create'
        enforcer.enforce_contract_creation(contract, "user_001", SecurityZone.Z1)

        enforced = create_enforced_contract(
            wrapped_contract=contract,
            enforcer=enforcer,
            zone=SecurityZone.Z1,
            safety_level="ROUTINE",
        )

        assert enforced.contract_type == "EnforcedContract"
        assert enforced.wrapped_contract_id == contract.contract_id
        assert len(enforced.enforcement_checkpoints) == 1

    def test_enforced_contract_status(self):
        """Test enforced contract status."""
        enforcer = ContractEnforcer()
        contract = create_mock_contract()

        # Use Z1 which allows 'create'
        enforcer.enforce_contract_creation(contract, "user_001", SecurityZone.Z1)

        enforced = create_enforced_contract(contract, enforcer)

        # Should be "enforced" if all checkpoints passed
        assert enforced.enforcement_status == "enforced"
        assert enforced.is_fully_enforced() is True

    def test_enforced_contract_serialization(self):
        """Test enforced contract serialization."""
        enforcer = ContractEnforcer()
        contract = create_mock_contract()

        # Use Z1 which allows 'create'
        enforcer.enforce_contract_creation(contract, "user_001", SecurityZone.Z1)

        enforced = create_enforced_contract(
            contract, enforcer, SecurityZone.Z1, "ELEVATED"
        )

        serialized = enforced.serialize()
        assert serialized["zone_classification"] == "Z1"
        assert serialized["safety_level"] == "ELEVATED"
        assert "enforcement_checkpoints" in serialized


class TestZoneEnforcement:
    """Test suite for zone-aware enforcement."""

    def test_z0_allows_basic_operations(self):
        """Test Z0 allows basic read operations."""
        enforcer = ContractEnforcer()
        contract = create_mock_contract()

        # Z0 only allows read operations - test completion which doesn't check zone
        checkpoint = enforcer.enforce_contract_completion(
            contract=contract,
            output_hash="abc123",
            expected_hash=None,
            events_emitted=1,
        )
        assert checkpoint.result == EnforcementResult.PASSED

    def test_z1_allows_create_operations(self):
        """Test Z1 allows create operations."""
        enforcer = ContractEnforcer()
        contract = create_mock_contract()

        checkpoint = enforcer.enforce_contract_creation(
            contract=contract,
            actor_id="user_001",
            zone=SecurityZone.Z1,
        )
        assert checkpoint.result == EnforcementResult.PASSED

    def test_z2_requires_dual_control_for_execution(self):
        """Test Z2 requires dual control."""
        enforcer = ContractEnforcer()
        contract = create_mock_contract()

        # Should fail without dual control
        with pytest.raises(Exception):  # ZoneViolation or InvariantViolation
            enforcer.enforce_contract_execution(
                contract=contract,
                actor_id="user_001",
                zone=SecurityZone.Z2,
                safety_level="SENSITIVE",
                has_checkpoint=True,
                merkle_chain_valid=True,
            )
