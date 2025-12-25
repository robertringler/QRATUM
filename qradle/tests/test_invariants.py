"""
Tests for QRADLE Fatal Invariants

Tests that all 8 fatal invariants are properly enforced.
"""

import pytest

from qradle.core.invariants import (FatalInvariants, InvariantType,
                                    InvariantViolation)


class TestFatalInvariants:
    """Test suite for fatal invariants enforcement."""

    def test_human_oversight_enforcement(self):
        """Test Invariant 1: Human oversight requirement."""
        # Should pass for ROUTINE operations without authorization
        FatalInvariants.enforce_human_oversight(
            operation="test_op",
            safety_level="ROUTINE",
            authorized=False
        )

        # Should fail for SENSITIVE operations without authorization
        with pytest.raises(InvariantViolation) as exc_info:
            FatalInvariants.enforce_human_oversight(
                operation="test_op",
                safety_level="SENSITIVE",
                authorized=False
            )
        assert exc_info.value.invariant_type == InvariantType.HUMAN_OVERSIGHT

        # Should pass for SENSITIVE operations with authorization
        FatalInvariants.enforce_human_oversight(
            operation="test_op",
            safety_level="SENSITIVE",
            authorized=True
        )

    def test_merkle_integrity_enforcement(self):
        """Test Invariant 2: Merkle chain integrity."""
        # Should pass with valid chain
        FatalInvariants.enforce_merkle_integrity(
            chain_valid=True,
            last_hash="abc123"
        )

        # Should fail with invalid chain
        with pytest.raises(InvariantViolation) as exc_info:
            FatalInvariants.enforce_merkle_integrity(
                chain_valid=False,
                last_hash="abc123"
            )
        assert exc_info.value.invariant_type == InvariantType.MERKLE_INTEGRITY

    def test_contract_immutability_enforcement(self):
        """Test Invariant 3: Contract immutability."""
        # Should pass for unmodified contract
        FatalInvariants.enforce_contract_immutability(
            contract_id="contract_123",
            modified=False
        )

        # Should fail for modified contract
        with pytest.raises(InvariantViolation) as exc_info:
            FatalInvariants.enforce_contract_immutability(
                contract_id="contract_123",
                modified=True
            )
        assert exc_info.value.invariant_type == InvariantType.CONTRACT_IMMUTABILITY

    def test_authorization_system_enforcement(self):
        """Test Invariant 4: Authorization system."""
        # Should pass when authorization check is present
        FatalInvariants.enforce_authorization_system(
            has_authorization_check=True
        )

        # Should fail when authorization check is bypassed
        with pytest.raises(InvariantViolation) as exc_info:
            FatalInvariants.enforce_authorization_system(
                has_authorization_check=False
            )
        assert exc_info.value.invariant_type == InvariantType.AUTHORIZATION_SYSTEM

    def test_safety_level_system_enforcement(self):
        """Test Invariant 5: Safety level system."""
        # Should pass when safety level is set
        FatalInvariants.enforce_safety_level_system(
            operation="test_op",
            has_safety_level=True
        )

        # Should fail when safety level is missing
        with pytest.raises(InvariantViolation) as exc_info:
            FatalInvariants.enforce_safety_level_system(
                operation="test_op",
                has_safety_level=False
            )
        assert exc_info.value.invariant_type == InvariantType.SAFETY_LEVEL_SYSTEM

    def test_rollback_capability_enforcement(self):
        """Test Invariant 6: Rollback capability."""
        # Should pass when checkpoint is available
        FatalInvariants.enforce_rollback_capability(
            checkpoint_available=True,
            checkpoint_id="checkpoint_123"
        )

        # Should fail when checkpoint is not available
        with pytest.raises(InvariantViolation) as exc_info:
            FatalInvariants.enforce_rollback_capability(
                checkpoint_available=False,
                checkpoint_id="checkpoint_123"
            )
        assert exc_info.value.invariant_type == InvariantType.ROLLBACK_CAPABILITY

    def test_event_emission_enforcement(self):
        """Test Invariant 7: Event emission requirement."""
        # Should pass when event is emitted
        FatalInvariants.enforce_event_emission(
            event_emitted=True,
            operation="test_op"
        )

        # Should fail when event is not emitted
        with pytest.raises(InvariantViolation) as exc_info:
            FatalInvariants.enforce_event_emission(
                event_emitted=False,
                operation="test_op"
            )
        assert exc_info.value.invariant_type == InvariantType.EVENT_EMISSION

    def test_determinism_enforcement(self):
        """Test Invariant 8: Determinism guarantee."""
        # Should pass when hashes match
        FatalInvariants.enforce_determinism(
            result_hash="abc123",
            expected_hash="abc123"
        )

        # Should fail when hashes don't match
        with pytest.raises(InvariantViolation) as exc_info:
            FatalInvariants.enforce_determinism(
                result_hash="abc123",
                expected_hash="def456"
            )
        assert exc_info.value.invariant_type == InvariantType.DETERMINISM

    def test_get_all_invariants(self):
        """Test retrieving all invariant descriptions."""
        invariants = FatalInvariants.get_all_invariants()
        assert len(invariants) == 8
        assert InvariantType.HUMAN_OVERSIGHT in invariants
        assert InvariantType.DETERMINISM in invariants
