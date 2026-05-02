"""Tests for Rollback Proof System."""

import pytest

from contracts.rollback_proof import (
    RollbackOrchestrator,
    RollbackProofGenerator,
    RollbackReason,
    StateSnapshot,
    create_rollback_contract,
)


class TestStateSnapshot:
    """Test suite for state snapshots."""

    def test_snapshot_creation(self):
        """Test creating a state snapshot."""
        snapshot = StateSnapshot(
            snapshot_id="snap_001",
            timestamp="2025-01-01T00:00:00Z",
            state_hash="abc123def456",
            state_data={"key": "value"},
            merkle_root="merkle_root_abc",
            zone="Z1",
        )
        assert snapshot.snapshot_id == "snap_001"
        assert snapshot.zone == "Z1"

    def test_snapshot_verification_valid(self):
        """Test snapshot verification with valid data."""
        import hashlib
        import json

        state_data = {"key": "value"}
        serialized = json.dumps(state_data, sort_keys=True)
        state_hash = hashlib.sha256(serialized.encode()).hexdigest()

        snapshot = StateSnapshot(
            snapshot_id="snap_001",
            timestamp="2025-01-01T00:00:00Z",
            state_hash=state_hash,
            state_data=state_data,
            merkle_root="merkle_root",
        )
        assert snapshot.verify() is True

    def test_snapshot_verification_invalid(self):
        """Test snapshot verification with tampered data."""
        snapshot = StateSnapshot(
            snapshot_id="snap_001",
            timestamp="2025-01-01T00:00:00Z",
            state_hash="wrong_hash",
            state_data={"key": "value"},
            merkle_root="merkle_root",
        )
        assert snapshot.verify() is False

    def test_snapshot_serialization(self):
        """Test snapshot serialization."""
        snapshot = StateSnapshot(
            snapshot_id="snap_001",
            timestamp="2025-01-01T00:00:00Z",
            state_hash="abc123",
            state_data={"key": "value"},
            merkle_root="merkle_root",
            zone="Z0",
        )
        serialized = snapshot.serialize()
        assert serialized["snapshot_id"] == "snap_001"
        assert serialized["zone"] == "Z0"


class TestRollbackProofGenerator:
    """Test suite for rollback proof generator."""

    def test_generator_creation(self):
        """Test creating a proof generator."""
        generator = RollbackProofGenerator()
        assert generator is not None

    def test_create_snapshot(self):
        """Test creating a snapshot via generator."""
        generator = RollbackProofGenerator()
        snapshot = generator.create_snapshot(
            state_data={"state": "data"},
            merkle_root="merkle_abc",
            zone="Z1",
        )
        assert snapshot.snapshot_id.startswith("snap_")
        assert snapshot.zone == "Z1"
        assert snapshot.verify() is True

    def test_generate_proof(self):
        """Test generating a rollback proof."""
        generator = RollbackProofGenerator()

        source = generator.create_snapshot(
            state_data={"version": 2},
            merkle_root="root_v2",
        )
        target = generator.create_snapshot(
            state_data={"version": 1},
            merkle_root="root_v1",
        )

        proof = generator.generate_proof(
            source_snapshot=source,
            target_snapshot=target,
            reason=RollbackReason.USER_INITIATED,
            authorized_by="admin_001",
        )

        assert proof.proof_id.startswith("proof_")
        assert proof.rollback_reason == RollbackReason.USER_INITIATED
        assert proof.authorized_by == "admin_001"

    def test_proof_verification(self):
        """Test proof verification."""
        generator = RollbackProofGenerator()

        source = generator.create_snapshot(
            state_data={"v": 2},
            merkle_root="r2",
        )
        target = generator.create_snapshot(
            state_data={"v": 1},
            merkle_root="r1",
        )

        proof = generator.generate_proof(
            source_snapshot=source,
            target_snapshot=target,
            reason=RollbackReason.EXECUTION_FAILURE,
            authorized_by="system",
        )

        assert proof.verify() is True

    def test_custody_log(self):
        """Test chain of custody log."""
        generator = RollbackProofGenerator()

        source = generator.create_snapshot({"v": 2}, "r2")
        target = generator.create_snapshot({"v": 1}, "r1")

        generator.generate_proof(source, target, RollbackReason.USER_INITIATED, "admin")

        log = generator.get_custody_log()
        assert len(log) > 0
        assert log[-1]["authorized_by"] == "admin"


class TestRollbackProof:
    """Test suite for rollback proofs."""

    def test_proof_serialization(self):
        """Test proof serialization."""
        generator = RollbackProofGenerator()

        source = generator.create_snapshot({"v": 2}, "r2")
        target = generator.create_snapshot({"v": 1}, "r1")

        proof = generator.generate_proof(
            source, target, RollbackReason.INVARIANT_VIOLATION, "system"
        )

        serialized = proof.serialize()
        assert serialized["rollback_reason"] == "invariant_violation"
        assert "proof_signature" in serialized
        assert serialized["verified"] is True


class TestRollbackContract:
    """Test suite for rollback contracts."""

    def test_contract_creation(self):
        """Test creating a rollback contract."""
        generator = RollbackProofGenerator()

        source = generator.create_snapshot({"v": 2}, "r2")
        target = generator.create_snapshot({"v": 1}, "r1")

        proof = generator.generate_proof(source, target, RollbackReason.EXECUTION_TIMEOUT, "system")

        contract = create_rollback_contract(
            rollback_proof=proof,
            source_contract_id="contract_001",
            affected_contracts=["contract_002", "contract_003"],
            rollback_depth=2,
            zone="Z1",
        )

        assert contract.contract_type == "RollbackContract"
        assert contract.rollback_depth == 2
        assert len(contract.affected_contracts) == 2

    def test_contract_proof_verification(self):
        """Test contract proof verification."""
        generator = RollbackProofGenerator()

        source = generator.create_snapshot({"v": 2}, "r2")
        target = generator.create_snapshot({"v": 1}, "r1")

        proof = generator.generate_proof(
            source, target, RollbackReason.SECURITY_INCIDENT, "security_admin"
        )

        contract = create_rollback_contract(
            rollback_proof=proof,
            source_contract_id="contract_001",
            affected_contracts=[],
        )

        assert contract.verify_proof() is True

    def test_contract_serialization(self):
        """Test contract serialization."""
        generator = RollbackProofGenerator()

        source = generator.create_snapshot({"v": 2}, "r2")
        target = generator.create_snapshot({"v": 1}, "r1")

        proof = generator.generate_proof(
            source, target, RollbackReason.COMPLIANCE_REQUIREMENT, "compliance_officer"
        )

        contract = create_rollback_contract(
            rollback_proof=proof,
            source_contract_id="contract_001",
            affected_contracts=["contract_002"],
            zone="Z2",
        )

        serialized = contract.serialize()
        assert serialized["zone_classification"] == "Z2"
        assert "rollback_proof" in serialized


class TestRollbackOrchestrator:
    """Test suite for rollback orchestrator."""

    def test_orchestrator_creation(self):
        """Test creating an orchestrator."""
        orchestrator = RollbackOrchestrator()
        assert orchestrator is not None

    def test_create_checkpoint(self):
        """Test creating checkpoints."""
        orchestrator = RollbackOrchestrator()

        snapshot = orchestrator.create_checkpoint(
            state_data={"checkpoint": 1},
            merkle_root="merkle_1",
            zone="Z1",
        )

        assert snapshot.zone == "Z1"
        assert orchestrator.get_snapshot(snapshot.snapshot_id) is not None

    def test_execute_rollback(self):
        """Test executing a rollback."""
        orchestrator = RollbackOrchestrator()

        # Create checkpoints
        snap1 = orchestrator.create_checkpoint({"version": 1}, "root_1", "Z0")
        orchestrator.create_checkpoint({"version": 2}, "root_2", "Z0")

        # Rollback to first checkpoint
        contract = orchestrator.execute_rollback(
            target_snapshot_id=snap1.snapshot_id,
            reason=RollbackReason.USER_INITIATED,
            authorized_by="admin",
            source_contract_id="contract_failing",
            affected_contracts=["contract_a", "contract_b"],
        )

        assert contract.rollback_depth >= 1
        assert contract.verify_proof() is True

    def test_execute_rollback_invalid_snapshot(self):
        """Test rollback with invalid snapshot raises error."""
        orchestrator = RollbackOrchestrator()

        with pytest.raises(ValueError) as exc_info:
            orchestrator.execute_rollback(
                target_snapshot_id="nonexistent",
                reason=RollbackReason.USER_INITIATED,
                authorized_by="admin",
                source_contract_id="contract_001",
            )
        assert "not found" in str(exc_info.value)

    def test_list_snapshots(self):
        """Test listing snapshots."""
        orchestrator = RollbackOrchestrator()

        orchestrator.create_checkpoint({"v": 1}, "r1")
        orchestrator.create_checkpoint({"v": 2}, "r2")

        snapshots = orchestrator.list_snapshots()
        assert len(snapshots) == 2

    def test_rollback_history(self):
        """Test getting rollback history."""
        orchestrator = RollbackOrchestrator()

        snap1 = orchestrator.create_checkpoint({"v": 1}, "r1")
        orchestrator.create_checkpoint({"v": 2}, "r2")

        orchestrator.execute_rollback(
            snap1.snapshot_id,
            RollbackReason.EXECUTION_FAILURE,
            "system",
            "contract_001",
        )

        history = orchestrator.get_rollback_history()
        assert len(history) == 1

    def test_verify_all_proofs(self):
        """Test verifying all proofs."""
        orchestrator = RollbackOrchestrator()

        snap1 = orchestrator.create_checkpoint({"v": 1}, "r1")
        orchestrator.create_checkpoint({"v": 2}, "r2")

        orchestrator.execute_rollback(
            snap1.snapshot_id,
            RollbackReason.SYSTEM_RECOVERY,
            "system",
            "contract_001",
        )

        results = orchestrator.verify_all_proofs()
        assert all(results.values())


class TestRollbackReason:
    """Test suite for rollback reasons."""

    def test_all_reasons_defined(self):
        """Test all rollback reasons are defined."""
        reasons = [
            RollbackReason.INVARIANT_VIOLATION,
            RollbackReason.EXECUTION_TIMEOUT,
            RollbackReason.EXECUTION_FAILURE,
            RollbackReason.USER_INITIATED,
            RollbackReason.SYSTEM_RECOVERY,
            RollbackReason.SECURITY_INCIDENT,
            RollbackReason.COMPLIANCE_REQUIREMENT,
        ]
        assert len(reasons) == 7
        for reason in reasons:
            assert reason.value is not None
