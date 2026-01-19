"""Tests for quantum verification and rollback."""

import pytest

try:
    from quasim.hybrid_quantum.verification import (
        QuantumVerifier,
        TopologicalDiagnosticObserver,
        VerificationResult,
        VerificationStatus,
    )

    VERIFICATION_AVAILABLE = True
except ImportError:
    VERIFICATION_AVAILABLE = False

try:
    from quasim.hybrid_quantum.rollback import (
        Checkpoint,
        CheckpointStatus,
        DualApprovalGate,
        RollbackManager,
    )

    ROLLBACK_AVAILABLE = True
except ImportError:
    ROLLBACK_AVAILABLE = False


@pytest.mark.skipif(not VERIFICATION_AVAILABLE, reason="Verification module not available")
class TestQuantumVerifier:
    """Test QuantumVerifier class."""

    def test_verifier_creation(self):
        """Test creating verifier with default parameters."""
        verifier = QuantumVerifier()
        assert verifier.min_shots == 1000
        assert verifier.max_noise_threshold == 0.2

    def test_verifier_custom_params(self):
        """Test creating verifier with custom parameters."""
        verifier = QuantumVerifier(
            min_shots=500,
            max_entropy_deviation=0.3,
            max_noise_threshold=0.1,
        )
        assert verifier.min_shots == 500
        assert verifier.max_noise_threshold == 0.1

    def test_verify_sufficient_shots(self):
        """Test verification passes with sufficient shots."""
        verifier = QuantumVerifier(min_shots=100)

        # Create mock result with sufficient shots
        class MockResult:
            counts = {"00": 500, "11": 500}

        result = verifier.verify(MockResult())

        assert result.statistical_validity is True
        assert result.status != VerificationStatus.FAILED

    def test_verify_insufficient_shots(self):
        """Test verification warns on insufficient shots."""
        verifier = QuantumVerifier(min_shots=1000)

        class MockResult:
            counts = {"00": 50, "11": 50}

        result = verifier.verify(MockResult())

        assert result.statistical_validity is False
        assert "Insufficient shots" in result.notes[0]

    def test_verify_with_expected_distribution(self):
        """Test verification with expected distribution."""
        verifier = QuantumVerifier(min_shots=100)

        class MockResult:
            counts = {"00": 500, "11": 500}

        expected = {"00": 0.5, "11": 0.5}
        result = verifier.verify(MockResult(), expected_distribution=expected)

        assert "entropy_analysis" in result.check_results
        assert result.check_results["entropy_analysis"]["passed"] is True

    def test_verify_with_reference_results(self):
        """Test consistency check with reference results."""
        verifier = QuantumVerifier(min_shots=100, min_consistency_score=0.8)

        class MockResult:
            counts = {"00": 500, "11": 500}

        reference = [
            {"00": 480, "11": 520},
            {"00": 510, "11": 490},
        ]

        result = verifier.verify(MockResult(), reference_results=reference)

        assert "consistency_check" in result.check_results
        assert result.consistency_score > 0.5


@pytest.mark.skipif(not VERIFICATION_AVAILABLE, reason="Verification module not available")
class TestVerificationResult:
    """Test VerificationResult dataclass."""

    def test_result_creation(self):
        """Test creating verification result."""
        result = VerificationResult(
            verification_id="ver-123",
            execution_id="exec-456",
            status=VerificationStatus.PASSED,
        )
        assert result.verification_id == "ver-123"
        assert result.is_approved is True

    def test_is_approved_property(self):
        """Test is_approved property."""
        passed = VerificationResult(
            verification_id="1",
            execution_id="1",
            status=VerificationStatus.PASSED,
        )
        failed = VerificationResult(
            verification_id="2",
            execution_id="2",
            status=VerificationStatus.FAILED,
        )

        assert passed.is_approved is True
        assert failed.is_approved is False


@pytest.mark.skipif(not VERIFICATION_AVAILABLE, reason="Verification module not available")
class TestTopologicalDiagnosticObserver:
    """Test TopologicalDiagnosticObserver class."""

    def test_observer_creation(self):
        """Test creating observer."""
        observer = TopologicalDiagnosticObserver()
        assert len(observer.get_observations()) == 0

    def test_observe_counts(self):
        """Test observing measurement counts."""
        observer = TopologicalDiagnosticObserver()

        counts = {"00": 500, "01": 300, "10": 150, "11": 50}
        diagnostics = observer.observe(counts)

        assert "observation_id" in diagnostics
        assert diagnostics["total_shots"] == 1000
        assert diagnostics["unique_outcomes"] == 4
        assert 0 <= diagnostics["concentration_index"] <= 1
        assert 0 <= diagnostics["sparsity_index"] <= 1

    def test_observer_history(self):
        """Test observation history tracking."""
        observer = TopologicalDiagnosticObserver()

        observer.observe({"00": 500, "11": 500})
        observer.observe({"00": 800, "11": 200})

        observations = observer.get_observations()
        assert len(observations) == 2

    def test_summary_statistics(self):
        """Test summary statistics computation."""
        observer = TopologicalDiagnosticObserver()

        observer.observe({"00": 500, "11": 500})
        observer.observe({"00": 800, "11": 200})

        summary = observer.get_summary_statistics()

        assert summary["count"] == 2
        assert "mean_concentration" in summary
        assert "mean_sparsity" in summary


@pytest.mark.skipif(not ROLLBACK_AVAILABLE, reason="Rollback module not available")
class TestRollbackManager:
    """Test RollbackManager class."""

    def test_manager_creation(self):
        """Test creating rollback manager."""
        manager = RollbackManager()
        assert manager.max_checkpoints == 100

    def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        manager = RollbackManager()

        state = {"value": 42, "data": [1, 2, 3]}
        checkpoint_id = manager.create_checkpoint(state, execution_id="exec-1")

        assert checkpoint_id != ""
        checkpoint = manager.get_checkpoint(checkpoint_id)
        assert checkpoint is not None
        assert checkpoint.state_snapshot == state

    def test_rollback_to_checkpoint(self):
        """Test rolling back to checkpoint."""
        manager = RollbackManager()

        state1 = {"value": 1}
        cp1 = manager.create_checkpoint(state1)

        state2 = {"value": 2}
        cp2 = manager.create_checkpoint(state2)

        # Rollback to first checkpoint
        restored = manager.rollback(cp1)

        assert restored == state1

    def test_rollback_execution(self):
        """Test rolling back by execution ID."""
        manager = RollbackManager()

        state = {"data": "before_quantum"}
        manager.create_checkpoint(state, execution_id="quantum-exec-1")

        restored = manager.rollback_execution("quantum-exec-1")

        assert restored == state

    def test_checkpoint_chain(self):
        """Test checkpoint chain tracking."""
        manager = RollbackManager()

        manager.create_checkpoint({"v": 1})
        manager.create_checkpoint({"v": 2})
        manager.create_checkpoint({"v": 3})

        chain = manager.get_checkpoint_chain()

        assert len(chain) == 3
        assert chain[0].state_snapshot["v"] == 1
        assert chain[2].state_snapshot["v"] == 3

    def test_checkpoint_integrity(self):
        """Test checkpoint integrity verification."""
        manager = RollbackManager()

        state = {"key": "value"}
        checkpoint_id = manager.create_checkpoint(state)

        verification = manager.verify_all_checkpoints()

        assert checkpoint_id in verification
        assert verification[checkpoint_id] is True

    def test_checkpoint_pruning(self):
        """Test old checkpoints are pruned."""
        manager = RollbackManager(max_checkpoints=3)

        for i in range(5):
            manager.create_checkpoint({"v": i})

        chain = manager.get_checkpoint_chain()

        assert len(chain) == 3
        # Oldest checkpoints should be pruned
        assert chain[0].state_snapshot["v"] == 2


@pytest.mark.skipif(not ROLLBACK_AVAILABLE, reason="Rollback module not available")
class TestCheckpoint:
    """Test Checkpoint dataclass."""

    def test_checkpoint_creation(self):
        """Test creating checkpoint."""
        checkpoint = Checkpoint(
            checkpoint_id="cp-123",
            state_snapshot={"key": "value"},
        )
        assert checkpoint.checkpoint_id == "cp-123"
        assert checkpoint.status == CheckpointStatus.ACTIVE
        assert checkpoint.state_hash != ""

    def test_checkpoint_integrity_verification(self):
        """Test checkpoint integrity verification."""
        checkpoint = Checkpoint(
            checkpoint_id="cp-123",
            state_snapshot={"key": "value"},
        )

        assert checkpoint.verify_integrity() is True

        # Tamper with state
        checkpoint.state_snapshot["key"] = "modified"

        assert checkpoint.verify_integrity() is False


@pytest.mark.skipif(not ROLLBACK_AVAILABLE, reason="Rollback module not available")
class TestDualApprovalGate:
    """Test DualApprovalGate class."""

    def test_gate_creation(self):
        """Test creating dual approval gate."""
        gate = DualApprovalGate(required_approvals=2)
        assert gate.required_approvals == 2

    def test_request_approval(self):
        """Test requesting approval."""
        gate = DualApprovalGate()

        request_id = gate.request_approval(
            execution_id="exec-1",
            circuit_info={"depth": 10, "gates": 20},
        )

        assert request_id != ""
        status = gate.get_request_status(request_id)
        assert status["status"] == "pending"

    def test_approve_request(self):
        """Test approving request."""
        gate = DualApprovalGate(required_approvals=2)

        request_id = gate.request_approval("exec-1", {})

        # First approval
        gate.approve(request_id, "approver_1")
        assert gate.is_approved(request_id) is False

        # Second approval
        gate.approve(request_id, "approver_2")
        assert gate.is_approved(request_id) is True

    def test_reject_request(self):
        """Test rejecting request."""
        gate = DualApprovalGate()

        request_id = gate.request_approval("exec-1", {})
        gate.reject(request_id, "rejector_1", reason="Circuit too deep")

        status = gate.get_request_status(request_id)
        assert status["status"] == "rejected"
        assert gate.is_approved(request_id) is False

    def test_duplicate_approval_ignored(self):
        """Test that duplicate approvals from same approver are ignored."""
        gate = DualApprovalGate(required_approvals=2)

        request_id = gate.request_approval("exec-1", {})

        gate.approve(request_id, "approver_1")
        gate.approve(request_id, "approver_1")  # Duplicate

        status = gate.get_request_status(request_id)
        assert len(status["approvals"]) == 1
        assert gate.is_approved(request_id) is False
