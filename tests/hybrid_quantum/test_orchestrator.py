"""Tests for the Hybrid Quantum Orchestrator."""

import pytest
import time
from unittest.mock import Mock, patch

try:
    from quasim.hybrid_quantum.orchestrator import (
        HybridQuantumOrchestrator,
        TrustMetric,
        FallbackStrategy,
        ExecutionContext,
        OrchestratorStatus,
        ExecutionMode,
        FailureType,
        QuantumVerificationError,
    )
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False


@pytest.mark.skipif(not ORCHESTRATOR_AVAILABLE, reason="Orchestrator module not available")
class TestTrustMetric:
    """Test TrustMetric class and invariant enforcement."""

    def test_trust_metric_creation(self):
        """Test creating trust metric with valid value."""
        metric = TrustMetric(value=1.0)
        assert metric.value == 1.0
        assert metric.is_valid
        assert metric.timestamp != ""

    def test_trust_invariant_enforced(self):
        """Test that ℛ(t) ≥ 0 invariant is enforced on creation."""
        with pytest.raises(ValueError, match="Trust invariant violated"):
            TrustMetric(value=-0.1)

    def test_trust_update_preserves_invariant(self):
        """Test that update enforces trust invariant."""
        metric = TrustMetric(value=1.0)
        
        with pytest.raises(ValueError, match="Trust invariant would be violated"):
            metric.update(-0.5)

    def test_trust_update_tracks_history(self):
        """Test that updates track history for variance computation."""
        metric = TrustMetric(value=1.0)
        
        metric.update(0.95)
        metric.update(0.90)
        metric.update(0.92)
        
        assert len(metric.history) == 3
        assert metric.history[0] == 1.0
        assert metric.value == 0.92

    def test_trust_variance_computation(self):
        """Test variance computation for P1 target."""
        metric = TrustMetric(value=1.0)
        
        # Add stable values
        for v in [0.99, 0.98, 0.99, 0.98, 0.99]:
            metric.update(v)
        
        assert metric.variance >= 0
        # With small changes, variance should be low
        assert metric.variance < 0.01

    def test_p1_target_check(self):
        """Test P1 variance target (≤0.001) check."""
        metric = TrustMetric(value=1.0, variance=0.0005)
        assert metric.meets_p1_target
        
        metric.variance = 0.002
        assert not metric.meets_p1_target


@pytest.mark.skipif(not ORCHESTRATOR_AVAILABLE, reason="Orchestrator module not available")
class TestFallbackStrategy:
    """Test FallbackStrategy class."""

    def test_default_fallback_strategy(self):
        """Test default fallback strategy values."""
        strategy = FallbackStrategy()
        assert strategy.max_retries == 3
        assert strategy.fallback_to_classical is True
        assert strategy.escalate_on_failure is True

    def test_max_retries_bounded(self):
        """Test that max_retries is bounded at 3."""
        strategy = FallbackStrategy(max_retries=10)
        assert strategy.max_retries == 3  # Should be capped

    def test_custom_fallback_function(self):
        """Test setting custom fallback function."""
        def custom_fallback(**kwargs):
            return {"result": "classical"}
        
        strategy = FallbackStrategy(classical_fallback_function=custom_fallback)
        assert strategy.classical_fallback_function is not None


@pytest.mark.skipif(not ORCHESTRATOR_AVAILABLE, reason="Orchestrator module not available")
class TestExecutionContext:
    """Test ExecutionContext class."""

    def test_context_creation(self):
        """Test creating execution context."""
        context = ExecutionContext(execution_id="test-123")
        assert context.execution_id == "test-123"
        assert context.status == OrchestratorStatus.IDLE
        assert context.retry_count == 0
        assert len(context.failures) == 0

    def test_record_failure(self):
        """Test recording failures in context."""
        context = ExecutionContext(execution_id="test-123")
        context.record_failure(FailureType.DECOHERENCE, "Qubit decoherence detected")
        
        assert len(context.failures) == 1
        assert context.failures[0]["type"] == "decoherence"
        assert "Qubit decoherence" in context.failures[0]["details"]

    def test_provenance_tracking(self):
        """Test provenance chain tracking."""
        context = ExecutionContext(execution_id="test-123")
        
        context.add_provenance("hash1")
        context.add_provenance("hash2")
        
        assert len(context.provenance_chain) == 2
        assert context.provenance_chain[0] == "hash1"


@pytest.mark.skipif(not ORCHESTRATOR_AVAILABLE, reason="Orchestrator module not available")
class TestHybridQuantumOrchestrator:
    """Test HybridQuantumOrchestrator class."""

    def test_orchestrator_creation(self):
        """Test creating orchestrator with defaults."""
        orchestrator = HybridQuantumOrchestrator()
        assert orchestrator.require_dual_approval is True
        assert orchestrator.zk_proof_latency_threshold == 5.0

    def test_orchestrator_custom_config(self):
        """Test creating orchestrator with custom config."""
        strategy = FallbackStrategy(max_retries=2)
        orchestrator = HybridQuantumOrchestrator(
            fallback_strategy=strategy,
            require_dual_approval=False,
            zk_proof_latency_threshold=10.0,
        )
        
        assert orchestrator.fallback_strategy.max_retries == 2
        assert orchestrator.require_dual_approval is False
        assert orchestrator.zk_proof_latency_threshold == 10.0

    def test_create_execution_context(self):
        """Test creating execution context via orchestrator."""
        orchestrator = HybridQuantumOrchestrator()
        context = orchestrator.create_execution_context(
            mode=ExecutionMode.HYBRID,
            metadata={"test": "value"},
        )
        
        assert context.execution_id != ""
        assert context.mode == ExecutionMode.HYBRID
        assert context.metadata["test"] == "value"

    def test_execute_hybrid_success(self):
        """Test successful hybrid execution."""
        orchestrator = HybridQuantumOrchestrator(require_dual_approval=False)
        context = orchestrator.create_execution_context()
        
        def quantum_op(**kwargs):
            return {"measurement": {"00": 500, "11": 500}}
        
        result = orchestrator.execute_hybrid(
            context=context,
            quantum_op=quantum_op,
        )
        
        assert result["success"] is True
        assert result["mode_used"] == "quantum"
        assert result["provenance_hash"] != ""
        assert result["trust_metric"] >= 0  # Trust invariant

    def test_execute_hybrid_with_verification(self):
        """Test execution with verification function."""
        orchestrator = HybridQuantumOrchestrator(require_dual_approval=False)
        context = orchestrator.create_execution_context()
        
        def quantum_op(**kwargs):
            return {"measurement": {"00": 500, "11": 500}}
        
        def verify(result):
            return "measurement" in result
        
        result = orchestrator.execute_hybrid(
            context=context,
            quantum_op=quantum_op,
            verification_fn=verify,
        )
        
        assert result["success"] is True

    def test_execute_hybrid_verification_failure(self):
        """Test execution fails verification."""
        orchestrator = HybridQuantumOrchestrator(require_dual_approval=False)
        strategy = FallbackStrategy(fallback_to_classical=False)
        orchestrator.fallback_strategy = strategy
        context = orchestrator.create_execution_context()
        
        def quantum_op(**kwargs):
            return {"bad": "data"}
        
        def verify(result):
            return "measurement" in result
        
        result = orchestrator.execute_hybrid(
            context=context,
            quantum_op=quantum_op,
            verification_fn=verify,
        )
        
        # Should fail since verification failed and no fallback
        assert result["success"] is False
        assert len(result["failures"]) > 0

    def test_execute_hybrid_with_fallback(self):
        """Test fallback to classical on quantum failure."""
        orchestrator = HybridQuantumOrchestrator(require_dual_approval=False)
        context = orchestrator.create_execution_context()
        
        call_count = {"quantum": 0, "classical": 0}
        
        def quantum_op(**kwargs):
            call_count["quantum"] += 1
            raise RuntimeError("Quantum hardware error")
        
        def classical_fallback(**kwargs):
            call_count["classical"] += 1
            return {"classical_result": True}
        
        result = orchestrator.execute_hybrid(
            context=context,
            quantum_op=quantum_op,
            classical_fallback=classical_fallback,
        )
        
        max_retries = orchestrator.fallback_strategy.max_retries
        expected_quantum_calls = max_retries + 1  # Initial attempt + retries
        
        assert result["success"] is True
        assert result["mode_used"] == "fallback"
        assert call_count["quantum"] == expected_quantum_calls  # Initial + retries
        assert call_count["classical"] == 1

    def test_bounded_retry_limit(self):
        """Test that retries are bounded at 3."""
        orchestrator = HybridQuantumOrchestrator(require_dual_approval=False)
        strategy = FallbackStrategy(max_retries=3, fallback_to_classical=False)
        orchestrator.fallback_strategy = strategy
        context = orchestrator.create_execution_context()
        
        call_count = [0]
        
        def quantum_op(**kwargs):
            call_count[0] += 1
            raise RuntimeError("Transient error")
        
        result = orchestrator.execute_hybrid(
            context=context,
            quantum_op=quantum_op,
        )
        
        max_retries = strategy.max_retries
        expected_calls = max_retries + 1  # Initial attempt + retries
        
        assert result["success"] is False
        assert call_count[0] == expected_calls  # Initial + max_retries
        assert result["retry_count"] == max_retries

    def test_dual_approval_proposal_creation(self):
        """Test proposal is created when dual approval required."""
        orchestrator = HybridQuantumOrchestrator(require_dual_approval=True)
        context = orchestrator.create_execution_context()
        
        def quantum_op(**kwargs):
            return {"result": "success"}
        
        result = orchestrator.execute_hybrid(
            context=context,
            quantum_op=quantum_op,
        )
        
        assert result["requires_approval"] is True
        assert "proposal_id" in result
        
        # Check proposal was created
        pending = orchestrator.get_pending_proposals()
        assert len(pending) == 1

    def test_approve_proposal(self):
        """Test proposal approval workflow."""
        orchestrator = HybridQuantumOrchestrator(require_dual_approval=True)
        context = orchestrator.create_execution_context()
        
        def quantum_op(**kwargs):
            return {"result": "success"}
        
        result = orchestrator.execute_hybrid(
            context=context,
            quantum_op=quantum_op,
        )
        
        proposal_id = result["proposal_id"]
        
        # First approval
        approval1 = orchestrator.approve_proposal(proposal_id, "approver_1")
        assert approval1["success"] is True
        assert approval1["is_fully_approved"] is False
        
        # Second approval
        approval2 = orchestrator.approve_proposal(proposal_id, "approver_2")
        assert approval2["success"] is True
        assert approval2["is_fully_approved"] is True

    def test_trust_report_generation(self):
        """Test trust report generation."""
        orchestrator = HybridQuantumOrchestrator(require_dual_approval=False)
        context = orchestrator.create_execution_context()
        
        def quantum_op(**kwargs):
            return {"result": "success"}
        
        orchestrator.execute_hybrid(context=context, quantum_op=quantum_op)
        
        report = orchestrator.get_trust_report()
        
        assert "timestamp" in report
        assert report["invariant_assertion"] == "ℛ(t) ≥ 0"
        assert report["invariant_satisfied"] is True
        assert report["total_executions"] == 1

    def test_failure_classification(self):
        """Test failure type classification."""
        orchestrator = HybridQuantumOrchestrator()
        
        # Test various error strings
        assert orchestrator._classify_failure(
            Exception("decoherence detected")
        ) == FailureType.DECOHERENCE
        
        assert orchestrator._classify_failure(
            Exception("noise threshold exceeded")
        ) == FailureType.NOISE_THRESHOLD_EXCEEDED
        
        assert orchestrator._classify_failure(
            Exception("zk proof verification failed")
        ) == FailureType.ZK_PROOF_FAILURE
        
        assert orchestrator._classify_failure(
            Exception("unknown error")
        ) == FailureType.TRANSIENT

    def test_zk_latency_warning(self):
        """Test warning when zk-proof latency exceeds threshold."""
        orchestrator = HybridQuantumOrchestrator(
            require_dual_approval=False,
            zk_proof_latency_threshold=0.001,  # Very low threshold
        )
        context = orchestrator.create_execution_context()
        
        def slow_op(**kwargs):
            time.sleep(0.01)  # 10ms delay
            return {"result": "success"}
        
        result = orchestrator.execute_hybrid(
            context=context,
            quantum_op=slow_op,
        )
        
        assert result["success"] is True
        assert "warnings" in result
        assert any("P0 target" in w for w in result.get("warnings", []))


@pytest.mark.skipif(not ORCHESTRATOR_AVAILABLE, reason="Orchestrator module not available")
class TestOrchestratorEnums:
    """Test orchestrator enumeration types."""

    def test_orchestrator_status_values(self):
        """Test OrchestratorStatus enum values."""
        assert OrchestratorStatus.IDLE.value == "idle"
        assert OrchestratorStatus.EXECUTING.value == "executing"
        assert OrchestratorStatus.AWAITING_APPROVAL.value == "awaiting_approval"
        assert OrchestratorStatus.COMPLETED.value == "completed"
        assert OrchestratorStatus.FAILED.value == "failed"

    def test_execution_mode_values(self):
        """Test ExecutionMode enum values."""
        assert ExecutionMode.QUANTUM.value == "quantum"
        assert ExecutionMode.CLASSICAL.value == "classical"
        assert ExecutionMode.HYBRID.value == "hybrid"
        assert ExecutionMode.FALLBACK.value == "fallback"

    def test_failure_type_values(self):
        """Test FailureType enum values."""
        assert FailureType.DECOHERENCE.value == "decoherence"
        assert FailureType.NOISE_THRESHOLD_EXCEEDED.value == "noise_threshold_exceeded"
        assert FailureType.ZK_PROOF_FAILURE.value == "zk_proof_failure"
        assert FailureType.TRANSIENT.value == "transient"


@pytest.mark.skipif(not ORCHESTRATOR_AVAILABLE, reason="Orchestrator module not available")
class TestQuantumVerificationError:
    """Test QuantumVerificationError exception."""

    def test_exception_message(self):
        """Test exception message is preserved."""
        try:
            raise QuantumVerificationError("Test error message")
        except QuantumVerificationError as e:
            assert "Test error message" in str(e)
