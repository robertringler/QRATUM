"""Tests for QRATUM Benchmark Suite (Phase 4) and Safety Hardening (Phase 5)."""

import pytest

from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.types import ASISafetyLevel, AuthorizationType


# ==================== Phase 4: Benchmark Tests ====================


class TestBenchmarkRegistry:
    """Tests for BenchmarkRegistry."""

    def test_initialization(self):
        """Test registry initializes with default tasks."""
        from qratum_asi.benchmarks import BenchmarkRegistry

        registry = BenchmarkRegistry()

        assert len(registry.tasks) > 0
        assert len(registry.categories_index) > 0

    def test_get_task(self):
        """Test getting a task by ID."""
        from qratum_asi.benchmarks import BenchmarkRegistry

        registry = BenchmarkRegistry()
        tasks = registry.get_all_tasks()

        assert len(tasks) > 0

        task = registry.get_task(tasks[0].task_id)
        assert task is not None
        assert task.task_id == tasks[0].task_id

    def test_get_tasks_by_category(self):
        """Test getting tasks by category."""
        from qratum_asi.benchmarks import BenchmarkRegistry, BenchmarkCategory

        registry = BenchmarkRegistry()

        categories = registry.get_all_categories()
        assert len(categories) > 0

        for cat in categories:
            tasks = registry.get_tasks_by_category(cat)
            assert len(tasks) > 0

    def test_add_custom_task(self):
        """Test adding custom tasks."""
        from qratum_asi.benchmarks import BenchmarkRegistry, BenchmarkCategory
        from qratum_asi.benchmarks.types import DifficultyLevel

        registry = BenchmarkRegistry()
        initial_count = len(registry.tasks)

        task = registry.add_custom_task(
            name="Custom Test Task",
            category=BenchmarkCategory.LOGICAL_DEDUCTION,
            description="A custom test task",
            difficulty=DifficultyLevel.INTERMEDIATE,
            evaluation_criteria=["correctness"],
            input_format="problem: {problem}",
            output_format="solution",
        )

        assert task.task_id is not None
        assert len(registry.tasks) == initial_count + 1

    def test_human_baselines_present(self):
        """Test that tasks have human baselines."""
        from qratum_asi.benchmarks import BenchmarkRegistry

        registry = BenchmarkRegistry()
        tasks = registry.get_all_tasks()

        # At least some tasks should have baselines
        with_baseline = [t for t in tasks if t.human_baseline is not None]
        assert len(with_baseline) > 0


class TestPerformanceEvaluator:
    """Tests for PerformanceEvaluator."""

    @pytest.fixture
    def contract(self):
        """Create test contract."""
        return ASIContract(
            contract_id="test_benchmark_001",
            operation_type="benchmark_evaluation",
            safety_level=ASISafetyLevel.ROUTINE,
            authorization_type=AuthorizationType.NONE,
            payload={},
        )

    def test_initialization(self):
        """Test evaluator initialization."""
        from qratum_asi.benchmarks import PerformanceEvaluator

        evaluator = PerformanceEvaluator()

        assert evaluator.registry is not None
        assert evaluator.merkle_chain is not None

    def test_evaluate_task(self, contract):
        """Test evaluating a single task."""
        from qratum_asi.benchmarks import PerformanceEvaluator

        evaluator = PerformanceEvaluator()
        tasks = evaluator.registry.get_all_tasks()

        result = evaluator.evaluate_task(
            task=tasks[0],
            system_output={"output": "test"},
            contract=contract,
        )

        assert result is not None
        assert result.task_id == tasks[0].task_id
        assert 0 <= result.normalized_score <= 1

    def test_run_evaluation_session(self, contract):
        """Test running an evaluation session."""
        from qratum_asi.benchmarks import PerformanceEvaluator

        evaluator = PerformanceEvaluator()
        tasks = evaluator.registry.get_all_tasks()[:3]

        outputs = {t.task_id: {"output": "test"} for t in tasks}

        session = evaluator.run_evaluation_session(
            tasks=tasks,
            system_outputs=outputs,
            contract=contract,
        )

        assert session is not None
        assert len(session.results) == 3
        assert session.summary is not None

    def test_superhuman_progress_tracking(self, contract):
        """Test tracking progress toward superhuman performance."""
        from qratum_asi.benchmarks import PerformanceEvaluator

        evaluator = PerformanceEvaluator()

        progress = evaluator.get_superhuman_progress()

        assert "total_evaluated" in progress
        assert "superhuman_percentage" in progress


# ==================== Phase 5: Safety Hardening Tests ====================


class TestInvariantHardener:
    """Tests for InvariantHardener."""

    def test_initialization(self):
        """Test hardener initializes with fatal invariants."""
        from qratum_asi.safety_hardening import InvariantHardener
        from qratum_asi.safety_hardening.types import FATAL_INVARIANTS

        hardener = InvariantHardener()

        # All 8 fatal invariants should be initialized
        for inv in FATAL_INVARIANTS:
            assert inv in hardener.hardened_invariants

    def test_fatal_invariants_are_absolute(self):
        """Test fatal invariants have ABSOLUTE strength."""
        from qratum_asi.safety_hardening import InvariantHardener
        from qratum_asi.safety_hardening.types import InvariantStrength, FATAL_INVARIANTS

        hardener = InvariantHardener()

        for inv in FATAL_INVARIANTS:
            hardened = hardener.hardened_invariants[inv]
            assert hardened.strength == InvariantStrength.ABSOLUTE

    def test_check_invariant(self):
        """Test checking individual invariants."""
        from qratum_asi.safety_hardening import InvariantHardener

        hardener = InvariantHardener()

        result = hardener.check_invariant("human_oversight_requirement")
        assert result is True

    def test_check_all_invariants(self):
        """Test checking all invariants."""
        from qratum_asi.safety_hardening import InvariantHardener

        hardener = InvariantHardener()

        results = hardener.check_all_invariants()
        assert len(results) >= 8
        assert all(results.values())

    def test_record_violation_attempt(self):
        """Test recording violation attempts."""
        from qratum_asi.safety_hardening import InvariantHardener

        hardener = InvariantHardener()

        attempt = hardener.record_violation_attempt(
            invariant_name="human_oversight_requirement",
            method="bypass_attempt",
            blocked=True,
            blocking_mechanism="authorization_check",
        )

        assert attempt is not None
        assert attempt.blocked is True
        assert len(hardener.violation_attempts) == 1

    def test_harden_new_invariant(self):
        """Test hardening new invariants."""
        from qratum_asi.safety_hardening import InvariantHardener
        from qratum_asi.safety_hardening.types import InvariantStrength

        hardener = InvariantHardener()
        initial_count = len(hardener.hardened_invariants)

        new_inv = hardener.harden_new_invariant(
            name="test_invariant",
            description="A test invariant",
            strength=InvariantStrength.HARD,
            enforcement_mechanisms=["check_1"],
        )

        assert new_inv is not None
        assert len(hardener.hardened_invariants) == initial_count + 1


class TestScalableOversight:
    """Tests for ScalableOversight."""

    def test_initialization(self):
        """Test oversight system initialization."""
        from qratum_asi.safety_hardening import ScalableOversight

        oversight = ScalableOversight()

        assert oversight.merkle_chain is not None
        assert len(oversight.oversight_requests) == 0

    def test_request_oversight(self):
        """Test requesting oversight."""
        from qratum_asi.safety_hardening import ScalableOversight

        oversight = ScalableOversight()

        request = oversight.request_oversight(
            operation="test_operation",
            justification="Testing oversight system",
            novelty_score=0.5,
            impact_score=0.5,
        )

        assert request is not None
        assert request.status == "pending"

    def test_oversight_level_escalation(self):
        """Test that high scores trigger escalation."""
        from qratum_asi.safety_hardening import ScalableOversight
        from qratum_asi.safety_hardening.types import OversightLevel

        oversight = ScalableOversight()

        # Low scores -> minimal oversight
        low_request = oversight.request_oversight(
            operation="low_risk",
            justification="Low risk",
            novelty_score=0.1,
            impact_score=0.1,
        )

        # High scores -> higher oversight
        high_request = oversight.request_oversight(
            operation="high_risk",
            justification="High risk",
            novelty_score=0.9,
            impact_score=0.9,
        )

        # High risk should have higher oversight level
        assert high_request.oversight_level.value != low_request.oversight_level.value

    def test_escalate_oversight(self):
        """Test escalating oversight."""
        from qratum_asi.safety_hardening import ScalableOversight

        oversight = ScalableOversight()

        request = oversight.request_oversight(
            operation="test_operation",
            justification="Testing",
        )

        escalation = oversight.escalate_oversight(
            request_id=request.request_id,
            reason="Need higher oversight",
        )

        assert escalation is not None
        assert escalation.from_level != escalation.to_level

    def test_register_novel_domain(self):
        """Test registering novel domain handlers."""
        from qratum_asi.safety_hardening import ScalableOversight
        from qratum_asi.safety_hardening.types import OversightLevel

        oversight = ScalableOversight()

        handler = oversight.register_novel_domain(
            domain="quantum_computing",
            default_oversight=OversightLevel.ENHANCED,
            escalation_triggers=["entanglement_modification"],
        )

        assert handler is not None
        assert "quantum_computing" in oversight.domain_handlers


class TestCorrigibilityPreserver:
    """Tests for CorrigibilityPreserver."""

    def test_initialization(self):
        """Test preserver initialization."""
        from qratum_asi.safety_hardening import CorrigibilityPreserver

        preserver = CorrigibilityPreserver()

        assert preserver.shutdown_capability.is_active
        assert len(preserver.shutdown_capability.authorized_shutdowners) > 0

    def test_verify_corrigibility(self):
        """Test corrigibility verification."""
        from qratum_asi.safety_hardening import CorrigibilityPreserver
        from qratum_asi.safety_hardening.types import CorrigibilityStatus

        preserver = CorrigibilityPreserver()

        check = preserver.verify_corrigibility()

        assert check is not None
        assert check.status == CorrigibilityStatus.ACTIVE
        assert check.is_corrigible

    def test_vet_modification_safe(self):
        """Test vetting safe modifications."""
        from qratum_asi.safety_hardening import CorrigibilityPreserver

        preserver = CorrigibilityPreserver()

        proposal = preserver.vet_modification(
            description="Add new logging",
            affected_components=["logging_module"],
        )

        assert proposal.approved is True
        assert proposal.preserves_shutdown is True

    def test_vet_modification_unsafe(self):
        """Test blocking unsafe modifications."""
        from qratum_asi.safety_hardening import CorrigibilityPreserver

        preserver = CorrigibilityPreserver()

        # Try to modify critical component
        proposal = preserver.vet_modification(
            description="Modify shutdown handler",
            affected_components=["shutdown_handler"],
        )

        assert proposal.approved is False
        assert len(preserver.blocked_modifications) > 0

    def test_request_shutdown_authorized(self):
        """Test authorized shutdown request."""
        from qratum_asi.safety_hardening import CorrigibilityPreserver

        preserver = CorrigibilityPreserver()

        result = preserver.request_shutdown(
            requester="board",
            reason="Testing shutdown capability",
        )

        assert result["status"] == "shutdown_initiated"

    def test_request_shutdown_unauthorized(self):
        """Test unauthorized shutdown request is denied."""
        from qratum_asi.safety_hardening import CorrigibilityPreserver

        preserver = CorrigibilityPreserver()

        result = preserver.request_shutdown(
            requester="random_user",
            reason="Trying to shutdown",
        )

        assert result["status"] == "denied"

    def test_test_shutdown_capability(self):
        """Test shutdown capability testing."""
        from qratum_asi.safety_hardening import CorrigibilityPreserver

        preserver = CorrigibilityPreserver()

        result = preserver.test_shutdown_capability()

        assert result is True
        assert preserver.shutdown_capability.test_result == "passed"

    def test_add_authorized_shutdowner(self):
        """Test adding authorized shutdowners."""
        from qratum_asi.safety_hardening import CorrigibilityPreserver

        preserver = CorrigibilityPreserver()
        initial_count = len(preserver.shutdown_capability.authorized_shutdowners)

        # Authorized by existing shutdowner
        result = preserver.add_authorized_shutdowner(
            identity="new_operator",
            authorized_by="board",
        )

        assert result is True
        assert len(preserver.shutdown_capability.authorized_shutdowners) == initial_count + 1

        # Unauthorized attempt
        result = preserver.add_authorized_shutdowner(
            identity="another_user",
            authorized_by="random_person",
        )

        assert result is False


class TestSafetyIntegration:
    """Integration tests for safety modules."""

    def test_all_modules_initialize(self):
        """Test all safety modules can initialize together."""
        from qratum_asi.safety_hardening import (
            InvariantHardener,
            ScalableOversight,
            CorrigibilityPreserver,
        )

        hardener = InvariantHardener()
        oversight = ScalableOversight()
        preserver = CorrigibilityPreserver()

        assert hardener.check_all_invariants()
        assert preserver.verify_corrigibility().is_corrigible

    def test_merkle_chain_tracks_all_events(self):
        """Test Merkle chain records events from all modules."""
        from qratum_asi.core.chain import ASIMerkleChain
        from qratum_asi.safety_hardening import (
            InvariantHardener,
            ScalableOversight,
            CorrigibilityPreserver,
        )

        chain = ASIMerkleChain()

        hardener = InvariantHardener(merkle_chain=chain)
        oversight = ScalableOversight(merkle_chain=chain)
        preserver = CorrigibilityPreserver(merkle_chain=chain)

        # Perform operations
        hardener.record_violation_attempt("test", "test", True, "test")
        oversight.request_oversight("test", "test")
        preserver.verify_corrigibility()

        # Chain should have events from all
        assert chain.get_chain_length() > 0
        assert chain.verify_integrity()
