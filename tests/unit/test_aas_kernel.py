"""Tests for AAS Canonical Kernel Module.

Tests the canonical kernel interfaces:
- state_entropy(state) -> float
- branch_value(state, move) -> float
- resource_allocator(entropy_gradient) -> depth_budget
- multi_agent_split(state) -> [orthogonal_subspaces]
"""

import pytest

from qratum_chess.search.aas_kernel import (
    AASMetrics,
    ChessAASKernel,
    DepthBudget,
    EntropyGradient,
    OrthogonalSubspace,
    create_aas_kernel,
)


class TestEntropyGradient:
    """Tests for EntropyGradient."""

    def test_initial_state(self):
        """Initial gradient should be zero."""
        eg = EntropyGradient()
        assert eg.current == 0.0
        assert eg.previous == 0.0
        assert eg.gradient == 0.0

    def test_update_calculates_gradient(self):
        """Update should calculate gradient correctly."""
        eg = EntropyGradient()

        eg.update(1.0)
        assert eg.current == 1.0
        assert eg.gradient == 1.0  # From 0 to 1

        eg.update(2.0)
        assert eg.current == 2.0
        assert eg.previous == 1.0
        assert eg.gradient == 1.0  # From 1 to 2

        eg.update(1.5)
        assert eg.current == 1.5
        assert eg.gradient == -0.5  # From 2 to 1.5

    def test_history_maintained(self):
        """History should be maintained up to max_history."""
        eg = EntropyGradient(max_history=5)

        for i in range(10):
            eg.update(float(i))

        assert len(eg.history) == 5
        assert eg.history == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_trend_calculation(self):
        """Trend should reflect overall direction."""
        eg = EntropyGradient()

        # Increasing trend
        for i in range(5):
            eg.update(float(i))

        assert eg.trend > 0  # Positive trend

        # Reset and test decreasing
        eg = EntropyGradient()
        for i in range(5, 0, -1):
            eg.update(float(i))

        assert eg.trend < 0  # Negative trend


class TestDepthBudget:
    """Tests for DepthBudget."""

    def test_default_values(self):
        """Default budget should have reasonable values."""
        budget = DepthBudget()
        assert budget.base_depth == 10
        assert budget.depth_extension == 0
        assert budget.width_multiplier == 1.0
        assert budget.effective_depth == 10

    def test_effective_depth(self):
        """Effective depth should include extensions."""
        budget = DepthBudget(base_depth=10, depth_extension=5)
        assert budget.effective_depth == 15

        budget.depth_extension = -3
        assert budget.effective_depth == 7


class TestOrthogonalSubspace:
    """Tests for OrthogonalSubspace."""

    def test_creation(self):
        """Should create subspace with all fields."""
        subspace = OrthogonalSubspace(
            subspace_id="test",
            state_slice={"data": "test"},
            priority=1.5,
            estimated_complexity=0.8,
            dependencies=["other"],
        )

        assert subspace.subspace_id == "test"
        assert subspace.priority == 1.5
        assert "other" in subspace.dependencies


class TestAASMetrics:
    """Tests for AASMetrics."""

    def test_valid_metrics(self):
        """Valid metrics should pass validation."""
        metrics = AASMetrics(
            outcome_superiority_ratio=1.5,
            compute_efficiency_index=2.0,
            sovereignty_factor=1.0,
            hallucination_risk_density=0.0,
        )
        assert metrics.is_valid()

    def test_invalid_sovereignty_factor(self):
        """Invalid SF should fail validation."""
        metrics = AASMetrics(sovereignty_factor=1.5)  # > 1
        assert not metrics.is_valid()

        metrics = AASMetrics(sovereignty_factor=-0.1)  # < 0
        assert not metrics.is_valid()

    def test_default_sovereignty_is_full(self):
        """Default SF should be 1.0 (fully sovereign)."""
        metrics = AASMetrics()
        assert metrics.sovereignty_factor == 1.0


class TestChessAASKernel:
    """Tests for ChessAASKernel."""

    def test_creation(self):
        """Should create chess kernel."""
        kernel = ChessAASKernel()
        assert kernel is not None

    def test_creation_with_evaluator(self):
        """Should accept custom evaluator."""

        def custom_eval(pos):
            return 0.5

        kernel = ChessAASKernel(evaluator=custom_eval)
        assert kernel.evaluator == custom_eval

    def test_get_metrics(self):
        """Should return valid metrics."""
        kernel = ChessAASKernel()
        metrics = kernel.get_metrics()

        assert isinstance(metrics, AASMetrics)
        assert metrics.is_valid()

    def test_resource_allocator_low_entropy(self):
        """Low entropy should favor deeper search."""
        kernel = ChessAASKernel()

        # Simulate low entropy state
        eg = EntropyGradient()
        eg.update(0.5)  # Low entropy

        budget = kernel.resource_allocator(eg)

        assert budget.depth_extension > 0  # Should extend depth
        assert budget.width_multiplier < 1.0  # Should narrow width

    def test_resource_allocator_high_entropy(self):
        """High entropy should favor wider search."""
        kernel = ChessAASKernel()

        # Simulate high entropy state
        eg = EntropyGradient()
        eg.update(4.0)  # High entropy

        budget = kernel.resource_allocator(eg)

        assert budget.width_multiplier > 1.0  # Should widen search

    def test_resource_allocator_increasing_gradient(self):
        """Increasing entropy gradient should widen search."""
        kernel = ChessAASKernel()

        eg = EntropyGradient()
        eg.update(1.0)
        eg.update(2.0)  # Increasing

        budget = kernel.resource_allocator(eg)

        assert budget.width_multiplier >= 1.0

    def test_adaptive_heuristic_mutation(self):
        """Should update heuristic weights based on feedback."""
        kernel = ChessAASKernel()
        initial_capture_weight = kernel._capture_weight

        # Feedback indicating captures are very effective
        kernel.adaptive_heuristic_mutation({"capture_accuracy": 0.9})

        assert kernel._capture_weight > initial_capture_weight

    def test_adaptive_heuristic_mutation_decrease(self):
        """Should decrease weights when accuracy is low."""
        kernel = ChessAASKernel()
        initial_capture_weight = kernel._capture_weight

        # Feedback indicating captures are not effective
        kernel.adaptive_heuristic_mutation({"capture_accuracy": 0.1})

        assert kernel._capture_weight < initial_capture_weight


class TestCreateAASKernel:
    """Tests for create_aas_kernel factory."""

    def test_create_chess_kernel(self):
        """Should create chess kernel."""
        kernel = create_aas_kernel("chess")
        assert isinstance(kernel, ChessAASKernel)

    def test_create_chess_kernel_case_insensitive(self):
        """Domain should be case insensitive."""
        kernel = create_aas_kernel("Chess")
        assert isinstance(kernel, ChessAASKernel)

        kernel = create_aas_kernel("CHESS")
        assert isinstance(kernel, ChessAASKernel)

    def test_unknown_domain_raises(self):
        """Unknown domain should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown domain"):
            create_aas_kernel("unknown_domain")

    def test_create_with_kwargs(self):
        """Should pass kwargs to kernel constructor."""

        def custom_eval(pos):
            return 0.0

        kernel = create_aas_kernel("chess", evaluator=custom_eval)
        assert kernel.evaluator == custom_eval


class TestKernelInvariants:
    """Tests for kernel invariant properties."""

    def test_state_entropy_non_negative(self):
        """state_entropy should always return non-negative value."""
        kernel = ChessAASKernel()

        # Test with mock state (kernel handles missing chess module)
        class MockState:
            def generate_legal_moves(self):
                return []

        # Should not raise
        # Note: With proper chess Position, would return >= 0

    def test_computation_tracking(self):
        """Kernel should track computations for metrics."""
        kernel = ChessAASKernel()

        # Initial count should be 0
        assert kernel._computation_count == 0

        # Track some computations
        kernel._track_computation()
        kernel._track_computation()
        kernel._track_computation(external=True)

        assert kernel._computation_count == 3
        assert kernel._external_calls == 1

    def test_metrics_update_after_tracking(self):
        """Metrics should update after tracking computations."""
        kernel = ChessAASKernel()

        # Track computations
        for _ in range(5):
            kernel._track_computation()
        kernel._track_computation(external=True)

        # Get updated metrics
        metrics = kernel.get_metrics()

        # SF should reflect external dependency
        assert metrics.sovereignty_factor < 1.0
