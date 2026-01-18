"""Tests for Self-Discovery Cycle (SDC) Module.

Tests the autonomous discovery loop implementation:
- Hypothesis generation
- State-space model derivation
- Bounded AAS search execution
- Empirical delta evaluation
- Discovery ledger commitment
- Verification agent spawning
"""

import pytest

from qratum.discovery import (
    DiscoveryEntry,
    DiscoveryLedger,
    DiscoveryStatus,
    EmpiricalDelta,
    Hypothesis,
    SearchResult,
    SelfDiscoveryCycle,
    StateSpaceModel,
    StatisticalVerificationAgent,
    create_sdc_for_domain,
)


class TestHypothesis:
    """Tests for Hypothesis dataclass."""

    def test_creation(self):
        """Should create hypothesis with all fields."""
        hyp = Hypothesis(
            description="test hypothesis",
            domain="chess",
            parameters={"param1": "value1"},
            confidence=0.8,
        )

        assert hyp.description == "test hypothesis"
        assert hyp.domain == "chess"
        assert hyp.confidence == 0.8
        assert hyp.hypothesis_id  # Should have UUID

    def test_auto_generated_id(self):
        """Should auto-generate unique IDs."""
        h1 = Hypothesis()
        h2 = Hypothesis()

        assert h1.hypothesis_id != h2.hypothesis_id

    def test_hash_deterministic(self):
        """Hash should be deterministic for same content."""
        h1 = Hypothesis(description="test", parameters={"a": 1})
        h2 = Hypothesis(description="test", parameters={"a": 1})

        assert h1.hash() == h2.hash()

    def test_hash_different_for_different_content(self):
        """Hash should differ for different content."""
        h1 = Hypothesis(description="test1")
        h2 = Hypothesis(description="test2")

        assert h1.hash() != h2.hash()


class TestStateSpaceModel:
    """Tests for StateSpaceModel."""

    def test_creation(self):
        """Should create model with all fields."""
        model = StateSpaceModel(
            hypothesis_id="hyp-123",
            state_dimensions=64,
            action_space_size=100,
            branching_factor=30.0,
        )

        assert model.hypothesis_id == "hyp-123"
        assert model.state_dimensions == 64


class TestEmpiricalDelta:
    """Tests for EmpiricalDelta."""

    def test_exceeds_threshold_positive(self):
        """Should detect when threshold exceeded."""
        delta = EmpiricalDelta(
            baseline_value=1.0,
            actual_value=1.2,
            delta=0.2,
            relative_delta=0.2,  # 20% improvement
            statistical_significance=0.98,  # High confidence
        )

        assert delta.exceeds_threshold

    def test_exceeds_threshold_negative(self):
        """Should not exceed threshold for small improvements."""
        delta = EmpiricalDelta(
            baseline_value=1.0,
            actual_value=1.05,
            delta=0.05,
            relative_delta=0.05,  # 5% improvement (below 10%)
            statistical_significance=0.98,
        )

        assert not delta.exceeds_threshold

    def test_exceeds_threshold_low_confidence(self):
        """Should not exceed threshold for low confidence."""
        delta = EmpiricalDelta(
            baseline_value=1.0,
            actual_value=1.2,
            delta=0.2,
            relative_delta=0.2,
            statistical_significance=0.5,  # Low confidence
        )

        assert not delta.exceeds_threshold


class TestDiscoveryLedger:
    """Tests for DiscoveryLedger."""

    def test_empty_ledger(self):
        """Empty ledger should have zero entries."""
        ledger = DiscoveryLedger()
        assert len(ledger) == 0

    def test_commit_entry(self):
        """Should commit entry and return chain hash."""
        ledger = DiscoveryLedger()

        entry = DiscoveryEntry(
            hypothesis=Hypothesis(description="test"),
            status=DiscoveryStatus.PENDING,
        )

        chain_hash = ledger.commit(entry)

        assert chain_hash
        assert len(ledger) == 1

    def test_chain_integrity(self):
        """Chain should maintain integrity."""
        ledger = DiscoveryLedger()

        for i in range(5):
            entry = DiscoveryEntry(
                hypothesis=Hypothesis(description=f"test-{i}"),
            )
            ledger.commit(entry)

        assert ledger.verify_chain()

    def test_get_verified_discoveries(self):
        """Should return only verified discoveries."""
        ledger = DiscoveryLedger()

        # Add pending entry
        e1 = DiscoveryEntry(status=DiscoveryStatus.PENDING)
        ledger.commit(e1)

        # Add verified entry
        e2 = DiscoveryEntry(status=DiscoveryStatus.VERIFIED)
        ledger.commit(e2)

        verified = ledger.get_verified_discoveries()
        assert len(verified) == 1
        assert verified[0].status == DiscoveryStatus.VERIFIED

    def test_get_pending_verification(self):
        """Should return only pending discoveries."""
        ledger = DiscoveryLedger()

        e1 = DiscoveryEntry(status=DiscoveryStatus.PENDING)
        e2 = DiscoveryEntry(status=DiscoveryStatus.VERIFIED)
        ledger.commit(e1)
        ledger.commit(e2)

        pending = ledger.get_pending_verification()
        assert len(pending) == 1


class TestStatisticalVerificationAgent:
    """Tests for StatisticalVerificationAgent."""

    def test_verify_significant_discovery(self):
        """Should verify statistically significant discovery."""
        agent = StatisticalVerificationAgent(min_samples=5, confidence_threshold=0.9)

        entry = DiscoveryEntry(
            empirical_delta=EmpiricalDelta(
                relative_delta=0.2,
                statistical_significance=0.95,
            )
        )

        assert agent.verify(entry)

    def test_reject_low_significance(self):
        """Should reject low significance discovery."""
        agent = StatisticalVerificationAgent(confidence_threshold=0.95)

        entry = DiscoveryEntry(
            empirical_delta=EmpiricalDelta(
                relative_delta=0.2,
                statistical_significance=0.8,  # Below threshold
            )
        )

        assert not agent.verify(entry)

    def test_reject_negative_delta(self):
        """Should reject negative relative delta."""
        agent = StatisticalVerificationAgent()

        entry = DiscoveryEntry(
            empirical_delta=EmpiricalDelta(
                relative_delta=-0.1,  # Regression
                statistical_significance=0.99,
            )
        )

        assert not agent.verify(entry)

    def test_reject_no_delta(self):
        """Should reject entry without delta."""
        agent = StatisticalVerificationAgent()
        entry = DiscoveryEntry()  # No empirical_delta

        assert not agent.verify(entry)


class TestSelfDiscoveryCycle:
    """Tests for SelfDiscoveryCycle."""

    @pytest.fixture
    def simple_sdc(self):
        """Create a simple SDC for testing."""
        hypothesis_count = 0

        def gen_hypothesis():
            nonlocal hypothesis_count
            hypothesis_count += 1
            return Hypothesis(
                description=f"hypothesis-{hypothesis_count}",
                confidence=0.7,
            )

        def derive_model(hyp):
            return StateSpaceModel(
                hypothesis_id=hyp.hypothesis_id,
                state_dimensions=10,
                action_space_size=5,
            )

        def execute_search(model):
            return SearchResult(
                model_id=model.model_id,
                best_state=None,
                best_value=0.6,
                nodes_explored=100,
                time_elapsed_ms=10.0,
                depth_reached=5,
                confidence=0.9,
            )

        def evaluate_delta(result, model):
            return EmpiricalDelta(
                model_id=model.model_id,
                baseline_value=0.5,
                actual_value=result.best_value,
                delta=0.1,
                relative_delta=0.2,  # 20% improvement
                statistical_significance=0.96,
            )

        return SelfDiscoveryCycle(
            domain="test",
            hypothesis_generator=gen_hypothesis,
            model_deriver=derive_model,
            search_executor=execute_search,
            delta_evaluator=evaluate_delta,
            delta_threshold=0.1,
        )

    def test_run_iteration(self, simple_sdc):
        """Should run single iteration."""
        entry = simple_sdc.run_iteration()

        assert entry is not None
        assert simple_sdc.iteration_count == 1

    def test_run_multiple_iterations(self, simple_sdc):
        """Should run multiple iterations."""
        discoveries = simple_sdc.run(iterations=5)

        assert simple_sdc.iteration_count == 5
        assert len(discoveries) == 5  # All exceed threshold

    def test_ledger_commitment(self, simple_sdc):
        """Should commit discoveries to ledger."""
        simple_sdc.run(iterations=3)

        assert simple_sdc.discovery_count == 3
        assert simple_sdc.ledger.verify_chain()

    def test_below_threshold_not_committed(self):
        """Below threshold discoveries should not be committed."""

        def evaluate_delta_low(result, model):
            return EmpiricalDelta(
                relative_delta=0.05,  # Below 0.1 threshold
                statistical_significance=0.9,
            )

        sdc = SelfDiscoveryCycle(
            domain="test",
            hypothesis_generator=lambda: Hypothesis(),
            model_deriver=lambda h: StateSpaceModel(),
            search_executor=lambda m: SearchResult(
                model_id="",
                best_state=None,
                best_value=0.5,
                nodes_explored=10,
                time_elapsed_ms=1.0,
                depth_reached=1,
                confidence=0.5,
            ),
            delta_evaluator=evaluate_delta_low,
            delta_threshold=0.1,
        )

        entry = sdc.run_iteration()
        assert entry is None  # Not committed
        assert sdc.discovery_count == 0

    def test_verification_agents(self, simple_sdc):
        """Should run verification agents."""
        agent = StatisticalVerificationAgent()
        simple_sdc.add_verification_agent(agent)

        entry = simple_sdc.run_iteration()

        assert entry.status == DiscoveryStatus.VERIFIED
        assert entry.verification_count == 1

    def test_get_metrics(self, simple_sdc):
        """Should return valid metrics."""
        simple_sdc.run(iterations=3)

        metrics = simple_sdc.get_metrics()
        assert metrics.is_valid()

    def test_stop_cycle(self, simple_sdc):
        """Should stop cycle when requested."""
        simple_sdc._running = True
        simple_sdc.stop()
        assert not simple_sdc._running


class TestCreateSDCForDomain:
    """Tests for create_sdc_for_domain factory."""

    def test_create_chess_sdc(self):
        """Should create chess SDC."""
        sdc = create_sdc_for_domain("chess")

        assert sdc.domain == "chess"
        assert sdc.ledger is not None

    def test_chess_sdc_runs(self):
        """Chess SDC should run iterations."""
        sdc = create_sdc_for_domain("chess", max_iterations=3)
        discoveries = sdc.run()

        assert sdc.iteration_count == 3
        # Some may not exceed threshold

    def test_unknown_domain_raises(self):
        """Unknown domain should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown domain"):
            create_sdc_for_domain("unknown")
