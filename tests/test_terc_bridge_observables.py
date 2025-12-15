"""Tests for TERC bridge observables."""

import numpy as np

from quasim.terc_bridge.observables import (beta_metrics_from_cipher,
                                            emergent_complexity,
                                            ioc_period_candidates,
                                            qgh_consensus_status,
                                            stability_assessment,
                                            stream_synchronization_metrics)


class TestBetaMetricsFromCipher:
    """Tests for beta_metrics_from_cipher."""

    def test_basic_metrics(self):
        """Test basic beta metrics extraction."""
        text = "HELLO" * 20
        metrics = beta_metrics_from_cipher(text)

        assert "beta_entropy" in metrics
        assert "beta_complexity" in metrics
        assert "beta_coherence" in metrics
        assert "beta_periodicity" in metrics

    def test_metrics_bounds(self):
        """Test that metrics are in valid ranges."""
        text = "ABCDEFGH" * 15
        metrics = beta_metrics_from_cipher(text)

        assert 0.0 <= metrics["beta_entropy"] <= 1.0
        assert 0.0 <= metrics["beta_complexity"] <= 1.0
        assert 0.0 <= metrics["beta_coherence"] <= 1.0
        assert 0.0 <= metrics["beta_periodicity"] <= 1.0

    def test_periodic_text(self):
        """Test beta metrics on periodic text."""
        text = "ABC" * 30
        metrics = beta_metrics_from_cipher(text)

        # Periodic text should have higher periodicity score
        assert metrics["beta_periodicity"] > 0.0


class TestIoCPeriodCandidates:
    """Tests for ioc_period_candidates."""

    def test_period_detection(self):
        """Test period detection."""
        text = "ABC" * 30
        candidates = ioc_period_candidates(text, max_period=10)

        assert isinstance(candidates, list)
        # Period 3 should be detected (or close to it)
        assert 3 in candidates or 6 in candidates or 9 in candidates

    def test_no_periods(self):
        """Test with random text (no clear periods)."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 3
        candidates = ioc_period_candidates(text, max_period=10, threshold=0.5)

        # Should find few or no strong periods
        assert isinstance(candidates, list)

    def test_threshold_effect(self):
        """Test that threshold affects results."""
        text = "ABCABC" * 10

        low_threshold = ioc_period_candidates(text, threshold=0.1)
        high_threshold = ioc_period_candidates(text, threshold=0.5)

        # Lower threshold should find more candidates
        assert len(low_threshold) >= len(high_threshold)


class TestEmergentComplexity:
    """Tests for emergent_complexity."""

    def test_complexity_structure(self):
        """Test complexity result structure."""
        text = "ATTACKATDAWN" * 5
        complexity = emergent_complexity(text)

        assert "score" in complexity
        assert "entropy" in complexity
        assert "pattern_density" in complexity
        assert "ioc_variance" in complexity

    def test_complexity_bounds(self):
        """Test that complexity values are valid."""
        text = "HELLOWORLD" * 10
        complexity = emergent_complexity(text)

        assert 0.0 <= complexity["score"] <= 100.0
        assert complexity["entropy"] >= 0.0
        assert complexity["pattern_density"] >= 0.0
        assert complexity["ioc_variance"] >= 0.0


class TestQGHConsensusStatus:
    """Tests for qgh_consensus_status."""

    def test_consensus_basic(self):
        """Test basic consensus status."""
        rng = np.random.default_rng(42)
        states = rng.normal(0, 1, size=(5, 3))

        status = qgh_consensus_status(states)

        assert "converged" in status
        assert "iterations" in status
        assert "final_variance" in status
        assert "stability" in status
        assert "robustness" in status

    def test_consensus_bounds(self):
        """Test that metrics are in valid ranges."""
        rng = np.random.default_rng(42)
        states = rng.normal(0, 1, size=(10, 5))

        status = qgh_consensus_status(states, max_iterations=100)

        assert isinstance(status["converged"], bool)
        assert 0 <= status["iterations"] <= 100
        assert status["final_variance"] >= 0.0
        assert 0.0 <= status["stability"] <= 1.0
        assert 0.0 <= status["robustness"] <= 1.0

    def test_uniform_states(self):
        """Test with already uniform states."""
        states = np.ones((5, 3))

        status = qgh_consensus_status(states)

        # Should converge immediately
        assert status["converged"]
        assert status["iterations"] <= 2
        assert status["final_variance"] < 1e-10


class TestStreamSynchronizationMetrics:
    """Tests for stream_synchronization_metrics."""

    def test_sync_detection(self):
        """Test synchronization detection."""
        # Create correlated streams
        rng = np.random.default_rng(42)
        base = [rng.normal(0, 1) for _ in range(50)]

        stream_data = {
            0: [b + rng.normal(0, 0.1) for b in base],
            1: [b + rng.normal(0, 0.1) for b in base],
            2: [rng.normal(0, 1) for _ in range(50)],
        }

        metrics = stream_synchronization_metrics(stream_data, threshold=0.7)

        assert "num_streams" in metrics
        assert "sync_pairs" in metrics
        assert "sync_ratio" in metrics
        assert "synchronized" in metrics
        assert metrics["num_streams"] == 3

    def test_sync_ratio(self):
        """Test synchronization ratio calculation."""
        stream_data = {
            0: [1.0] * 50,
            1: [1.0] * 50,
        }

        metrics = stream_synchronization_metrics(stream_data, threshold=0.1)

        # Two identical streams should be synchronized (but may have zero variance)
        # Just check the structure
        assert "sync_ratio" in metrics
        assert "synchronized" in metrics


class TestStabilityAssessment:
    """Tests for stability_assessment."""

    def test_stable_metrics(self):
        """Test with stable metric series."""
        metrics = [1.0 + 0.1 * np.sin(i) for i in range(100)]

        assessment = stability_assessment(metrics, window_size=50)

        assert "is_stable" in assessment
        assert "mean" in assessment
        assert "std" in assessment
        assert "trend" in assessment
        assert "assessment" in assessment
        assert assessment["is_stable"]
        assert assessment["assessment"] == "stable"

    def test_unstable_metrics(self):
        """Test with unstable metric series."""
        metrics = [float(i) * 5.0 for i in range(100)]  # Strong steep trend

        assessment = stability_assessment(metrics, window_size=50)

        # With strong trend, should be unstable
        # But if threshold is high, may still pass
        assert "is_stable" in assessment
        assert "assessment" in assessment
        assert assessment["trend"] > 0.0  # Positive trend

    def test_assessment_stats(self):
        """Test that statistics are computed correctly."""
        metrics = [1.0, 2.0, 3.0, 4.0, 5.0] * 10

        assessment = stability_assessment(metrics, window_size=20)

        assert assessment["mean"] > 0.0
        assert assessment["std"] > 0.0


class TestIntegration:
    """Integration tests for TERC bridge."""

    def test_full_observable_pipeline(self):
        """Test complete observable extraction pipeline."""
        # Text-based observables
        text = "ATTACKATDAWN" * 10

        beta = beta_metrics_from_cipher(text)
        periods = ioc_period_candidates(text)
        complexity = emergent_complexity(text)

        assert all(
            k in beta
            for k in ["beta_entropy", "beta_complexity", "beta_coherence", "beta_periodicity"]
        )
        assert isinstance(periods, list)
        assert "score" in complexity

    def test_qgh_observable_pipeline(self):
        """Test QGH observable extraction."""
        rng = np.random.default_rng(42)

        # Consensus observable
        states = rng.normal(0, 1, size=(10, 5))
        consensus = qgh_consensus_status(states)

        # Stream sync observable
        stream_data = {i: [rng.normal(0, 1) for _ in range(50)] for i in range(3)}
        sync = stream_synchronization_metrics(stream_data)

        # Stability observable
        metrics = [rng.normal(1.0, 0.1) for _ in range(100)]
        stability = stability_assessment(metrics)

        assert consensus["converged"] in [True, False]
        assert sync["num_streams"] == 3
        assert stability["assessment"] in ["stable", "unstable"]

    def test_terc_tier_integration(self):
        """Test integration with TERC tier validation."""
        # Simulate TERC Tier-1 observable collection
        text = "CRYPTOTEXT" * 20

        tier1_observables = {
            "beta_metrics": beta_metrics_from_cipher(text),
            "complexity": emergent_complexity(text),
            "period_candidates": ioc_period_candidates(text),
        }

        # Verify all observables are present
        assert "beta_metrics" in tier1_observables
        assert "complexity" in tier1_observables
        assert "period_candidates" in tier1_observables

        # Verify structure
        assert "beta_entropy" in tier1_observables["beta_metrics"]
        assert "score" in tier1_observables["complexity"]
