"""Tests for QGH non-speculative algorithms."""

import numpy as np

from quasim.qgh.nonspec_algorithms import (
    CausalHistoryHash,
    DistributedStreamMonitor,
    SelfConsistencyPropagator,
    StabilityMonitor,
    SuperpositionResolver,
)


class TestCausalHistoryHash:
    """Tests for CausalHistoryHash."""

    def test_add_event(self):
        """Test adding events to history."""
        chh = CausalHistoryHash(history_size=10)
        data = np.array([1.0, 2.0, 3.0])

        event_hash = chh.add_event("event1", data)

        assert isinstance(event_hash, str)
        assert len(event_hash) > 0
        assert chh.verify_causality("event1")

    def test_verify_missing_event(self):
        """Test verification of missing event."""
        chh = CausalHistoryHash()

        assert not chh.verify_causality("nonexistent")

    def test_history_size_limit(self):
        """Test that history respects size limit."""
        chh = CausalHistoryHash(history_size=5)

        for i in range(10):
            chh.add_event(f"event{i}", np.array([float(i)]))

        history = chh.get_history()
        assert len(history) == 5

        # Oldest events should be removed
        assert not chh.verify_causality("event0")
        assert chh.verify_causality("event9")

    def test_deterministic_hash(self):
        """Test that same data produces same hash."""
        chh = CausalHistoryHash()
        data = np.array([1.0, 2.0, 3.0])

        hash1 = chh.add_event("test", data)
        hash2 = chh.add_event("test", data)

        assert hash1 == hash2

    def test_to_dict_from_dict(self):
        """Test serialization and deserialization."""
        chh = CausalHistoryHash(history_size=10, hash_algo="sha256")
        chh.add_event("event1", np.array([1.0, 2.0]))
        chh.add_event("event2", np.array([3.0, 4.0]))

        data = chh.to_dict()
        restored = CausalHistoryHash.from_dict(data)

        assert restored.history_size == chh.history_size
        assert restored.hash_algo == chh.hash_algo
        assert restored.verify_causality("event1")
        assert restored.verify_causality("event2")


class TestSuperpositionResolver:
    """Tests for SuperpositionResolver."""

    def test_resolve_converges(self):
        """Test that resolution converges."""
        resolver = SuperpositionResolver(max_iterations=100, tolerance=1e-6)
        state = np.array([0.5, 0.3, 0.15, 0.05])

        def normalize(x):
            return x / np.sum(x)

        result = resolver.resolve(state, normalize)

        assert result["converged"]
        assert result["iterations"] < 100
        assert np.abs(np.sum(result["state"]) - 1.0) < 1e-6

    def test_resolve_custom_consistency(self):
        """Test resolution with custom consistency function."""
        resolver = SuperpositionResolver(max_iterations=50)
        state = np.array([1.0, 2.0, 3.0])

        def make_uniform(x):
            return np.ones_like(x) / len(x)

        result = resolver.resolve(state, make_uniform)

        assert result["converged"]
        assert np.allclose(result["state"], [1 / 3, 1 / 3, 1 / 3], atol=1e-5)

    def test_max_iterations(self):
        """Test max iterations limit."""
        resolver = SuperpositionResolver(max_iterations=5, tolerance=1e-10)
        state = np.array([0.5, 0.5])

        def slow_converge(x):
            return x * 0.99 + 0.01 * np.array([0.5, 0.5])

        result = resolver.resolve(state, slow_converge)

        # May or may not converge in 5 iterations depending on initial state
        # Just check that iterations is limited
        assert result["iterations"] <= 5

    def test_to_dict_from_dict(self):
        """Test configuration serialization."""
        resolver = SuperpositionResolver(max_iterations=200, tolerance=1e-8)

        data = resolver.to_dict()
        restored = SuperpositionResolver.from_dict(data)

        assert restored.max_iterations == resolver.max_iterations
        assert restored.tolerance == resolver.tolerance


class TestDistributedStreamMonitor:
    """Tests for DistributedStreamMonitor."""

    def test_add_sample(self):
        """Test adding samples to streams."""
        monitor = DistributedStreamMonitor(num_streams=3, buffer_size=100)

        monitor.add_sample(0, 1.5)
        monitor.add_sample(0, 2.5)
        monitor.add_sample(1, 3.0)

        stats = monitor.get_stream_stats(0)
        assert stats["count"] == 2
        assert stats["mean"] == 2.0

    def test_stream_stats(self):
        """Test stream statistics calculation."""
        monitor = DistributedStreamMonitor(num_streams=2, buffer_size=100)

        for i in range(10):
            monitor.add_sample(0, float(i))

        stats = monitor.get_stream_stats(0)

        assert stats["count"] == 10
        assert stats["min"] == 0.0
        assert stats["max"] == 9.0
        assert abs(stats["mean"] - 4.5) < 0.1

    def test_detect_sync_patterns(self):
        """Test synchronization pattern detection."""
        monitor = DistributedStreamMonitor(num_streams=3, buffer_size=100)
        rng = np.random.default_rng(42)

        # Create correlated streams
        for _i in range(50):
            base = rng.normal(0, 1)
            monitor.add_sample(0, base + rng.normal(0, 0.1))
            monitor.add_sample(1, base + rng.normal(0, 0.1))
            monitor.add_sample(2, rng.normal(0, 1))  # Independent

        sync_pairs = monitor.detect_sync_patterns(threshold=0.7)

        # Streams 0 and 1 should be synchronized
        assert (0, 1) in sync_pairs or (1, 0) in sync_pairs

    def test_buffer_size_limit(self):
        """Test that buffer respects size limit."""
        monitor = DistributedStreamMonitor(num_streams=1, buffer_size=5)

        for i in range(10):
            monitor.add_sample(0, float(i))

        stats = monitor.get_stream_stats(0)
        assert stats["count"] == 5  # Only last 5 samples

    def test_to_dict_from_dict(self):
        """Test serialization and deserialization."""
        monitor = DistributedStreamMonitor(num_streams=2, buffer_size=10)
        monitor.add_sample(0, 1.0)
        monitor.add_sample(0, 2.0)
        monitor.add_sample(1, 3.0)

        data = monitor.to_dict()
        restored = DistributedStreamMonitor.from_dict(data)

        assert restored.num_streams == monitor.num_streams
        assert restored.get_stream_stats(0)["count"] == 2
        assert restored.get_stream_stats(1)["count"] == 1


class TestSelfConsistencyPropagator:
    """Tests for SelfConsistencyPropagator."""

    def test_propagate_converges(self):
        """Test that propagation converges."""
        propagator = SelfConsistencyPropagator(num_nodes=5, damping=0.5, max_iterations=100)
        rng = np.random.default_rng(42)
        states = rng.normal(0, 1, size=(5, 3))

        result = propagator.propagate(states)

        assert result["converged"]
        assert result["iterations"] < 100

        # Check that states are more similar after propagation
        final_var = np.var(result["states"], axis=0)
        initial_var = np.var(states, axis=0)
        assert np.all(final_var <= initial_var)

    def test_propagate_uniform_states(self):
        """Test propagation with already uniform states."""
        propagator = SelfConsistencyPropagator(num_nodes=5, damping=0.5)
        states = np.ones((5, 3))

        result = propagator.propagate(states)

        # Should converge immediately
        assert result["converged"]
        assert result["iterations"] == 1

    def test_custom_adjacency(self):
        """Test with custom adjacency matrix."""
        propagator = SelfConsistencyPropagator(num_nodes=3, damping=0.5)
        states = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

        # Linear chain: 0 <-> 1 <-> 2
        adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)

        result = propagator.propagate(states, adjacency=adjacency)

        assert result["converged"]

    def test_to_dict_from_dict(self):
        """Test configuration serialization."""
        propagator = SelfConsistencyPropagator(
            num_nodes=10, damping=0.7, max_iterations=200, tolerance=1e-6
        )

        data = propagator.to_dict()
        restored = SelfConsistencyPropagator.from_dict(data)

        assert restored.num_nodes == propagator.num_nodes
        assert restored.damping == propagator.damping
        assert restored.max_iterations == propagator.max_iterations
        assert restored.tolerance == propagator.tolerance


class TestStabilityMonitor:
    """Tests for StabilityMonitor."""

    def test_stable_system(self):
        """Test detection of stable system."""
        monitor = StabilityMonitor(window_size=20, threshold=2.0)

        # Add stable metrics (oscillating around mean)
        for i in range(30):
            monitor.add_metric(1.0 + 0.1 * np.sin(i))

        assert monitor.is_stable()

    def test_unstable_system(self):
        """Test detection of unstable system."""
        monitor = StabilityMonitor(window_size=20, threshold=0.1)  # Very low threshold

        # Add unstable metrics (strong linear trend)
        for i in range(50):
            monitor.add_metric(float(i) * 10.0)  # Very steep trend

        # Check that trend is detected
        stats = monitor.get_stats()
        assert stats["trend"] > 1.0  # Strong positive trend

    def test_get_stats(self):
        """Test statistics computation."""
        monitor = StabilityMonitor(window_size=10, threshold=2.0)

        for i in range(15):
            monitor.add_metric(float(i))

        stats = monitor.get_stats()

        assert stats["count"] == 10  # Window size limit
        assert stats["mean"] > 0.0
        assert stats["std"] > 0.0
        assert stats["trend"] > 0.0  # Positive trend

    def test_insufficient_data(self):
        """Test stability with insufficient data."""
        monitor = StabilityMonitor(window_size=50, threshold=2.0)

        # Add only a few metrics
        for i in range(10):
            monitor.add_metric(float(i))

        # Should be considered stable with insufficient data
        assert monitor.is_stable()

    def test_to_dict_from_dict(self):
        """Test serialization and deserialization."""
        monitor = StabilityMonitor(window_size=20, threshold=1.5)
        monitor.add_metric(1.0)
        monitor.add_metric(2.0)
        monitor.add_metric(3.0)

        data = monitor.to_dict()
        restored = StabilityMonitor.from_dict(data)

        assert restored.window_size == monitor.window_size
        assert restored.threshold == monitor.threshold
        assert restored.get_stats()["count"] == 3


class TestIntegration:
    """Integration tests for QGH algorithms."""

    def test_full_workflow(self):
        """Test complete workflow with all components."""
        # Create history tracker
        chh = CausalHistoryHash(history_size=100)

        # Create stream monitor
        monitor = DistributedStreamMonitor(num_streams=3, buffer_size=100)

        # Create stability monitor
        stability = StabilityMonitor(window_size=30)

        # Simulate distributed system
        rng = np.random.default_rng(42)
        for i in range(50):
            # Track events
            event_data = rng.normal(0, 1, size=5)
            chh.add_event(f"event_{i}", event_data)

            # Monitor streams
            base = rng.normal(0, 1)
            for stream_id in range(3):
                monitor.add_sample(stream_id, base + rng.normal(0, 0.2))

            # Monitor stability
            stability.add_metric(abs(base))

        # Verify all components worked
        assert chh.verify_causality("event_49")
        assert monitor.get_stream_stats(0)["count"] == 50
        sync_pairs = monitor.detect_sync_patterns()
        assert len(sync_pairs) > 0
        assert stability.get_stats()["count"] == 30
