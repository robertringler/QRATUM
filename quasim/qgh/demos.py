"""Demo examples for QGH non-speculative algorithms."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from quasim.qgh.nonspec_algorithms import (
    CausalHistoryHash,
    DistributedStreamMonitor,
    SelfConsistencyPropagator,
    StabilityMonitor,
    SuperpositionResolver,
)

logger = logging.getLogger(__name__)


def demo_tensor_processing() -> dict[str, Any]:
    """Demonstrate tensor processing with SuperpositionResolver.

    Returns
    -------
    dict[str, Any]
        Demo results
    """

    logger.info("Running tensor processing demo...")

    # Create superposed state
    state = np.array([0.5, 0.3, 0.15, 0.05])

    # Define consistency function (normalize to sum to 1.0)
    def normalize(x: np.ndarray) -> np.ndarray:
        return x / np.sum(x)

    resolver = SuperpositionResolver(max_iterations=50, tolerance=1e-8)
    result = resolver.resolve(state, normalize)

    logger.info(f"Converged: {result['converged']}, Iterations: {result['iterations']}")

    return {
        "initial_state": state.tolist(),
        "final_state": result["state"].tolist(),
        "converged": result["converged"],
        "iterations": result["iterations"],
    }


def demo_distributed_monitoring() -> dict[str, Any]:
    """Demonstrate distributed stream monitoring.

    Returns
    -------
    dict[str, Any]
        Demo results
    """

    logger.info("Running distributed monitoring demo...")

    monitor = DistributedStreamMonitor(num_streams=3, buffer_size=100)
    rng = np.random.default_rng(42)

    # Simulate correlated streams
    for _i in range(50):
        base = rng.normal(0, 1)
        monitor.add_sample(0, base + rng.normal(0, 0.1))
        monitor.add_sample(1, base + rng.normal(0, 0.1))  # Correlated with stream 0
        monitor.add_sample(2, rng.normal(0, 1))  # Independent

    sync_patterns = monitor.detect_sync_patterns(threshold=0.7)
    stats = [monitor.get_stream_stats(i) for i in range(3)]

    logger.info(f"Detected {len(sync_patterns)} synchronized pairs: {sync_patterns}")

    return {
        "sync_patterns": sync_patterns,
        "stream_stats": stats,
        "num_streams": 3,
    }


def demo_federated_consensus() -> dict[str, Any]:
    """Demonstrate federated consensus with SelfConsistencyPropagator.

    Returns
    -------
    dict[str, Any]
        Demo results
    """

    logger.info("Running federated consensus demo...")

    # Create nodes with different initial states
    num_nodes = 5
    state_dim = 3
    rng = np.random.default_rng(42)
    initial_states = rng.normal(0, 1, size=(num_nodes, state_dim))

    propagator = SelfConsistencyPropagator(num_nodes=num_nodes, damping=0.5, max_iterations=100)
    result = propagator.propagate(initial_states)

    logger.info(f"Consensus converged: {result['converged']}, Iterations: {result['iterations']}")

    # Check if all nodes reached similar states
    final_states = result["states"]
    state_variance = np.var(final_states, axis=0)

    return {
        "converged": result["converged"],
        "iterations": result["iterations"],
        "initial_variance": np.var(initial_states, axis=0).tolist(),
        "final_variance": state_variance.tolist(),
        "consensus_achieved": bool(np.all(state_variance < 0.1)),
    }


def demo_av_fusion() -> dict[str, Any]:
    """Demonstrate AV (Autonomous Vehicle) sensor fusion.

    Returns
    -------
    dict[str, Any]
        Demo results
    """

    logger.info("Running AV sensor fusion demo...")

    # Simulate multiple sensor streams
    monitor = DistributedStreamMonitor(num_streams=4, buffer_size=200)
    rng = np.random.default_rng(42)

    # Sensor 0: LIDAR, Sensor 1: RADAR, Sensor 2: Camera, Sensor 3: GPS
    true_position = 10.0
    for _i in range(100):
        monitor.add_sample(0, true_position + rng.normal(0, 0.5))  # LIDAR
        monitor.add_sample(1, true_position + rng.normal(0, 0.8))  # RADAR
        monitor.add_sample(2, true_position + rng.normal(0, 1.0))  # Camera
        monitor.add_sample(3, true_position + rng.normal(0, 2.0))  # GPS

    stats = [monitor.get_stream_stats(i) for i in range(4)]
    monitor.detect_sync_patterns(threshold=0.6)

    # Weighted fusion based on sensor accuracy (inverse variance)
    weights = np.array([1.0 / (s["std"] ** 2 + 0.01) for s in stats])
    weights /= np.sum(weights)
    fused_position = np.sum([s["mean"] * w for s, w in zip(stats, weights)])

    logger.info(f"Fused position: {fused_position:.2f}, True: {true_position:.2f}")

    return {
        "sensor_stats": stats,
        "weights": weights.tolist(),
        "fused_position": float(fused_position),
        "true_position": true_position,
        "error": float(abs(fused_position - true_position)),
    }


def demo_portfolio_optimization() -> dict[str, Any]:
    """Demonstrate portfolio optimization with stability monitoring.

    Returns
    -------
    dict[str, Any]
        Demo results
    """

    logger.info("Running portfolio optimization demo...")

    stability = StabilityMonitor(window_size=30, threshold=1.5)
    rng = np.random.default_rng(42)

    # Simulate portfolio returns with increasing volatility
    returns = []
    for i in range(50):
        volatility = 1.0 + 0.05 * i  # Increasing volatility
        ret = rng.normal(0.1, volatility)
        returns.append(ret)
        stability.add_metric(abs(ret))

    is_stable = stability.is_stable()
    stats = stability.get_stats()

    logger.info(f"Portfolio stable: {is_stable}, Trend: {stats['trend']:.4f}")

    return {
        "returns": returns[-10:],  # Last 10 returns
        "is_stable": is_stable,
        "stats": stats,
        "recommendation": "reduce_risk" if not is_stable else "maintain",
    }


def demo_signal_processing() -> dict[str, Any]:
    """Demonstrate signal processing with causal history.

    Returns
    -------
    dict[str, Any]
        Demo results
    """

    logger.info("Running signal processing demo...")

    chh = CausalHistoryHash(history_size=100)
    rng = np.random.default_rng(42)

    # Process signal events
    events = []
    for i in range(20):
        event_id = f"signal_{i}"
        data = rng.normal(0, 1, size=10)
        event_hash = chh.add_event(event_id, data)
        events.append((event_id, event_hash))

    # Verify causality
    verified = sum(1 for event_id, _ in events if chh.verify_causality(event_id))

    logger.info(f"Verified {verified}/{len(events)} events in causal history")

    return {
        "total_events": len(events),
        "verified_events": verified,
        "history_size": len(chh.get_history()),
        "causality_preserved": verified == len(events),
    }


if __name__ == "__main__":
    # Run demo analyses
    logging.basicConfig(level=logging.INFO)

    demos = [
        ("Tensor Processing", demo_tensor_processing),
        ("Distributed Monitoring", demo_distributed_monitoring),
        ("Federated Consensus", demo_federated_consensus),
        ("AV Sensor Fusion", demo_av_fusion),
        ("Portfolio Optimization", demo_portfolio_optimization),
        ("Signal Processing", demo_signal_processing),
    ]

    for name, demo_fn in demos:
        print("\n" + "=" * 70)
        print(f"QGH Demo: {name}")
        print("=" * 70)
        result = demo_fn()
        print(f"Result: {result}")
