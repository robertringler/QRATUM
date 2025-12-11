#!/usr/bin/env python3
"""QGH Practical Examples

Demonstrates practical applications of QGH non-speculative algorithms
in real-world scenarios.
"""


import numpy as np

from quasim.qgh.nonspec_algorithms import (
    CausalHistoryHash,
    DistributedStreamMonitor,
    SelfConsistencyPropagator,
    StabilityMonitor,
    SuperpositionResolver,
)


def example_federated_learning():
    """Federated learning with consensus."""
    print("=" * 70)
    print("Example 1: Federated Learning Consensus")
    print("=" * 70)

    # Simulate 10 edge devices with local model parameters
    num_devices = 10
    param_dim = 50

    print(f"Simulating {num_devices} devices, each with {param_dim} parameters")

    # Each device has slightly different local model
    rng = np.random.default_rng(42)
    local_models = rng.normal(0, 1, size=(num_devices, param_dim))

    print(f"Initial variance: {np.mean(np.var(local_models, axis=0)):.4f}")

    # Propagate to global consensus
    propagator = SelfConsistencyPropagator(
        num_nodes=num_devices, damping=0.6, max_iterations=100, tolerance=1e-5
    )

    result = propagator.propagate(local_models)

    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final variance: {np.mean(np.var(result['states'], axis=0)):.4f}")

    # Global model is the average
    global_model = np.mean(result["states"], axis=0)
    print(f"Global model norm: {np.linalg.norm(global_model):.4f}")
    print()


def example_autonomous_vehicle_fusion():
    """Multi-sensor fusion for autonomous vehicles."""
    print("=" * 70)
    print("Example 2: Autonomous Vehicle Sensor Fusion")
    print("=" * 70)

    # Simulate 4 sensors: LIDAR, RADAR, Camera, GPS
    monitor = DistributedStreamMonitor(num_streams=4, buffer_size=1000)
    rng = np.random.default_rng(42)

    # True position trajectory
    num_samples = 200
    true_positions = 10.0 + 0.05 * np.arange(num_samples)

    print(f"Processing {num_samples} samples from 4 sensors")

    # Each sensor has different accuracy
    sensor_noise = [0.2, 0.5, 1.0, 2.5]  # LIDAR most accurate, GPS least
    sensor_names = ["LIDAR", "RADAR", "Camera", "GPS"]

    for t in range(num_samples):
        for sensor_id, noise in enumerate(sensor_noise):
            measured = true_positions[t] + rng.normal(0, noise)
            monitor.add_sample(sensor_id, measured)

    # Get statistics
    print("\nSensor Statistics:")
    stats_list = []
    for sensor_id, name in enumerate(sensor_names):
        stats = monitor.get_stream_stats(sensor_id)
        stats_list.append(stats)
        print(f"  {name:8s}: mean={stats['mean']:6.2f}, std={stats['std']:.2f}")

    # Weighted fusion (inverse variance weighting)
    weights = np.array([1.0 / (s["std"] ** 2 + 0.01) for s in stats_list])
    weights /= np.sum(weights)

    fused_position = sum(s["mean"] * w for s, w in zip(stats_list, weights))
    true_avg = np.mean(true_positions)

    print(f"\nFused position: {fused_position:.2f}")
    print(f"True position (avg): {true_avg:.2f}")
    print(f"Fusion error: {abs(fused_position - true_avg):.2f}")

    # Check synchronization
    sync_pairs = monitor.detect_sync_patterns(threshold=0.6)
    print(f"\nSynchronized sensor pairs: {len(sync_pairs)}")
    for i, j in sync_pairs:
        print(f"  {sensor_names[i]} <-> {sensor_names[j]}")
    print()


def example_portfolio_risk_monitoring():
    """Portfolio risk and stability monitoring."""
    print("=" * 70)
    print("Example 3: Portfolio Risk Monitoring")
    print("=" * 70)

    stability = StabilityMonitor(window_size=30, threshold=1.5)
    rng = np.random.default_rng(42)

    # Simulate portfolio returns over time with increasing volatility
    num_days = 100
    print(f"Monitoring portfolio over {num_days} trading days")

    returns = []
    for day in range(num_days):
        # Volatility increases over time (crisis scenario)
        base_vol = 0.01
        crisis_factor = 1.0 + 0.03 * (day / num_days) ** 2
        daily_return = rng.normal(0.0005, base_vol * crisis_factor)
        returns.append(daily_return)

        # Monitor absolute returns (volatility proxy)
        stability.add_metric(abs(daily_return))

    # Check stability at different points
    print("\nStability Assessment:")
    print(f"  Overall stable: {stability.is_stable()}")

    stats = stability.get_stats()
    print(f"  Mean volatility: {stats['mean']:.4f}")
    print(f"  Trend: {stats['trend']:.6f}")

    if not stability.is_stable():
        print("\n⚠ WARNING: Portfolio volatility increasing!")
        print("  Recommendation: Consider reducing risk exposure")
    else:
        print("\n✓ Portfolio volatility stable")
    print()


def example_event_tracking():
    """Causal event tracking."""
    print("=" * 70)
    print("Example 4: Quantum Measurement Event Tracking")
    print("=" * 70)

    chh = CausalHistoryHash(history_size=100, hash_algo="sha256")
    rng = np.random.default_rng(42)

    # Simulate quantum measurement events
    num_events = 50
    print(f"Tracking {num_events} quantum measurement events")

    for i in range(num_events):
        # Each event has measurement data
        measurement = rng.normal(0, 1, size=5)
        event_id = f"measurement_{i}"
        event_hash = chh.add_event(event_id, measurement)

        if i < 3:  # Show first few
            print(f"  Event {i}: {event_hash[:16]}...")

    # Verify causality
    print("\nCausality verification:")
    print(f"  Event 0 in history: {chh.verify_causality('measurement_0')}")
    print(f"  Event 49 in history: {chh.verify_causality('measurement_49')}")
    print(f"  Nonexistent event: {chh.verify_causality('measurement_999')}")

    history = chh.get_history()
    print(f"\nTotal events in history: {len(history)}")
    print()


def example_superposition_resolution():
    """Quantum superposition resolution."""
    print("=" * 70)
    print("Example 5: Quantum State Resolution")
    print("=" * 70)

    # Initial superposed state (unnormalized probabilities)
    initial_state = np.array([0.6, 0.25, 0.1, 0.05])
    print(f"Initial state: {initial_state}")
    print(f"Sum: {np.sum(initial_state)}")

    # Resolve to valid probability distribution
    resolver = SuperpositionResolver(max_iterations=100, tolerance=1e-8)

    def normalize_probability(state):
        """Consistency function: must sum to 1.0."""
        return state / np.sum(state)

    result = resolver.resolve(initial_state, normalize_probability)

    print(f"\nFinal state: {result['state']}")
    print(f"Sum: {np.sum(result['state']):.10f}")
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final delta: {result['final_delta']:.2e}")
    print()


def main():
    """Run all examples."""
    examples = [
        example_federated_learning,
        example_autonomous_vehicle_fusion,
        example_portfolio_risk_monitoring,
        example_event_tracking,
        example_superposition_resolution,
    ]


    for example_fn in examples:
        example_fn()

    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
