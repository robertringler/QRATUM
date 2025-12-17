"""Tests for Phase III autonomous kernel evolution."""

from __future__ import annotations

import time

# Verification tests
from certs.verifier import StabilityVerifier

# Energy monitor tests
from evolve.energy_monitor import EnergyMonitor

# Introspection tests
from evolve.introspection import IntrospectionAgent, KernelMetrics

# Precision manager tests
from evolve.precision_manager import PrecisionLevel, PrecisionManager

# RL controller tests
from evolve.rl_controller import RLController

# Federated learning tests
from federated.aggregator import FederatedAggregator

# Memory optimizer tests
from memgraph.memory_optimizer import MemoryGraphOptimizer

# Causal profiler tests
from profiles.causal.profiler import CausalProfiler

# Quantum search tests
from quantum_search.ising_optimizer import IsingOptimizer

# Scheduler tests
from schedules.scheduler import DifferentiableScheduler


def test_introspection_agent():
    """Test introspection agent metrics collection."""

    agent = IntrospectionAgent(log_dir="/tmp/evolve_test/logs")

    metrics = KernelMetrics(
        kernel_id="test_kernel",
        latency_ms=50.0,
        energy_joules=5.0,
        tile_size=256,
        warp_count=32,
    )

    agent.record_metrics(metrics)
    stats = agent.compute_statistics()

    assert stats["total_kernels"] == 1
    assert stats["avg_latency_ms"] == 50.0
    assert stats["avg_energy_j"] == 5.0


def test_rl_controller_evolution():
    """Test RL controller population evolution."""

    controller = RLController(population_size=10, seed=42)

    # Initialize population
    population = controller.initialize_population()
    assert len(population) == 10

    # Evaluate fitness
    for genome in population:
        controller.evaluate_fitness(genome, latency_ms=100.0, energy_j=10.0)

    # Select and evolve
    new_population = controller.select_and_evolve()
    assert len(new_population) == 10
    assert controller.generation == 1
    assert controller.best_genome is not None


def test_precision_manager():
    """Test hierarchical precision management."""

    manager = PrecisionManager(map_dir="/tmp/evolve_test/precision_maps")

    # Create precision map
    precision_map = manager.create_default_map("test_kernel")
    assert len(precision_map.zones) == 3

    # Test zone lookup
    precision = manager.get_precision_for_zone("test_kernel", "outer")
    assert precision == PrecisionLevel.FP32

    # Test error accumulation and fallback
    needs_fallback = manager.update_accumulated_error("test_kernel", 5e-6)
    assert not needs_fallback

    needs_fallback = manager.update_accumulated_error("test_kernel", 1e-4)
    assert needs_fallback


def test_energy_monitor():
    """Test energy monitoring and thermal control."""

    monitor = EnergyMonitor(max_temp_celsius=85.0, max_power_watts=400.0)

    # Sample telemetry (normal conditions)
    telemetry = monitor.sample_telemetry(
        temperature=70.0, power=300.0, gflops=1000.0, duration_s=1.0
    )

    assert telemetry.gflops_per_watt > 0
    assert not monitor.check_thermal_limits(telemetry)

    # Test thermal limit
    hot_telemetry = monitor.sample_telemetry(
        temperature=90.0, power=450.0, gflops=1000.0, duration_s=1.0
    )

    assert monitor.check_thermal_limits(hot_telemetry)
    control = monitor.apply_feedback_control(hot_telemetry)
    assert control["throttle_factor"] < 1.0


def test_differentiable_scheduler():
    """Test differentiable scheduling optimization."""

    scheduler = DifferentiableScheduler(learning_rate=0.01)

    # Optimize schedule
    metadata = scheduler.optimize_schedule("test_schedule", steps=50)

    assert metadata.optimization_steps == 50
    assert metadata.loss_value > 0
    assert metadata.params.block_size >= 64.0
    assert metadata.params.block_size <= 1024.0


def test_ising_optimizer():
    """Test quantum-inspired Ising optimization."""

    optimizer = IsingOptimizer(num_parameters=5, seed=42)

    # Run simulated annealing
    best_config = optimizer.simulated_annealing(
        initial_temp=5.0, final_temp=0.1, cooling_rate=0.9, steps_per_temp=50
    )

    assert len(best_config.spins) == 5
    assert all(s in [-1, 1] for s in best_config.spins)

    # Convert to kernel config
    kernel_config, energy = optimizer.optimize_kernel_config()
    assert "tile_size" in kernel_config
    assert "warp_count" in kernel_config


def test_memory_graph_optimizer():
    """Test topological memory graph optimization."""

    optimizer = MemoryGraphOptimizer()

    # Build graph
    optimizer.add_node("tensor_a", size_bytes=1024, access_frequency=10)
    optimizer.add_node("tensor_b", size_bytes=2048, access_frequency=5)
    optimizer.add_node("tensor_c", size_bytes=512, access_frequency=8)

    optimizer.add_edge("tensor_a", "tensor_b")
    optimizer.add_edge("tensor_b", "tensor_c")

    # Optimize layout
    layout = optimizer.optimize_layout("test_layout")

    assert len(layout.node_order) == 3
    assert layout.total_path_length >= 0
    assert 0 <= layout.cache_miss_rate <= 1.0


def test_causal_profiler():
    """Test causal profiling."""

    profiler = CausalProfiler(delay_increment_ms=1.0)

    # Define simple workloads
    def baseline_workload():
        time.sleep(0.01)  # 10ms

    def perturbed_workload():
        time.sleep(0.01)
        profiler.inject_delay(1.0)  # Add 1ms delay

    # Profile function
    result = profiler.profile_function("test_function", baseline_workload, perturbed_workload)

    assert result.causal_impact > 0

    # Generate influence map
    influence_map = profiler.compute_influence_map(total_runtime_ms=20.0)
    assert "test_function" in influence_map.influence_scores


def test_federated_aggregator():
    """Test federated learning aggregator."""

    aggregator = FederatedAggregator(aggregation_dir="/tmp/evolve_test/federated")

    # Add telemetry from multiple deployments
    config1 = {"tile_size": 256, "warp_count": 32}
    config2 = {"tile_size": 512, "warp_count": 64}

    aggregator.add_telemetry(
        "deployment_1",
        config1,
        latency_ms=100.0,
        energy_j=10.0,
        throughput_gops=1000.0,
        timestamp=time.time(),
    )

    aggregator.add_telemetry(
        "deployment_2",
        config1,
        latency_ms=105.0,
        energy_j=10.5,
        throughput_gops=980.0,
        timestamp=time.time(),
    )

    aggregator.add_telemetry(
        "deployment_3",
        config2,
        latency_ms=80.0,
        energy_j=12.0,
        throughput_gops=1200.0,
        timestamp=time.time(),
    )

    # Aggregate
    aggregated = aggregator.aggregate_performance()
    assert len(aggregated) == 2  # Two unique configs

    # Predict performance
    prediction = aggregator.predict_performance(config1)
    assert prediction is not None
    assert prediction.sample_count == 2


def test_stability_verifier():
    """Test formal stability verification."""

    verifier = StabilityVerifier()

    # Verify kernel
    certificate = verifier.verify_kernel("test_kernel", precision="fp32")

    assert certificate.kernel_id == "test_kernel"
    assert certificate.verified
    assert certificate.floating_point_stable
    assert certificate.max_error_bound == 1e-6
    assert len(certificate.invariants) == 3

    # Generate report
    report = verifier.generate_report("test_kernel")
    assert "VERIFIED" in report
    assert "test_kernel" in report


def test_end_to_end_evolution():
    """Test end-to-end evolution workflow."""

    # Initialize components
    agent = IntrospectionAgent(log_dir="/tmp/evolve_test/e2e_logs")
    controller = RLController(population_size=5, seed=42)

    # Initialize population
    population = controller.initialize_population()

    # Simulate evolution cycle
    for genome in population:
        # Simulate kernel execution with this genome
        metrics = KernelMetrics(
            kernel_id=f"kernel_{population.index(genome)}",
            latency_ms=100.0 + genome.tile_size / 10.0,
            energy_joules=10.0 + genome.warp_count / 10.0,
            tile_size=genome.tile_size,
            warp_count=genome.warp_count,
        )

        agent.record_metrics(metrics)

        # Evaluate fitness
        controller.evaluate_fitness(genome, metrics.latency_ms, metrics.energy_joules)

    # Evolve population
    new_population = controller.select_and_evolve()

    assert len(new_population) == 5
    assert controller.best_genome is not None

    # Verify best genome is better than average
    avg_fitness = sum(g.fitness for g in population) / len(population)
    assert controller.best_genome.fitness <= avg_fitness
