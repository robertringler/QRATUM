#!/usr/bin/env python3
"""Demo script showcasing Phase III autonomous kernel evolution."""
from __future__ import annotations

import time

from certs.verifier import StabilityVerifier
from evolve.energy_monitor import EnergyMonitor
from evolve.introspection import IntrospectionAgent, KernelMetrics
from evolve.precision_manager import PrecisionManager
from evolve.rl_controller import RLController
from federated.aggregator import FederatedAggregator
from memgraph.memory_optimizer import MemoryGraphOptimizer
from profiles.causal.profiler import CausalProfiler
from quantum_search.ising_optimizer import IsingOptimizer
from schedules.scheduler import DifferentiableScheduler


def demo_evolution():
    """Demonstrate kernel evolution with RL controller."""
    print("\n" + "=" * 70)
    print("DEMO 1: Self-Evolving Kernel Architectures")
    print("=" * 70)

    agent = IntrospectionAgent()
    controller = RLController(population_size=10, seed=42)

    print("Initializing population of 10 genomes...")
    population = controller.initialize_population()

    print("\nEvolution over 5 generations:")
    for gen in range(5):
        print(f"\nGeneration {gen + 1}:")

        for i, genome in enumerate(population):
            # Simulate kernel execution
            latency = 100.0 + genome.tile_size / 5.0 + genome.warp_count * 0.5
            energy = 10.0 + genome.tile_size / 50.0 + genome.warp_count * 0.2

            metrics = KernelMetrics(
                kernel_id=f"kernel_g{gen}_i{i}",
                latency_ms=latency,
                energy_joules=energy,
                tile_size=genome.tile_size,
                warp_count=genome.warp_count,
            )
            agent.record_metrics(metrics)
            controller.evaluate_fitness(genome, latency, energy)

        population = controller.select_and_evolve()

        if controller.best_genome:
            print(f"  Best fitness: {controller.best_genome.fitness:.4f}")
            print(f"  Best config: tile={controller.best_genome.tile_size}, "
                  f"warp={controller.best_genome.warp_count}, "
                  f"precision={controller.best_genome.precision}")

    controller.save_policy()
    print("\nSaved evolved policy to evolve/policies/policy.json")


def demo_precision():
    """Demonstrate hierarchical precision management."""
    print("\n" + "=" * 70)
    print("DEMO 2: Hierarchical Hybrid Precision Graphs")
    print("=" * 70)

    manager = PrecisionManager()

    print("Creating precision map for kernel...")
    precision_map = manager.create_default_map("demo_kernel")

    print("\nPrecision zones:")
    for zone in precision_map.zones:
        print(f"  {zone.zone_id:10s}: {zone.precision.value:6s} "
              f"(region={zone.region}, tolerance={zone.error_tolerance:.1e})")

    print(f"\nGlobal error budget: {precision_map.global_error_budget:.1e}")

    # Simulate error accumulation
    print("\nSimulating error accumulation...")
    manager.update_accumulated_error("demo_kernel", 3e-6)
    print("  Added error: 3e-6 (within budget)")

    needs_fallback = manager.update_accumulated_error("demo_kernel", 8e-6)
    if needs_fallback:
        print("  Added error: 8e-6 → FALLBACK TRIGGERED!")
        print(f"  All zones upgraded to {precision_map.fallback_precision.value}")

    manager.save_map("demo_kernel")
    print("\nSaved precision map to evolve/precision_maps/")


def demo_scheduling():
    """Demonstrate differentiable scheduling."""
    print("\n" + "=" * 70)
    print("DEMO 3: Differentiable Compiler Scheduling")
    print("=" * 70)

    scheduler = DifferentiableScheduler(learning_rate=0.02)

    print("Optimizing schedule with gradient descent (100 steps)...")
    metadata = scheduler.optimize_schedule("demo_schedule", steps=100)

    print("\nOptimized parameters:")
    print(f"  Block size:          {metadata.params.block_size:.1f}")
    print(f"  Thread count:        {metadata.params.thread_count:.1f}")
    print(f"  Register pressure:   {metadata.params.register_pressure:.1f}")
    print(f"  Coalesce factor:     {metadata.params.memory_coalesce_factor:.2f}")
    print(f"  Prefetch distance:   {metadata.params.prefetch_distance:.1f}")

    print("\nPerformance metrics:")
    print(f"  Latency:  {metadata.latency_ms:.2f} ms")
    print(f"  Energy:   {metadata.energy_j:.2f} J")
    print(f"  Loss:     {metadata.loss_value:.4f}")

    scheduler.save_schedule("demo_schedule")
    print("\nSaved schedule to schedules/demo_schedule.json")


def demo_quantum_search():
    """Demonstrate quantum-inspired optimization."""
    print("\n" + "=" * 70)
    print("DEMO 4: Quantum-Inspired Kernel Search")
    print("=" * 70)

    optimizer = IsingOptimizer(num_parameters=5, seed=42)

    print("Running simulated annealing...")
    print("  Initial temperature: 10.0")
    print("  Final temperature:   0.01")
    print("  Cooling rate:        0.95")

    kernel_config, energy = optimizer.optimize_kernel_config()

    print(f"\nOptimized configuration (energy={energy:.4f}):")
    for key, value in kernel_config.items():
        print(f"  {key:20s}: {value}")

    print(f"\nTotal iterations: {optimizer.history.iterations}")
    optimizer.save_history()
    print("Saved optimization history to quantum_search/history.json")


def demo_memory_graph():
    """Demonstrate memory graph optimization."""
    print("\n" + "=" * 70)
    print("DEMO 5: Topological Memory Graph Optimizer")
    print("=" * 70)

    optimizer = MemoryGraphOptimizer()

    print("Building memory graph...")
    optimizer.add_node("input_tensor", size_bytes=4096, access_frequency=20)
    optimizer.add_node("weight_matrix", size_bytes=8192, access_frequency=15)
    optimizer.add_node("output_tensor", size_bytes=4096, access_frequency=10)
    optimizer.add_node("bias_vector", size_bytes=512, access_frequency=5)

    optimizer.add_edge("input_tensor", "weight_matrix")
    optimizer.add_edge("weight_matrix", "output_tensor")
    optimizer.add_edge("bias_vector", "output_tensor")

    print("Optimizing layout...")
    layout = optimizer.optimize_layout("demo_layout")

    print("\nOptimized node order:")
    for i, node_id in enumerate(layout.node_order):
        print(f"  {i}: {node_id}")

    print("\nMetrics:")
    print(f"  Total path length: {layout.total_path_length:.2f}")
    print(f"  Cache miss rate:   {layout.cache_miss_rate:.2%}")

    optimizer.save_layout("demo_layout")
    print("\nSaved layout to memgraph/demo_layout.json")


def demo_causal_profiling():
    """Demonstrate causal profiling."""
    print("\n" + "=" * 70)
    print("DEMO 6: Causal Profiling & Counterfactual Benchmarking")
    print("=" * 70)

    profiler = CausalProfiler(delay_increment_ms=2.0)

    print("Running perturbation experiments...")

    # Define workloads
    def func_a_baseline():
        time.sleep(0.01)

    def func_a_perturbed():
        time.sleep(0.01)
        profiler.inject_delay(2.0)

    def func_b_baseline():
        time.sleep(0.005)

    def func_b_perturbed():
        time.sleep(0.005)
        profiler.inject_delay(2.0)

    # Profile functions
    result_a = profiler.profile_function("func_a", func_a_baseline, func_a_perturbed)
    result_b = profiler.profile_function("func_b", func_b_baseline, func_b_perturbed)

    print("\nFunction A:")
    print(f"  Baseline:  {result_a.baseline_latency_ms:.2f} ms")
    print(f"  Perturbed: {result_a.perturbed_latency_ms:.2f} ms")
    print(f"  Impact:    {result_a.causal_impact:+.2f} ms")

    print("\nFunction B:")
    print(f"  Baseline:  {result_b.baseline_latency_ms:.2f} ms")
    print(f"  Perturbed: {result_b.perturbed_latency_ms:.2f} ms")
    print(f"  Impact:    {result_b.causal_impact:+.2f} ms")

    # Generate influence map
    influence_map = profiler.compute_influence_map(total_runtime_ms=50.0)

    print("\nCausal influence scores:")
    for func_name, score in influence_map.influence_scores.items():
        print(f"  {func_name:10s}: {score:+.4f}")

    profiler.save_influence_map(influence_map)
    print("\nSaved influence map to profiles/causal/influence_map.json")


def demo_energy():
    """Demonstrate energy monitoring."""
    print("\n" + "=" * 70)
    print("DEMO 7: Energy-Adaptive Regulation")
    print("=" * 70)

    monitor = EnergyMonitor(max_temp_celsius=85.0, max_power_watts=400.0)

    print("Simulating thermal monitoring...")

    # Normal conditions
    print("\nScenario 1: Normal operation")
    telemetry = monitor.sample_telemetry(
        temperature=70.0, power=300.0, gflops=1200.0, duration_s=1.0
    )
    print(f"  Temperature: {telemetry.temperature_celsius:.1f}°C")
    print(f"  Power:       {telemetry.power_watts:.1f}W")
    print(f"  Efficiency:  {telemetry.gflops_per_watt:.2f} GFLOP/W")

    control = monitor.apply_feedback_control(telemetry)
    print(f"  Throttle:    {control['throttle_factor']:.2f}x (no action needed)")

    # Hot conditions
    print("\nScenario 2: Thermal stress")
    hot_telemetry = monitor.sample_telemetry(
        temperature=92.0, power=450.0, gflops=1000.0, duration_s=1.0
    )
    print(f"  Temperature: {hot_telemetry.temperature_celsius:.1f}°C (OVER LIMIT)")
    print(f"  Power:       {hot_telemetry.power_watts:.1f}W (OVER LIMIT)")
    print(f"  Efficiency:  {hot_telemetry.gflops_per_watt:.2f} GFLOP/W")

    control = monitor.apply_feedback_control(hot_telemetry)
    print(f"  Throttle:    {control['throttle_factor']:.2f}x (reducing performance)")
    print(f"  Migration:   {'YES' if control['needs_migration'] else 'NO'}")

    dashboard = monitor.generate_dashboard()
    print("\nDashboard summary:")
    print(f"  Total energy:     {dashboard.total_energy_j:.2f} J")
    print(f"  Avg efficiency:   {dashboard.average_efficiency:.2f} GFLOP/W")
    print(f"  Throttle events:  {dashboard.thermal_throttle_events}")

    monitor.save_dashboard(dashboard)
    print("\nSaved dashboard to evolve/energy_dashboard.json")


def demo_verification():
    """Demonstrate formal verification."""
    print("\n" + "=" * 70)
    print("DEMO 8: Formal Stability Certification")
    print("=" * 70)

    verifier = StabilityVerifier()

    print("Verifying kernel stability...")
    certificate = verifier.verify_kernel("demo_kernel", precision="fp32")

    print(f"\nVerification result: {'✓ VERIFIED' if certificate.verified else '✗ FAILED'}")
    print(f"Floating-point stable: {certificate.floating_point_stable}")
    print(f"Max error bound: {certificate.max_error_bound:.1e}")

    print("\nInvariants checked:")
    for inv in certificate.invariants:
        status = "✓" if inv.verified else "✗"
        print(f"  {status} [{inv.constraint_type:12s}] {inv.expression}")

    verifier.save_certificate("demo_kernel")
    print("\nSaved certificate to certs/demo_kernel_certificate.json")


def demo_federated():
    """Demonstrate federated learning."""
    print("\n" + "=" * 70)
    print("DEMO 9: Federated Kernel Intelligence")
    print("=" * 70)

    aggregator = FederatedAggregator()

    print("Collecting anonymized telemetry from 3 deployments...")

    config_a = {"tile_size": 256, "warp_count": 32, "precision": "fp16"}
    config_b = {"tile_size": 512, "warp_count": 64, "precision": "fp32"}

    # Deployment 1
    aggregator.add_telemetry(
        "deployment_us_west", config_a, latency_ms=95.0, energy_j=9.5,
        throughput_gops=1050.0, timestamp=time.time()
    )

    # Deployment 2
    aggregator.add_telemetry(
        "deployment_eu_central", config_a, latency_ms=102.0, energy_j=10.2,
        throughput_gops=980.0, timestamp=time.time()
    )

    # Deployment 3
    aggregator.add_telemetry(
        "deployment_asia_east", config_b, latency_ms=78.0, energy_j=12.0,
        throughput_gops=1280.0, timestamp=time.time()
    )

    print("Aggregating performance data...")
    aggregated = aggregator.aggregate_performance()

    print(f"\nAggregated statistics ({len(aggregated)} unique configs):")
    for config_hash, perf in aggregated.items():
        print(f"\nConfig {config_hash[:8]}:")
        print(f"  Samples:     {perf.sample_count}")
        print(f"  Avg latency: {perf.avg_latency_ms:.2f} ms")
        print(f"  Avg energy:  {perf.avg_energy_j:.2f} J")
        print(f"  Throughput:  {perf.avg_throughput_gops:.2f} GFLOP/s")

    # Predict performance
    prediction = aggregator.predict_performance(config_a)
    if prediction:
        print("\nPerformance prediction for config_a:")
        print(f"  Expected latency: {prediction.avg_latency_ms:.2f} ms")
        print(f"  Based on {prediction.sample_count} samples")

    aggregator.save_aggregated_data()
    print("\nSaved aggregated data to federated/aggregated/aggregated_performance.json")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "Phase III Autonomous Kernel Evolution Demo" + " " * 15 + "║")
    print("╚" + "═" * 68 + "╝")

    demos = [
        demo_evolution,
        demo_precision,
        demo_scheduling,
        demo_quantum_search,
        demo_memory_graph,
        demo_causal_profiling,
        demo_energy,
        demo_verification,
        demo_federated,
    ]

    for demo in demos:
        demo()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nAll Phase III components demonstrated successfully!")
    print("\nGenerated artifacts:")
    print("  - evolve/policies/policy.json")
    print("  - evolve/precision_maps/")
    print("  - schedules/demo_schedule.json")
    print("  - quantum_search/history.json")
    print("  - memgraph/demo_layout.json")
    print("  - profiles/causal/influence_map.json")
    print("  - evolve/energy_dashboard.json")
    print("  - certs/demo_kernel_certificate.json")
    print("  - federated/aggregated/aggregated_performance.json")
    print("\nSee PHASE3_OVERVIEW.md for detailed documentation.")
    print()


if __name__ == "__main__":
    main()
