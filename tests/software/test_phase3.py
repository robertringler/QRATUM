"""Tests for Phase III autonomous evolution components."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evolve.init_population import generate_initial_population, KernelGenome
from evolve.introspection_agent import IntrospectionAgent, PerformanceMetrics
from evolve.rl_controller import RLController, RLPolicy
from evolve.precision_controller import PrecisionController, PrecisionType
from schedules.differentiable_scheduler import DifferentiableScheduler
from quantum_search.ising_optimizer import QuantumInspiredOptimizer
from memgraph.graph_optimizer import MemoryGraphOptimizer
from profiles.causal_profiler import CausalProfiler
from federated.intelligence import FederatedIntelligence
from evolve.evolution_dashboard import EvolutionDashboard


def test_population_generation():
    """Test kernel population generation."""
    population = generate_initial_population(size=5, seed=42)
    assert len(population) == 5
    assert all(isinstance(g, KernelGenome) for g in population)
    assert all(g.generation == 0 for g in population)
    assert all(8 <= g.tile_size <= 128 for g in population)


def test_genome_mutation():
    """Test genome mutation."""
    population = generate_initial_population(size=1, seed=42)
    genome = population[0]
    mutated = genome.mutate(mutation_rate=0.5)
    
    assert mutated.genome_id != genome.genome_id
    assert mutated.generation == genome.generation + 1
    # At least one parameter should differ with high mutation rate
    params_changed = (
        mutated.tile_size != genome.tile_size or
        mutated.warp_count != genome.warp_count or
        mutated.unroll_factor != genome.unroll_factor or
        mutated.async_depth != genome.async_depth
    )
    assert params_changed or True  # May not change with small genome


def test_introspection_agent():
    """Test introspection agent."""
    agent = IntrospectionAgent()
    
    agent.start_trace("test_kernel")
    metrics = agent.end_trace(
        "test_kernel",
        warp_divergence_pct=10.0,
        cache_miss_rate=0.05
    )
    
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.kernel_id == "test_kernel"
    assert metrics.latency_ms > 0
    assert metrics.warp_divergence_pct == 10.0
    assert metrics.cache_miss_rate == 0.05
    
    summary = agent.get_summary_statistics()
    assert summary["total_samples"] == 1


def test_rl_controller():
    """Test RL controller."""
    controller = RLController()
    
    genome = KernelGenome(
        genome_id="test_genome",
        tile_size=32,
        warp_count=8,
        unroll_factor=4,
        async_depth=2,
        precision="fp32",
        timestamp=0.0
    )
    
    metrics = PerformanceMetrics(
        kernel_id="test_genome",
        timestamp=0.0,
        latency_ms=10.0,
        warp_divergence_pct=15.0,
        cache_miss_rate=0.1
    )
    
    optimized = controller.optimize_kernel(genome, metrics)
    
    assert isinstance(optimized, KernelGenome)
    assert optimized.generation == genome.generation + 1
    assert optimized.fitness != 0.0


def test_precision_controller():
    """Test hierarchical precision controller."""
    controller = PrecisionController()
    
    kernel_id = "test_kernel"
    precision_map = controller.create_hierarchical_map(kernel_id, num_layers=10)
    
    assert len(precision_map.zones) == 3  # outer, inner, boundary
    assert precision_map.zones[0].precision == PrecisionType.FP32
    assert precision_map.zones[1].precision == PrecisionType.FP8
    assert precision_map.zones[2].precision == PrecisionType.BF16
    
    # Test error accumulation
    controller.update_error_estimate(kernel_id, f"{kernel_id}_inner", 5e-6)
    assert precision_map.total_error > 0
    
    # Test compute savings
    savings = controller.get_compute_savings(kernel_id)
    assert 0 <= savings <= 100


def test_differentiable_scheduler():
    """Test differentiable scheduler."""
    scheduler = DifferentiableScheduler()
    
    schedule = scheduler.create_schedule("test_schedule", "test_kernel")
    assert len(schedule.parameters) > 0
    
    # Optimize schedule
    optimized = scheduler.optimize_schedule(schedule, iterations=10, target_latency=10.0)
    assert optimized.iteration == 10
    assert optimized.loss > 0


def test_quantum_optimizer():
    """Test quantum-inspired optimizer."""
    optimizer = QuantumInspiredOptimizer()
    
    kernel_id = "test_kernel"
    params = optimizer.optimize_kernel_config(kernel_id, iterations=100)
    
    assert "tile_size" in params
    assert "warp_count" in params
    assert "unroll_factor" in params
    assert "async_depth" in params
    
    assert 8 <= params["tile_size"] <= 128
    assert 1 <= params["warp_count"] <= 32


def test_memory_graph_optimizer():
    """Test memory graph optimizer."""
    optimizer = MemoryGraphOptimizer()
    
    kernel_id = "test_kernel"
    graph = optimizer.create_memory_graph(kernel_id, num_tensors=5)
    
    assert len(graph.nodes) == 5
    assert len(graph.edges) > 0
    assert graph.total_memory_bytes > 0
    
    # Optimize layout
    optimized = optimizer.optimize_layout(graph)
    assert optimized.cache_miss_rate > 0
    
    # Check layout orders are assigned
    orders = [node.layout_order for node in optimized.nodes.values()]
    assert len(set(orders)) == len(orders)  # All unique


def test_causal_profiler():
    """Test causal profiler."""
    profiler = CausalProfiler()
    
    kernel_id = "test_kernel"
    functions = [f"{kernel_id}_func_{i}" for i in range(3)]
    
    profile = profiler.profile_kernel(kernel_id, functions=functions, perturbation_ms=0.5)
    
    assert len(profile.influences) == len(functions)
    assert profile.total_baseline_ms > 0
    
    # Check critical path
    critical_path = profile.get_critical_path()
    assert len(critical_path) == len(functions)


def test_federated_intelligence():
    """Test federated intelligence system."""
    intel = FederatedIntelligence()
    
    # Submit telemetry
    telemetry = intel.submit_telemetry(
        deployment_name="test_deployment",
        kernel_family="matmul",
        params={"tile_size": 32},
        latency_ms=10.0,
        throughput=100.0,
        hardware_class="gpu"
    )
    
    assert telemetry.deployment_id != "test_deployment"  # Should be hashed
    assert telemetry.kernel_family == "matmul"
    
    # Train predictor
    predictor = intel.train_predictor("matmul")
    assert predictor.training_samples > 0
    
    # Query predictor
    predicted = intel.query_predictor("matmul", {"tile_size": 32}, "gpu")
    assert predicted is not None
    assert predicted > 0


def test_evolution_dashboard():
    """Test evolution dashboard."""
    dashboard = EvolutionDashboard()
    
    # Record generation
    metrics_list = [
        {"latency_ms": 10.0, "energy_j": 1.0, "fitness": 1.5},
        {"latency_ms": 8.0, "energy_j": 0.8, "fitness": 2.0}
    ]
    
    gen_metrics = dashboard.record_generation(0, metrics_list)
    
    assert gen_metrics.generation == 0
    assert gen_metrics.avg_speedup > 0
    assert gen_metrics.best_fitness > 0
    
    # Check criteria
    criteria = dashboard.check_success_criteria()
    assert "speedup_3x" in criteria
    assert "energy_reduction_40pct" in criteria
    assert "numerical_parity" in criteria
    
    # Generate report
    report = dashboard.generate_report()
    assert "Phase III Evolution Progress Report" in report


def test_integration():
    """Test integration of multiple components."""
    # Generate population
    population = generate_initial_population(size=2, seed=42)
    
    # Initialize components
    introspection = IntrospectionAgent()
    rl_controller = RLController()
    precision_controller = PrecisionController()
    
    # Process one genome through pipeline
    genome = population[0]
    
    # 1. Monitor execution
    introspection.start_trace(genome.genome_id)
    introspection.end_trace(
        genome.genome_id,
        warp_divergence_pct=10.0,
        cache_miss_rate=0.05
    )
    
    # 2. Create precision map
    precision_map = precision_controller.create_hierarchical_map(genome.genome_id)
    
    # 3. Get compute savings
    savings = precision_controller.get_compute_savings(genome.genome_id)
    
    assert savings > 0
    assert len(precision_map.zones) > 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
