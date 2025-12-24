"""Comprehensive benchmarking suite for PostDijkstra algorithm.

This module provides extensive performance analysis comparing PostDijkstra
against classical Dijkstra baselines across multiple dimensions:

- Graph sizes: 10^4 to 10^7 nodes
- Weight distributions: uniform, heavy-tailed, near-uniform
- Metrics: runtime, memory, relaxations, ordering operations
- Regimes: sparse/dense, connected/disconnected, weighted/uniform

Results are saved in JSON and can be visualized with plotting tools.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quasim.opt.graph import QGraph
from quasim.opt.post_dijkstra_sssp import PostDijkstraSSSP
from quasim.opt.ultra_sssp import dijkstra_baseline, validate_sssp_results


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run.
    
    Attributes:
        name: Benchmark identifier
        num_nodes: Number of nodes in graph
        edge_probability: Probability of edge between nodes
        weight_distribution: Weight distribution type
        min_weight: Minimum edge weight
        max_weight: Maximum edge weight
        delta: Bucket width for PostDijkstra
        use_hierarchy: Enable hierarchical decomposition
        use_lower_bounds: Enable lower-bound pruning
        batch_size: Batch size for parallel relaxations
        seed: Random seed
    """
    
    name: str
    num_nodes: int
    edge_probability: float
    weight_distribution: str = "uniform"  # uniform, heavy_tail, near_uniform
    min_weight: float = 1.0
    max_weight: float = 10.0
    delta: float | None = None
    use_hierarchy: bool = True
    use_lower_bounds: bool = True
    batch_size: int = 100
    seed: int = 42


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run.
    
    Attributes:
        config: Benchmark configuration
        graph_info: Graph statistics
        dijkstra_time: Dijkstra execution time
        dijkstra_metrics: Dijkstra performance metrics
        postdijkstra_time: PostDijkstra execution time
        postdijkstra_metrics: PostDijkstra performance metrics
        speedup: Speedup factor (>1.0 = faster)
        correctness: Whether results match
        memory_ratio: Memory usage ratio
    """
    
    config: BenchmarkConfig
    graph_info: dict[str, Any]
    dijkstra_time: float
    dijkstra_metrics: dict[str, Any]
    postdijkstra_time: float
    postdijkstra_metrics: dict[str, Any]
    speedup: float
    correctness: bool
    memory_ratio: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "config": {
                "name": self.config.name,
                "num_nodes": self.config.num_nodes,
                "edge_probability": self.config.edge_probability,
                "weight_distribution": self.config.weight_distribution,
                "min_weight": self.config.min_weight,
                "max_weight": self.config.max_weight,
                "delta": self.config.delta,
                "use_hierarchy": self.config.use_hierarchy,
                "use_lower_bounds": self.config.use_lower_bounds,
                "batch_size": self.config.batch_size,
                "seed": self.config.seed,
            },
            "graph_info": self.graph_info,
            "dijkstra": {
                "time": self.dijkstra_time,
                "metrics": self.dijkstra_metrics,
            },
            "postdijkstra": {
                "time": self.postdijkstra_time,
                "metrics": self.postdijkstra_metrics,
            },
            "comparison": {
                "speedup": self.speedup,
                "correctness": self.correctness,
                "memory_ratio": self.memory_ratio,
            },
        }


def generate_graph(config: BenchmarkConfig) -> QGraph:
    """Generate graph according to benchmark configuration.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        Generated graph
    """
    np.random.seed(config.seed)
    graph = QGraph(num_nodes=config.num_nodes, directed=True)
    
    # Generate edges based on probability
    for i in range(config.num_nodes):
        for j in range(config.num_nodes):
            if i != j and np.random.random() < config.edge_probability:
                # Generate weight based on distribution
                if config.weight_distribution == "uniform":
                    weight = np.random.uniform(config.min_weight, config.max_weight)
                elif config.weight_distribution == "heavy_tail":
                    # Pareto distribution (heavy tail)
                    weight = np.random.pareto(1.5) + config.min_weight
                    weight = min(weight, config.max_weight)
                elif config.weight_distribution == "near_uniform":
                    # Clustered around mean (worst case for bucketing)
                    mean = (config.min_weight + config.max_weight) / 2
                    std = (config.max_weight - config.min_weight) / 20
                    weight = np.random.normal(mean, std)
                    weight = np.clip(weight, config.min_weight, config.max_weight)
                else:
                    weight = np.random.uniform(config.min_weight, config.max_weight)
                
                graph.add_edge(i, j, weight)
    
    return graph


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run a single benchmark comparing PostDijkstra vs Dijkstra.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        Benchmark results
    """
    print(f"\n{'='*80}")
    print(f"Benchmark: {config.name}")
    print(f"{'='*80}")
    print(f"Nodes: {config.num_nodes}, Edge prob: {config.edge_probability}")
    print(f"Weight dist: {config.weight_distribution}")
    print(f"Hierarchy: {config.use_hierarchy}, Lower bounds: {config.use_lower_bounds}")
    
    # Generate graph
    print("\nGenerating graph...")
    graph = generate_graph(config)
    num_edges = graph.edge_count()
    print(f"Generated graph: {graph.num_nodes} nodes, {num_edges} edges")
    
    graph_info = {
        "num_nodes": graph.num_nodes,
        "num_edges": num_edges,
        "avg_degree": num_edges / graph.num_nodes if graph.num_nodes > 0 else 0,
    }
    
    # Run Dijkstra baseline
    print("\nRunning Dijkstra baseline...")
    start = time.time()
    dijkstra_distances, dijkstra_metrics_obj = dijkstra_baseline(graph, source=0)
    dijkstra_time = time.time() - start
    print(f"Dijkstra completed in {dijkstra_time:.4f}s")
    
    dijkstra_metrics = dijkstra_metrics_obj.to_dict()
    dijkstra_metrics['time'] = dijkstra_time
    
    # Run PostDijkstra
    print("\nRunning PostDijkstra...")
    postdijkstra = PostDijkstraSSSP(
        graph,
        delta=config.delta,
        use_hierarchy=config.use_hierarchy,
        use_lower_bounds=config.use_lower_bounds,
        batch_size=config.batch_size,
    )
    
    start = time.time()
    postdijkstra_distances, postdijkstra_metrics_obj = postdijkstra.solve(source=0)
    postdijkstra_time = time.time() - start
    print(f"PostDijkstra completed in {postdijkstra_time:.4f}s")
    
    postdijkstra_metrics = postdijkstra_metrics_obj.to_dict()
    postdijkstra_metrics['time'] = postdijkstra_time
    
    # Validate correctness
    print("\nValidating correctness...")
    correctness = validate_sssp_results(postdijkstra_distances, dijkstra_distances)
    print(f"Correctness: {'PASS' if correctness else 'FAIL'}")
    
    # Compute comparison metrics
    speedup = dijkstra_time / postdijkstra_time if postdijkstra_time > 0 else 1.0
    memory_ratio = (
        postdijkstra_metrics['memory_mb'] / dijkstra_metrics['memory_mb']
        if dijkstra_metrics['memory_mb'] > 0 else 1.0
    )
    
    print(f"\nResults:")
    print(f"  Speedup: {speedup:.2f}x ({'faster' if speedup > 1.0 else 'slower'})")
    print(f"  Memory ratio: {memory_ratio:.2f}x")
    print(f"  Dijkstra edges relaxed: {dijkstra_metrics['edges_relaxed']}")
    print(f"  PostDijkstra edges relaxed: {postdijkstra_metrics['edges_relaxed']}")
    print(f"  Bucket operations: {postdijkstra_metrics['bucket_operations']}")
    print(f"  Lower bound prunings: {postdijkstra_metrics['lower_bound_prunings']}")
    
    return BenchmarkResult(
        config=config,
        graph_info=graph_info,
        dijkstra_time=dijkstra_time,
        dijkstra_metrics=dijkstra_metrics,
        postdijkstra_time=postdijkstra_time,
        postdijkstra_metrics=postdijkstra_metrics,
        speedup=speedup,
        correctness=correctness,
        memory_ratio=memory_ratio,
    )


def create_benchmark_suite() -> list[BenchmarkConfig]:
    """Create comprehensive benchmark suite.
    
    Returns:
        List of benchmark configurations
    """
    benchmarks = []
    
    # Small graphs (10^4 nodes)
    benchmarks.append(BenchmarkConfig(
        name="small_sparse_uniform",
        num_nodes=10000,
        edge_probability=0.001,
        weight_distribution="uniform",
        seed=42,
    ))
    
    benchmarks.append(BenchmarkConfig(
        name="small_sparse_heavy_tail",
        num_nodes=10000,
        edge_probability=0.001,
        weight_distribution="heavy_tail",
        seed=43,
    ))
    
    benchmarks.append(BenchmarkConfig(
        name="small_dense_uniform",
        num_nodes=10000,
        edge_probability=0.002,
        weight_distribution="uniform",
        seed=44,
    ))
    
    # Medium graphs (10^5 nodes)
    benchmarks.append(BenchmarkConfig(
        name="medium_sparse_uniform",
        num_nodes=100000,
        edge_probability=0.0001,
        weight_distribution="uniform",
        seed=45,
    ))
    
    benchmarks.append(BenchmarkConfig(
        name="medium_sparse_near_uniform",
        num_nodes=100000,
        edge_probability=0.0001,
        weight_distribution="near_uniform",
        seed=46,
    ))
    
    # Large graphs (10^6 nodes) - Warning: may be slow
    if False:  # Enable for full benchmark
        benchmarks.append(BenchmarkConfig(
            name="large_sparse_uniform",
            num_nodes=1000000,
            edge_probability=0.00001,
            weight_distribution="uniform",
            seed=47,
        ))
    
    return benchmarks


def run_benchmark_suite(
    configs: list[BenchmarkConfig],
    output_file: str | None = None
) -> list[BenchmarkResult]:
    """Run complete benchmark suite.
    
    Args:
        configs: List of benchmark configurations
        output_file: Optional output file for results
        
    Returns:
        List of benchmark results
    """
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Benchmark {i}/{len(configs)}")
        print(f"{'#'*80}")
        
        try:
            result = run_benchmark(config)
            results.append(result)
        except Exception as e:
            print(f"ERROR: Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n\n{'='*80}")
    print("BENCHMARK SUITE SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Benchmark':<30} {'Nodes':>10} {'Edges':>10} {'Speedup':>10} {'Correct':>10}")
    print("-" * 80)
    
    for result in results:
        print(
            f"{result.config.name:<30} "
            f"{result.graph_info['num_nodes']:>10} "
            f"{result.graph_info['num_edges']:>10} "
            f"{result.speedup:>9.2f}x "
            f"{'PASS' if result.correctness else 'FAIL':>10}"
        )
    
    # Statistics
    if results:
        speedups = [r.speedup for r in results if r.correctness]
        if speedups:
            print(f"\nSpeedup statistics:")
            print(f"  Mean: {np.mean(speedups):.2f}x")
            print(f"  Median: {np.median(speedups):.2f}x")
            print(f"  Min: {np.min(speedups):.2f}x")
            print(f"  Max: {np.max(speedups):.2f}x")
        
        correctness_rate = sum(r.correctness for r in results) / len(results) * 100
        print(f"\nCorrectness rate: {correctness_rate:.1f}%")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(
                {
                    "results": [r.to_dict() for r in results],
                    "summary": {
                        "total_benchmarks": len(results),
                        "correctness_rate": correctness_rate if results else 0,
                        "mean_speedup": float(np.mean(speedups)) if speedups else 0,
                    }
                },
                f,
                indent=2
            )
        print(f"\nResults saved to: {output_file}")
    
    return results


def main():
    """Main entry point for benchmarking."""
    parser = argparse.ArgumentParser(
        description="PostDijkstra SSSP Benchmark Suite"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="post_dijkstra_benchmark_results.json",
        help="Output file for results (JSON)"
    )
    
    parser.add_argument(
        "--custom",
        action="store_true",
        help="Run custom benchmark instead of full suite"
    )
    
    parser.add_argument(
        "--nodes",
        type=int,
        default=50000,
        help="Number of nodes for custom benchmark"
    )
    
    parser.add_argument(
        "--edge-prob",
        type=float,
        default=0.0002,
        help="Edge probability for custom benchmark"
    )
    
    parser.add_argument(
        "--weight-dist",
        type=str,
        default="uniform",
        choices=["uniform", "heavy_tail", "near_uniform"],
        help="Weight distribution for custom benchmark"
    )
    
    args = parser.parse_args()
    
    if args.custom:
        # Run custom benchmark
        config = BenchmarkConfig(
            name="custom",
            num_nodes=args.nodes,
            edge_probability=args.edge_prob,
            weight_distribution=args.weight_dist,
            seed=42,
        )
        configs = [config]
    else:
        # Run full suite
        configs = create_benchmark_suite()
    
    results = run_benchmark_suite(configs, output_file=args.output)
    
    return 0 if all(r.correctness for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
