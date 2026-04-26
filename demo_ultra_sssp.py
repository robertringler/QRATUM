#!/usr/bin/env python3
"""Demo script for UltraSSSP Large-Scale SSSP Simulation on QRATUM.

This script demonstrates the UltraSSSP algorithm adapted for QRATUM's
computational stack, including:
- QRATUM-native graph structures
- Adaptive frontier clustering
- Hierarchical graph contraction
- Performance benchmarking
- Validation against classical Dijkstra

Usage:
    python demo_ultra_sssp.py [--nodes NUM] [--edge-prob PROB] [--batch-size SIZE]
"""

import argparse
import json
import sys
from pathlib import Path

# Add quasim to path
sys.path.insert(0, str(Path(__file__).parent))

from quasim.opt.ultra_sssp import SSSPSimulationConfig, run_sssp_simulation


def main():
    """Run UltraSSSP demonstration."""
    parser = argparse.ArgumentParser(description="UltraSSSP Large-Scale SSSP Simulation for QRATUM")

    # Graph parameters
    parser.add_argument(
        "--nodes", type=int, default=1000, help="Number of nodes in graph (default: 1000)"
    )
    parser.add_argument(
        "--edge-prob", type=float, default=0.01, help="Edge probability (default: 0.01)"
    )
    parser.add_argument(
        "--max-weight", type=float, default=10.0, help="Maximum edge weight (default: 10.0)"
    )
    parser.add_argument("--source", type=int, default=0, help="Source node id (default: 0)")

    # Algorithm parameters
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Frontier batch size (default: 100)"
    )
    parser.add_argument(
        "--use-hierarchy", action="store_true", help="Enable hierarchical graph contraction"
    )
    parser.add_argument(
        "--hierarchy-levels", type=int, default=3, help="Number of hierarchy levels (default: 3)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--no-validate", action="store_true", help="Skip validation against Dijkstra baseline"
    )

    # Output options
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file for results (optional)"
    )

    args = parser.parse_args()

    # Create configuration
    config = SSSPSimulationConfig(
        num_nodes=args.nodes,
        edge_probability=args.edge_prob,
        max_edge_weight=args.max_weight,
        source_node=args.source,
        batch_size=args.batch_size,
        use_hierarchy=args.use_hierarchy,
        hierarchy_levels=args.hierarchy_levels,
        seed=args.seed,
        validate_against_dijkstra=not args.no_validate,
    )

    # Print configuration
    print("=" * 80)
    print("UltraSSSP Simulation on QRATUM")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Nodes: {config.num_nodes}")
    print(f"  Edge probability: {config.edge_probability}")
    print(f"  Max edge weight: {config.max_edge_weight}")
    print(f"  Source node: {config.source_node}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Use hierarchy: {config.use_hierarchy}")
    if config.use_hierarchy:
        print(f"  Hierarchy levels: {config.hierarchy_levels}")
    print(f"  Validate: {config.validate_against_dijkstra}")
    print(f"  Seed: {config.seed}")
    print()

    # Run simulation
    try:
        results = run_sssp_simulation(config)

        # Print summary
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)

        print("\nGraph:")
        print(f"  Nodes: {results['graph_info']['num_nodes']}")
        print(f"  Edges: {results['graph_info']['num_edges']}")

        print("\nUltraSSSP Metrics:")
        ultra = results["ultra_sssp_metrics"]
        print(f"  Total time: {ultra['total_time']:.4f}s")
        print(f"  Avg iteration time: {ultra['avg_iteration_time']:.6f}s")
        print(f"  Memory usage: {ultra['memory_mb']:.2f} MB")
        print(f"  Nodes visited: {ultra['nodes_visited']}")
        print(f"  Edges relaxed: {ultra['edges_relaxed']}")
        print(f"  Avg frontier size: {ultra['avg_frontier_size']:.1f}")

        if config.validate_against_dijkstra:
            print("\nValidation:")
            print(f"  Correctness: {'PASS' if results['correctness'] else 'FAIL'}")
            print(f"  Performance ratio vs Dijkstra: {results['speedup']:.2f}x")

            print("\nDijkstra Baseline Metrics:")
            dijkstra = results["dijkstra_metrics"]
            print(f"  Total time: {dijkstra['total_time']:.4f}s")
            print(f"  Nodes visited: {dijkstra['nodes_visited']}")
            print(f"  Edges relaxed: {dijkstra['edges_relaxed']}")

        # Sample distances
        distances = results["distances"]
        reachable = [d for d in distances if d != float("inf")]
        if reachable:
            print("\nDistance Statistics:")
            print(f"  Reachable nodes: {len(reachable)}/{len(distances)}")
            print(f"  Min distance: {min(reachable):.2f}")
            print(f"  Max distance: {max(reachable):.2f}")
            print(f"  Avg distance: {sum(reachable)/len(reachable):.2f}")

        # Save results if requested
        if args.output:
            output_data = {
                "config": {
                    "num_nodes": config.num_nodes,
                    "edge_probability": config.edge_probability,
                    "max_edge_weight": config.max_edge_weight,
                    "source_node": config.source_node,
                    "batch_size": config.batch_size,
                    "use_hierarchy": config.use_hierarchy,
                    "hierarchy_levels": config.hierarchy_levels,
                    "seed": config.seed,
                },
                "results": {
                    "distances": distances,
                    "ultra_sssp_metrics": ultra,
                    "dijkstra_metrics": results.get("dijkstra_metrics"),
                    "correctness": results.get("correctness"),
                    "speedup": results.get("speedup"),
                    "graph_info": results["graph_info"],
                    "iteration_count": (
                        len(ultra.get("avg_iteration_time", []))
                        if isinstance(ultra.get("avg_iteration_time"), list)
                        else ultra.get("nodes_visited", 0)
                    ),
                },
            }

            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")

        print("\n" + "=" * 80)
        print("Simulation completed successfully!")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
