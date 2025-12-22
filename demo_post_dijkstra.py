#!/usr/bin/env python3
"""Demo script for PostDijkstra SSSP algorithm.

This demonstrates the PostDijkstra shortest-path algorithm that escapes
Dijkstra's priority-queue bottleneck through multi-axis optimization.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from quasim.opt.graph import QGraph
from quasim.opt.post_dijkstra_sssp import PostDijkstraSSSP
from quasim.opt.ultra_sssp import dijkstra_baseline, validate_sssp_results


def main():
    """Run PostDijkstra demonstration."""
    parser = argparse.ArgumentParser(
        description="PostDijkstra SSSP Algorithm Demonstration"
    )
    
    # Graph parameters
    parser.add_argument(
        "--nodes",
        type=int,
        default=1000,
        help="Number of nodes (default: 1000)"
    )
    parser.add_argument(
        "--edge-prob",
        type=float,
        default=0.01,
        help="Edge probability (default: 0.01)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    # Algorithm parameters
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Bucket width (default: auto-compute)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for parallel relaxations (default: 100)"
    )
    parser.add_argument(
        "--no-hierarchy",
        action="store_true",
        help="Disable hierarchical decomposition"
    )
    parser.add_argument(
        "--no-lower-bounds",
        action="store_true",
        help="Disable lower-bound pruning"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation against Dijkstra"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PostDijkstra SSSP Demonstration")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Nodes: {args.nodes}")
    print(f"  Edge probability: {args.edge_prob}")
    print(f"  Delta: {args.delta if args.delta else 'auto'}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Hierarchy: {not args.no_hierarchy}")
    print(f"  Lower bounds: {not args.no_lower_bounds}")
    print(f"  Seed: {args.seed}")
    
    # Generate graph
    print(f"\nGenerating random graph...")
    graph = QGraph.random_graph(
        num_nodes=args.nodes,
        edge_probability=args.edge_prob,
        seed=args.seed
    )
    num_edges = graph.edge_count()
    print(f"Generated: {num_edges} edges (avg degree: {num_edges/args.nodes:.2f})")
    
    # Run PostDijkstra
    print(f"\nRunning PostDijkstra algorithm...")
    sssp = PostDijkstraSSSP(
        graph,
        delta=args.delta,
        use_hierarchy=not args.no_hierarchy,
        use_lower_bounds=not args.no_lower_bounds,
        batch_size=args.batch_size
    )
    
    start_time = time.time()
    distances, metrics = sssp.solve(source=0)
    elapsed = time.time() - start_time
    
    print(f"PostDijkstra completed in {elapsed:.4f}s")
    print(f"\nMetrics:")
    print(f"  Nodes visited: {metrics.nodes_visited}")
    print(f"  Edges relaxed: {metrics.edges_relaxed}")
    print(f"  Bucket operations: {metrics.bucket_operations}")
    print(f"  Lower bound prunings: {metrics.lower_bound_prunings}")
    print(f"  Parallel batches: {metrics.parallel_batches}")
    print(f"  Memory: {metrics.memory_bytes / (1024*1024):.2f} MB")
    
    # Validate if requested
    if not args.no_validate:
        print(f"\nRunning Dijkstra baseline for validation...")
        start_time = time.time()
        dij_distances, dij_metrics = dijkstra_baseline(graph, source=0)
        dij_elapsed = time.time() - start_time
        
        correctness = validate_sssp_results(distances, dij_distances)
        speedup = dij_elapsed / elapsed if elapsed > 0 else 1.0
        
        print(f"Dijkstra completed in {dij_elapsed:.4f}s")
        print(f"\nValidation:")
        print(f"  Correctness: {'PASS' if correctness else 'FAIL'}")
        print(f"  Performance ratio: {speedup:.2f}x")
        
        if not correctness:
            print(f"\nWARNING: Results do not match Dijkstra baseline!")
            print(f"This may indicate a bug or numerical precision issue.")
    
    # Distance statistics
    reachable = [d for d in distances if d != float('inf')]
    if reachable:
        print(f"\nDistance statistics:")
        print(f"  Reachable nodes: {len(reachable)}/{len(distances)}")
        print(f"  Min distance: {min(reachable):.4f}")
        print(f"  Max distance: {max(reachable):.4f}")
        print(f"  Avg distance: {sum(reachable)/len(reachable):.4f}")
    
    # Save results if requested
    if args.output:
        results = {
            "config": {
                "num_nodes": args.nodes,
                "edge_probability": args.edge_prob,
                "delta": args.delta,
                "batch_size": args.batch_size,
                "use_hierarchy": not args.no_hierarchy,
                "use_lower_bounds": not args.no_lower_bounds,
                "seed": args.seed,
            },
            "graph_info": {
                "num_nodes": graph.num_nodes,
                "num_edges": num_edges,
            },
            "metrics": metrics.to_dict(),
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print("\n" + "=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
