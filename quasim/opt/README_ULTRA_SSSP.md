# UltraSSSP: Large-Scale Single-Source Shortest Path for QRATUM

## Overview

UltraSSSP is a high-performance shortest path algorithm implementation adapted for QRATUM's computational stack. It provides:

- **QRATUM-native graph structures** (`QGraph`) optimized for large-scale path finding
- **Adaptive frontier clustering** for efficient batch processing
- **Hierarchical graph contraction** for multi-level optimization
- **Quantum pivot selection hooks** (placeholders for QPU integration)
- **Comprehensive benchmarking** and validation against classical Dijkstra

## Features

### 1. Graph Data Structures

#### QGraph

Memory-efficient adjacency list representation supporting:

- Directed and undirected graphs
- Non-negative edge weights
- Fast neighbor iteration
- Random graph generation

```python
from quasim.opt.graph import QGraph

# Create graph
graph = QGraph(num_nodes=100)
graph.add_edge(0, 1, weight=5.0)

# Generate random graph
graph = QGraph.random_graph(
    num_nodes=1000,
    edge_probability=0.01,
    seed=42
)
```

#### HierarchicalGraph

Multi-level graph hierarchy for contraction-based optimization:

- Automatic graph coarsening
- Supernode mapping
- Efficient level traversal

```python
from quasim.opt.graph import HierarchicalGraph

hierarchy = HierarchicalGraph.from_contraction(
    base_graph=graph,
    num_levels=3,
    contraction_factor=0.5
)
```

### 2. UltraSSSP Algorithm

Advanced SSSP with:

- **Batch Processing**: Groups frontier nodes for parallel processing
- **Correctness Guarantee**: Maintains optimal path distances
- **Memory Efficiency**: Optimized data structures for large graphs
- **Performance Tracking**: Detailed metrics collection

```python
from quasim.opt.ultra_sssp import UltraSSSP

sssp = UltraSSSP(
    graph=graph,
    batch_size=100,
    use_hierarchy=False
)

distances, metrics = sssp.solve(source=0)
```

### 3. Validation and Benchmarking

Built-in comparison against classical Dijkstra:

```python
from quasim.opt.ultra_sssp import dijkstra_baseline, validate_sssp_results

# Run baseline
dijkstra_distances, _ = dijkstra_baseline(graph, source=0)

# Validate
is_correct = validate_sssp_results(ultra_distances, dijkstra_distances)
```

### 4. Quantum Integration (Placeholder)

Hooks for future QPU integration:

```python
def quantum_pivot_function(candidates, distances):
    """Custom quantum pivot selection using QPU."""
    # TODO: Integrate with QRATUM QPU API
    # Use quantum amplitude amplification or similar
    return selected_node

sssp = UltraSSSP(
    graph=graph,
    quantum_pivot_fn=quantum_pivot_function
)
```

## Usage

### Basic Example

```python
from quasim.opt.graph import QGraph
from quasim.opt.ultra_sssp import UltraSSSP, dijkstra_baseline

# Create graph
graph = QGraph(num_nodes=100)
graph.add_edge(0, 1, 5.0)
graph.add_edge(1, 2, 3.0)

# Solve SSSP
sssp = UltraSSSP(graph, batch_size=20)
distances, metrics = sssp.solve(source=0)

print(f"Distance to node 2: {distances[2]}")
print(f"Nodes visited: {metrics.nodes_visited}")
print(f"Time: {metrics.total_time:.4f}s")
```

### Complete Simulation

```python
from quasim.opt.ultra_sssp import SSSPSimulationConfig, run_sssp_simulation

config = SSSPSimulationConfig(
    num_nodes=1000,
    edge_probability=0.01,
    batch_size=100,
    use_hierarchy=True,
    hierarchy_levels=3,
    validate_against_dijkstra=True
)

results = run_sssp_simulation(config)

print(f"Correctness: {results['correctness']}")
print(f"Speedup: {results['speedup']:.2f}x")
```

### Command-Line Demo

```bash
# Basic usage
python demo_ultra_sssp.py --nodes 1000 --edge-prob 0.01

# With hierarchical contraction
python demo_ultra_sssp.py --nodes 5000 --edge-prob 0.005 \
    --use-hierarchy --hierarchy-levels 3 --batch-size 200

# Save results
python demo_ultra_sssp.py --nodes 1000 --edge-prob 0.01 \
    --output results.json
```

## Configuration Parameters

### Graph Generation

- `num_nodes`: Number of nodes in graph (default: 1000)
- `edge_probability`: Probability of edge between any two nodes (default: 0.01)
- `max_edge_weight`: Maximum edge weight (default: 10.0)
- `source_node`: Source node for SSSP (default: 0)

### Algorithm Parameters

- `batch_size`: Frontier batch size (default: 100)
- `use_hierarchy`: Enable hierarchical contraction (default: False)
- `hierarchy_levels`: Number of hierarchy levels (default: 3)
- `seed`: Random seed for reproducibility (default: 42)

### Validation

- `validate_against_dijkstra`: Run validation (default: True)

## Performance Metrics

The algorithm collects comprehensive metrics:

```python
metrics = {
    "total_time": 0.245,          # Total execution time (s)
    "avg_iteration_time": 0.002,  # Average per iteration (s)
    "memory_mb": 15.3,            # Memory usage (MB)
    "nodes_visited": 998,         # Nodes processed
    "edges_relaxed": 9856,        # Edge relaxations
    "avg_frontier_size": 42.3,    # Average frontier size
    "correctness": True           # Validation result
}
```

## Architecture

### Module Structure

```
quasim/opt/
├── graph.py          # QGraph and HierarchicalGraph
├── ultra_sssp.py     # UltraSSSP algorithm
└── __init__.py       # Module exports

tests/opt/
├── test_graph.py     # Graph structure tests
└── test_ultra_sssp.py # Algorithm tests

demo_ultra_sssp.py    # Command-line demo
```

### Key Classes

- **QGraph**: QRATUM-native graph structure
- **HierarchicalGraph**: Multi-level graph hierarchy
- **UltraSSSP**: Main shortest path algorithm
- **SSSPMetrics**: Performance and validation metrics
- **SSSPSimulationConfig**: Configuration management

## Integration with QRATUM

### Current Integration

- Uses QRATUM's `quasim` module structure
- Compatible with QRATUM runtime configuration
- Follows QRATUM coding conventions

### Future Integration (TODO)

- [ ] QPU API integration for quantum pivot selection
- [ ] Integration with QRATUM distributed runtime
- [ ] Hardware-aware graph partitioning
- [ ] Quantum-classical hybrid optimization
- [ ] Integration with QRATUM telemetry system

## Testing

Run tests:

```bash
# Run all core tests
python -c "from tests.opt import test_graph, test_ultra_sssp; ..."

# Run demo with various configurations
python demo_ultra_sssp.py --nodes 100 --edge-prob 0.05
python demo_ultra_sssp.py --nodes 500 --use-hierarchy
```

Test coverage includes:

- Graph data structure operations
- Random graph generation
- Dijkstra baseline implementation
- UltraSSSP correctness validation
- Hierarchical graph contraction
- Complete simulation pipeline

## Algorithm Details

### Correctness Guarantee

UltraSSSP maintains correctness by:

1. Processing nodes in distance order (like Dijkstra)
2. Handling stale priority queue entries
3. Extracting batches with similar distances
4. Re-queuing nodes if distance improves

### Batch Processing

The algorithm extracts batches of nodes with similar distances from the frontier:

```python
def _extract_batch(frontier, distances, visited):
    # Find minimum distance
    min_dist = frontier[0][0]
    
    # Extract nodes within epsilon of minimum
    batch = []
    for dist, node in frontier:
        if dist <= min_dist + epsilon:
            batch.append((dist, node))
    
    return batch
```

### Memory Optimization

- Adjacency list representation (O(V + E) space)
- Efficient frontier management with heapq
- Lazy deletion of stale entries
- Optional hierarchical contraction

## Limitations

1. **Batch Size**: Small batches behave like Dijkstra (no speedup)
2. **Dense Graphs**: May not scale well for very dense graphs
3. **Negative Weights**: Not supported (inherent to Dijkstra-based algorithms)
4. **Quantum Features**: Currently placeholders only

## Future Work

### Quantum Enhancements

- Quantum amplitude amplification for pivot selection
- Quantum walk-based path finding
- Hybrid quantum-classical optimization

### Performance Optimizations

- GPU acceleration for batch processing
- Distributed graph partitioning
- Advanced frontier clustering algorithms

### Features

- Multiple source shortest paths (MSSP)
- All-pairs shortest paths (APSP)
- Dynamic graph updates
- Approximate distance queries

## References

- Dijkstra, E. W. (1959). "A note on two problems in connexion with graphs"
- Meyer, U., & Sanders, P. (2003). "Δ-stepping: a parallelizable shortest path algorithm"
- QRATUM Platform Documentation

## License

Part of the QRATUM project. See LICENSE file for details.

## Contact

For issues or questions, please refer to the QRATUM repository issue tracker.
