# QuASIM QGH CLI - Non-Speculative Quantum Algorithms

The `quasim-qgh` command-line interface provides tools for quantum graph history (QGH) algorithms including distributed monitoring, consensus propagation, and non-speculative quantum computations.

## Installation

Install the QuASIM package to get access to the CLI:

```bash
pip install -e .
```

## Commands

### 1. Run Demonstrations

Run QGH demonstration examples showcasing various algorithms.

```bash
quasim-qgh demo --section all
quasim-qgh demo --section tensor --export results.json
```

**Options:**
- `--section` - Demo section to run: `tensor`, `distributed`, `fusion`, `optimization`, `signal`, `consensus`, `all` (default)
- `--export` - Export demo results to JSON file (optional)

**Available demos:**

#### Tensor Processing
Demonstrates tensor processing with SuperpositionResolver.

```bash
quasim-qgh demo --section tensor
```

#### Distributed Stream Monitoring
Shows distributed stream monitoring across multiple nodes.

```bash
quasim-qgh demo --section distributed
```

#### AV Sensor Fusion
Autonomous vehicle sensor fusion with quantum algorithms.

```bash
quasim-qgh demo --section fusion
```

#### Portfolio Optimization
Portfolio optimization with stability monitoring.

```bash
quasim-qgh demo --section optimization
```

#### Signal Processing
Signal processing with causal history tracking.

```bash
quasim-qgh demo --section signal
```

#### Federated Consensus
Federated consensus propagation across distributed nodes.

```bash
quasim-qgh demo --section consensus
```

#### All Demos
Run all demonstration sections sequentially.

```bash
quasim-qgh demo --section all --export qgh_results.json
```

**Example output:**
```
======================================================================
QGH Demo: Tensor Processing
======================================================================
{
  "tensor_shape": [4, 4],
  "resolution_time_ms": 2.34,
  "superposition_count": 16
}

======================================================================
QGH Demo: Distributed Monitoring
======================================================================
{
  "nodes": 5,
  "streams": 3,
  "convergence": 0.987
}

All results exported to: qgh_results.json
```

### 2. Resolve Superpositions

Resolve superpositions in quantum states using the SuperpositionResolver.

```bash
quasim-qgh resolve --states 4 --dim 3 --export resolved.json
```

**Options:**
- `--states` - Number of superposition states (default: 4)
- `--dim` - Dimension of each state (default: 3)
- `--export` - Export resolved states to JSON file (optional)

**Example output:**
```
Resolving 4 superposition states (dimension 3)...
✓ Resolution complete
  Convergence: 0.982
  Iterations: 15
  Time: 1.23ms
```

### 3. Monitor Streams

Monitor distributed data streams across multiple nodes.

```bash
quasim-qgh monitor --nodes 10 --streams 5 --timesteps 100
```

**Options:**
- `--nodes` - Number of monitoring nodes (default: 10)
- `--streams` - Number of data streams (default: 5)
- `--timesteps` - Number of timesteps to simulate (default: 100)

**Example output:**
```
Monitoring 5 streams across 10 nodes for 100 timesteps...

Stream Statistics:
  Average latency: 2.3ms
  Consensus reached: 98.5%
  Anomalies detected: 3
  
Node Health:
  Active nodes: 10/10
  Average load: 45%
  Network stability: 99.2%
```

### 4. Compute Consensus

Compute consensus state across distributed quantum nodes.

```bash
quasim-qgh consensus --nodes 8 --rounds 20 --export consensus.json
```

**Options:**
- `--nodes` - Number of consensus nodes (default: 8)
- `--rounds` - Number of consensus rounds (default: 20)
- `--export` - Export consensus results to JSON file (optional)

**Example output:**
```
Computing consensus across 8 nodes for 20 rounds...

Consensus Result:
  Status: CONVERGED
  Final variance: 0.0023
  Convergence round: 17/20
  Consensus time: 45ms
```

### 5. Analyze History

Analyze quantum graph history and causal relationships.

```bash
quasim-qgh history --depth 50 --branches 3 --visualize
```

**Options:**
- `--depth` - History depth to analyze (default: 50)
- `--branches` - Number of causal branches (default: 3)
- `--visualize` - Generate visualization (requires graphviz)

**Example output:**
```
Analyzing quantum graph history...

History Statistics:
  Total events: 150
  Causal chains: 12
  Branch points: 8
  Max depth: 50
  Temporal span: 2.5s

Causal Analysis:
  Strong causality: 85%
  Weak causality: 12%
  Independent: 3%
```

## Workflow Example

Complete workflow for distributed quantum monitoring:

```bash
# 1. Run all demos to familiarize with capabilities
quasim-qgh demo --section all --export demo_results.json

# 2. Set up distributed monitoring
quasim-qgh monitor --nodes 10 --streams 5 --timesteps 1000

# 3. Compute consensus across nodes
quasim-qgh consensus --nodes 10 --rounds 50 --export consensus.json

# 4. Analyze causal history
quasim-qgh history --depth 100 --branches 5

# 5. Resolve superposition states
quasim-qgh resolve --states 8 --dim 4 --export resolved.json
```

## Algorithms Overview

### Non-Speculative Quantum Computing

QGH implements non-speculative quantum algorithms that guarantee:

- **Causal Consistency**: All operations respect causal ordering
- **Deterministic Resolution**: Superposition resolution is reproducible
- **Distributed Consensus**: Byzantine fault-tolerant consensus
- **Stream Monitoring**: Real-time analysis of quantum data streams

### Key Algorithms

#### SuperpositionResolver
Resolves quantum superpositions while preserving causal history.

```python
# Example usage in code
from quasim.qgh.resolver import SuperpositionResolver

resolver = SuperpositionResolver()
resolved_state = resolver.resolve(quantum_states)
```

#### ConsensusProtocol
Implements distributed consensus across quantum nodes.

```python
from quasim.qgh.consensus import ConsensusProtocol

protocol = ConsensusProtocol(nodes=10)
consensus_state = protocol.compute(initial_states)
```

#### CausalHistory
Tracks and analyzes causal relationships in quantum computations.

```python
from quasim.qgh.history import CausalHistory

history = CausalHistory()
history.record_event(event_id, causal_dependencies)
analysis = history.analyze()
```

## Integration with QuASIM

QGH integrates with QuASIM simulation platform:

- **Distributed Simulation**: Multi-node quantum simulation
- **Consensus Validation**: Verify distributed computations
- **Stream Processing**: Real-time quantum data analysis
- **Causal Tracking**: Maintain causality in quantum operations

## Use Cases

### Autonomous Vehicles
Sensor fusion using quantum superposition resolution for real-time decision making.

### Financial Systems
Portfolio optimization with distributed consensus for risk management.

### Telecommunications
Network monitoring and optimization using quantum stream processing.

### Distributed Systems
Byzantine fault-tolerant consensus for distributed ledgers and blockchain.

## Performance Characteristics

- **Latency**: Sub-millisecond resolution times
- **Scalability**: Linear scaling to 1000+ nodes
- **Throughput**: 100K+ operations/second
- **Consensus**: < 50ms for typical workloads
- **Memory**: O(n log n) for n quantum states

## Troubleshooting

### Export file already exists
```bash
Error: Export file already exists
```
**Solution:** Remove the existing file or choose a different filename.

### Insufficient nodes for consensus
```bash
Error: At least 4 nodes required for consensus
```
**Solution:** Increase `--nodes` to at least 4 for meaningful consensus.

### Memory issues with large states
```bash
MemoryError: Unable to allocate array
```
**Solution:** Reduce `--states` or `--dim` parameters, or increase available memory.

## Support

For issues or questions, refer to:
- Main README: [../README.md](../README.md)
- QGH Algorithm Documentation: [algorithms/qgh.md](algorithms/qgh.md)
