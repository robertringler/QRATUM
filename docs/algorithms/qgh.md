# QGH Non-Speculative Algorithms

## Overview

QGH (Quantum Graph Hash) provides non-speculative algorithms for distributed quantum systems, including causal history tracking, superposition resolution, distributed monitoring, consensus propagation, and stability monitoring.

## Core Algorithms

### 1. CausalHistoryHash

Maintains hash-based causal history of quantum events with deterministic ordering.

**Features**:

- SHA-256 hashing of events
- FIFO history with configurable size
- Causality verification

**Use Case**: Track quantum measurement sequences, ensure causal ordering in distributed systems.

**Example**:

```python
from quasim.qgh import CausalHistoryHash

chh = CausalHistoryHash(history_size=1000)
chh.add_event("measurement_1", np.array([0.5, 0.3, 0.2]))
assert chh.verify_causality("measurement_1")
```

### 2. SuperpositionResolver

Resolves quantum superpositions through iterative consistency checks.

**Algorithm**:

1. Start with initial superposed state
2. Apply consistency function iteratively
3. Check convergence: `||s_t - s_{t-1}|| < tolerance`
4. Return converged state or max iterations

**Math**:

- State update: `s_{t+1} = consistency_fn(s_t)`
- Convergence: `||s_{t+1} - s_t|| < ε`

**Use Case**: Collapse quantum states to classical observations, distributed consensus.

**Example**:

```python
from quasim.qgh import SuperpositionResolver

resolver = SuperpositionResolver(max_iterations=100, tolerance=1e-6)
state = np.array([0.5, 0.3, 0.15, 0.05])

result = resolver.resolve(state, lambda x: x / np.sum(x))
print(f"Converged: {result['converged']}, Iterations: {result['iterations']}")
```

### 3. DistributedStreamMonitor

Monitors multiple data streams with synchronization detection.

**Features**:

- Per-stream circular buffers
- Real-time statistics
- Correlation-based sync detection

**Math**: For streams `i` and `j`:

- Correlation: `ρ_{ij} = cov(s_i, s_j) / (σ_i * σ_j)`
- Synchronized if: `|ρ_{ij}| > threshold`

**Use Case**: Monitor distributed sensors, detect synchronized failures.

**Example**:

```python
from quasim.qgh import DistributedStreamMonitor

monitor = DistributedStreamMonitor(num_streams=4, buffer_size=1000)

for i in range(100):
    monitor.add_sample(stream_id=0, value=np.random.normal())
    monitor.add_sample(stream_id=1, value=np.random.normal())

sync_pairs = monitor.detect_sync_patterns(threshold=0.7)
print(f"Synchronized pairs: {sync_pairs}")
```

### 4. SelfConsistencyPropagator

Propagates self-consistency constraints through quantum network.

**Algorithm**:

1. Initialize node states
2. For each iteration:
   - For each node: average neighbor states
   - Apply damped update: `s_i' = (1-d)*s_i + d*avg(neighbors)`
3. Check convergence

**Math**:

- Message passing: `m_{ij} = s_j`
- Update: `s_i^{t+1} = (1-λ)s_i^t + λ * (Σ_j A_{ij} s_j^t) / deg(i)`
- Damping factor: `λ ∈ [0,1]`

**Use Case**: Federated learning, distributed consensus, network synchronization.

**Example**:

```python
from quasim.qgh import SelfConsistencyPropagator

prop = SelfConsistencyPropagator(num_nodes=10, damping=0.5)
states = np.random.rand(10, 5)

result = prop.propagate(states)
print(f"Consensus: {result['converged']}, Variance: {result['final_variance']:.4f}")
```

### 5. StabilityMonitor

Monitors system stability and detects instabilities.

**Algorithm**:

1. Maintain sliding window of metrics
2. Fit linear trend: `y = mx + b`
3. Compute normalized slope: `m' = |m| / σ`
4. Unstable if: `m' > threshold`

**Math**:

- Trend detection: `slope = polyfit(x, y, degree=1)[0]`
- Normalized: `slope_norm = |slope| / std(y)`
- Stable if: `slope_norm ≤ threshold`

**Use Case**: Detect system divergence, monitor portfolio volatility, quality control.

**Example**:

```python
from quasim.qgh import StabilityMonitor

monitor = StabilityMonitor(window_size=50, threshold=2.0)

for i in range(100):
    monitor.add_metric(np.random.normal(1.0, 0.1))

print(f"Stable: {monitor.is_stable()}")
stats = monitor.get_stats()
print(f"Trend: {stats['trend']:.4f}")
```

## CLI Usage

### Tensor Processing Demo

```bash
quasim-qgh demo --section tensor --export tensor_results.json
```

### Distributed Monitoring Demo

```bash
quasim-qgh demo --section distributed
```

### Federated Consensus Demo

```bash
quasim-qgh demo --section consensus
```

### AV Sensor Fusion Demo

```bash
quasim-qgh demo --section fusion
```

### Resolve Superposition

```bash
quasim-qgh resolve --state "0.5,0.3,0.15,0.05" --iterations 50
```

### Monitor Streams

```bash
quasim-qgh monitor --num-streams 4 --samples 100 --threshold 0.7
```

## Practical Examples

### Example 1: Federated Learning Consensus

```python
from quasim.qgh import SelfConsistencyPropagator

# 5 nodes, each with different model parameters
num_nodes = 5
param_dim = 100

# Initialize with random parameters
local_params = np.random.normal(0, 1, size=(num_nodes, param_dim))

# Propagate to consensus
propagator = SelfConsistencyPropagator(
    num_nodes=num_nodes,
    damping=0.5,
    max_iterations=100
)

result = propagator.propagate(local_params)

if result['converged']:
    global_params = np.mean(result['states'], axis=0)
    print(f"Global model converged in {result['iterations']} iterations")
```

### Example 2: AV Sensor Fusion

```python
from quasim.qgh import DistributedStreamMonitor

# Monitor LIDAR, RADAR, Camera, GPS
monitor = DistributedStreamMonitor(num_streams=4, buffer_size=1000)

# Simulate sensor readings
for t in range(200):
    true_pos = 10.0 + 0.1 * t
    monitor.add_sample(0, true_pos + np.random.normal(0, 0.5))  # LIDAR
    monitor.add_sample(1, true_pos + np.random.normal(0, 0.8))  # RADAR
    monitor.add_sample(2, true_pos + np.random.normal(0, 1.0))  # Camera
    monitor.add_sample(3, true_pos + np.random.normal(0, 2.0))  # GPS

# Fuse with weighted average (inverse variance weighting)
stats = [monitor.get_stream_stats(i) for i in range(4)]
weights = np.array([1.0 / s['std']**2 for s in stats])
weights /= np.sum(weights)

fused = sum(s['mean'] * w for s, w in zip(stats, weights))
print(f"Fused position: {fused:.2f}")
```

### Example 3: Portfolio Stability Monitoring

```python
from quasim.qgh import StabilityMonitor

monitor = StabilityMonitor(window_size=30, threshold=1.5)

# Simulate daily returns
for day in range(100):
    volatility = 1.0 + 0.05 * day  # Increasing volatility
    daily_return = np.random.normal(0.001, volatility * 0.01)
    monitor.add_metric(abs(daily_return))

if not monitor.is_stable():
    print("WARNING: Portfolio volatility increasing - recommend risk reduction")
```

## TERC Integration

QGH algorithms provide observables for TERC validation:

- **Consensus status**: Convergence metrics → Tier-1 robustness
- **Stream synchronization**: Sync ratios → Tier-5 distributed consistency
- **Stability metrics**: Trend detection → System health monitoring

See `docs/terc_bridge.md` for integration details.

## Serialization

All QGH algorithms support `to_dict()` / `from_dict()` for checkpointing:

```python
# Save state
chh = CausalHistoryHash(history_size=100)
state = chh.to_dict()
with open('checkpoint.json', 'w') as f:
    json.dump(state, f)

# Restore state
with open('checkpoint.json') as f:
    state = json.load(f)
restored = CausalHistoryHash.from_dict(state)
```

## Performance Notes

- **CausalHistoryHash**: O(1) add, O(1) verify
- **SuperpositionResolver**: O(n * iterations) where n = state_dim
- **DistributedStreamMonitor**: O(streams * buffer_size) for sync detection
- **SelfConsistencyPropagator**: O(nodes² *state_dim* iterations)
- **StabilityMonitor**: O(window_size) for stability check
