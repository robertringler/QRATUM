# Performance Tuning

Optimize QRATUM for your specific workload and hardware.

## Quick Wins

### 1. Reduce Shot Count During Development

```python
# Development (fast iteration)
config = QuantumConfig(shots=100)

# Production (accurate results)
config = QuantumConfig(shots=4096)
```

### 2. Use Deterministic Seeds

```python
# Avoid re-initialization overhead
config = QuantumConfig(seed=42)
```

### 3. Batch Computations

```python
# Instead of:
for r in bond_lengths:
    result = vqe.compute_h2_energy(bond_length=r)

# Use batching:
results = vqe.compute_h2_energy_batch(bond_lengths=bond_lengths)
```

## Backend Selection

### Performance Comparison

| Backend | Speed | Accuracy | Max Qubits | Use Case |
|---------|-------|----------|------------|----------|
| Qiskit Aer (statevector) | Fast | Exact | ~30 | Development |
| Qiskit Aer (MPS) | Medium | Approximate | ~50 | Large circuits |
| cuQuantum | Very Fast | Exact | ~30 | GPU available |
| IBM Quantum | Slow | Real | ~127 | Hardware validation |

### Selecting Backends

```python
# CPU simulation (default)
config = QuantumConfig(backend_type="simulator")

# GPU-accelerated (requires cuQuantum)
config = QuantumConfig(
    backend_type="cuquantum",
    device=0  # GPU index
)

# Real quantum hardware
config = QuantumConfig(
    backend_type="ibmq",
    ibmq_token="your_token"
)
```

## Optimizer Tuning

### Optimizer Comparison

| Optimizer | Speed | Robustness | Best For |
|-----------|-------|------------|----------|
| COBYLA | Fast | Good | Noisy landscapes |
| L-BFGS-B | Medium | Good | Smooth landscapes |
| SLSQP | Medium | Good | Constrained optimization |
| SPSA | Fast | Excellent | Quantum hardware |

### Configuration

```python
result = vqe.compute_h2_energy(
    bond_length=0.735,
    optimizer="COBYLA",      # Choose optimizer
    max_iterations=200,      # More iterations for better convergence
    tolerance=1e-8           # Tighter tolerance
)
```

### Custom Optimizer

```python
from scipy.optimize import minimize

def custom_optimizer(fun, x0, bounds=None):
    result = minimize(
        fun,
        x0,
        method='Powell',
        options={'maxiter': 500, 'xtol': 1e-8}
    )
    return result.x, result.fun

config = QuantumConfig(
    optimizer=custom_optimizer
)
```

## Circuit Optimization

### Ansatz Depth

```python
# Shallow ansatz (fast, less expressive)
result = vqe.compute_h2_energy(ansatz_layers=1)

# Deep ansatz (slow, more expressive)
result = vqe.compute_h2_energy(ansatz_layers=5)
```

### Circuit Compilation

```python
from qiskit import transpile

# Optimize for target backend
circuit = vqe.get_circuit()
optimized = transpile(
    circuit,
    optimization_level=3,  # Maximum optimization
    basis_gates=['cx', 'u3']
)
```

## Memory Optimization

### For Large Systems

```python
import os

# Limit memory usage
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
```

### Garbage Collection

```python
import gc

for r in bond_lengths:
    result = vqe.compute_h2_energy(bond_length=r)
    results.append(result.energy)
    gc.collect()  # Free memory between iterations
```

## Parallelization

### Parameter Sweeps

```python
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def compute_energy(bond_length):
    config = QuantumConfig(seed=42)
    vqe = MolecularVQE(config)
    result = vqe.compute_h2_energy(bond_length=bond_length)
    return result.energy

bond_lengths = np.linspace(0.4, 2.5, 100)

with ProcessPoolExecutor(max_workers=8) as executor:
    energies = list(executor.map(compute_energy, bond_lengths))
```

### Multi-GPU

```python
# Distribute across multiple GPUs
configs = [
    QuantumConfig(backend_type="cuquantum", device=0),
    QuantumConfig(backend_type="cuquantum", device=1),
    QuantumConfig(backend_type="cuquantum", device=2),
    QuantumConfig(backend_type="cuquantum", device=3),
]
```

## Caching

### Result Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_energy(bond_length: float, seed: int) -> float:
    config = QuantumConfig(seed=seed)
    vqe = MolecularVQE(config)
    result = vqe.compute_h2_energy(bond_length=bond_length)
    return result.energy
```

### Circuit Caching

```python
from qiskit.circuit import CircuitInstruction

# Cache compiled circuits
circuit_cache = {}

def get_circuit(n_qubits, depth):
    key = (n_qubits, depth)
    if key not in circuit_cache:
        circuit_cache[key] = build_circuit(n_qubits, depth)
    return circuit_cache[key].copy()
```

## Profiling

### Python Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your computation
result = vqe.compute_h2_energy(bond_length=0.735)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def run_vqe():
    config = QuantumConfig(seed=42)
    vqe = MolecularVQE(config)
    result = vqe.compute_h2_energy(bond_length=0.735)
    return result

run_vqe()
```

## Benchmarking

### Standard Benchmark

```bash
# Run built-in benchmarks
make bench
```

### Custom Benchmark

```python
import time
import numpy as np

def benchmark_vqe(n_runs=10):
    config = QuantumConfig(seed=42)
    vqe = MolecularVQE(config)
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = vqe.compute_h2_energy(bond_length=0.735)
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

print(benchmark_vqe())
```

## Production Recommendations

### Hardware Requirements

| Workload | CPU | RAM | GPU | Storage |
|----------|-----|-----|-----|---------|
| Development | 4 cores | 8 GB | Optional | 20 GB |
| Testing | 8 cores | 16 GB | Optional | 50 GB |
| Production | 16+ cores | 64+ GB | Recommended | 100+ GB |

### Configuration Checklist

- [ ] Appropriate shot count for use case
- [ ] Optimal optimizer selected
- [ ] GPU acceleration enabled (if available)
- [ ] Memory limits configured
- [ ] Caching enabled
- [ ] Monitoring configured
