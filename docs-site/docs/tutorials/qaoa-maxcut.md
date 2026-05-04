# QAOA for MaxCut

This tutorial covers the Quantum Approximate Optimization Algorithm (QAOA) for solving the MaxCut graph partitioning problem.

## Overview

**MaxCut** partitions graph vertices into two sets to maximize edges between sets. QAOA uses:

1. A problem Hamiltonian encoding the objective
2. A mixing Hamiltonian for exploration
3. Alternating layers of problem and mixing operators
4. Classical optimization of variational parameters

## Theory

For a graph $G = (V, E)$, the MaxCut objective:

$$
C = \sum_{(i,j) \in E} \frac{1 - Z_i Z_j}{2}
$$

QAOA prepares the state:

$$
|\gamma, \beta\rangle = U_B(\beta_p) U_C(\gamma_p) \cdots U_B(\beta_1) U_C(\gamma_1) |+\rangle^n
$$

where $U_C(\gamma) = e^{-i\gamma C}$ and $U_B(\beta) = e^{-i\beta \sum_i X_i}$.

## Basic Example

```python
from quasim.quantum.core import QuantumConfig
from quasim.quantum.qaoa_optimization import QAOA

# Configure quantum backend
config = QuantumConfig(
    backend_type="simulator",
    shots=1024,
    seed=42
)

# Create QAOA solver with 3 layers
qaoa = QAOA(config, p_layers=3)

# Define graph edges (4-node graph)
edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (0, 2)
]

# Solve MaxCut
result = qaoa.solve_maxcut(
    edges=edges,
    max_iterations=100,
    classical_reference=True
)

print(f"Best partition: {result.solution}")
print(f"Cut value: {abs(result.energy):.0f} edges")
print(f"Approximation ratio: {result.approximation_ratio:.2%}")
```

## Understanding Results

```
Best partition: 0101
Cut value: 4 edges
Approximation ratio: 100.00%
```

- **Partition**: `0101` means vertices 0,2 in set A; vertices 1,3 in set B
- **Cut value**: 4 edges cross the partition
- **Approximation ratio**: Compared to optimal (classical) solution

## Larger Graph Example

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create random graph
G = nx.gnm_random_graph(n=8, m=12, seed=42)
edges = list(G.edges())

# Solve with QAOA
result = qaoa.solve_maxcut(
    edges=edges,
    max_iterations=200,
    classical_reference=True
)

# Visualize solution
partition = [int(b) for b in result.solution]
colors = ['lightblue' if p == 0 else 'lightcoral' for p in partition]

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, 
        node_color=colors,
        with_labels=True,
        node_size=700,
        font_size=14,
        font_weight='bold')
plt.title(f'MaxCut Solution: {abs(result.energy):.0f} edges cut\n'
          f'Approx. ratio: {result.approximation_ratio:.2%}')
plt.savefig('maxcut_solution.png', dpi=150)
plt.show()
```

## Ising Model for Materials Science

QAOA can solve Ising models (proxy for lattice defects):

```python
import numpy as np

# 3-spin Ising model coupling matrix
# J_ij > 0: antiferromagnetic
# J_ij < 0: ferromagnetic
coupling_matrix = np.array([
    [0, -1, 0.5],
    [-1, 0, -1],
    [0.5, -1, 0]
])

config = QuantumConfig(backend_type="simulator", shots=1024)
qaoa = QAOA(config, p_layers=3)

result = qaoa.solve_ising(
    coupling_matrix=coupling_matrix,
    max_iterations=100
)

print(f"Ground state: {result.solution}")  # '0'=spin up, '1'=spin down
print(f"Energy: {result.energy:.4f}")
```

## Tuning QAOA Performance

### Number of Layers

More layers = better approximation but longer runtime:

```python
# Few layers (fast, approximate)
qaoa_shallow = QAOA(config, p_layers=1)

# Deep circuit (slower, more accurate)
qaoa_deep = QAOA(config, p_layers=5)
```

### Optimization Strategy

```python
result = qaoa.solve_maxcut(
    edges=edges,
    optimizer="COBYLA",      # Good for noisy landscapes
    max_iterations=200,
    tolerance=1e-6
)
```

### Initial Parameters

```python
import numpy as np

# Custom initial parameters (gamma, beta for each layer)
initial_params = np.array([0.5, 0.3, 0.4, 0.2, 0.3, 0.1])  # p=3 layers

result = qaoa.solve_maxcut(
    edges=edges,
    initial_params=initial_params,
    max_iterations=100
)
```

## Benchmarks

| Graph Size | Optimal Cut | QAOA (p=3) | Approx. Ratio | Runtime |
|------------|-------------|------------|---------------|---------|
| 4 nodes | 4 | 3-4 | 75-100% | ~20s |
| 8 nodes | 8 | 6-8 | 75-100% | ~60s |
| 12 nodes | 12 | 9-11 | 75-92% | ~120s |
| 16 nodes | 16 | 11-14 | 69-88% | ~300s |

!!! note "Scaling Limits"

    Classical simulation scales as $O(2^n)$ for $n$ qubits.
    Practical limit: ~20-25 nodes on classical simulators.

## Common Issues

### Poor Approximation Ratio

- Increase `p_layers`
- Increase `max_iterations`
- Try different initial parameters

### Slow Convergence

- Use `COBYLA` optimizer for noisy landscapes
- Reduce graph size for development

### Memory Errors

- Reduce graph size
- Use GPU backend if available

## Advanced: Custom Problem Hamiltonians

```python
import numpy as np

# Custom Hamiltonian matrix (diagonal)
# Example: Weighted MaxCut
H_custom = np.diag([0, 1, 1, 2, 1, 2, 2, 3])

result = qaoa.solve_custom(
    hamiltonian=H_custom,
    n_qubits=3,
    p_layers=3
)
```

## Next Steps

- [Kubernetes Deployment](kubernetes-deployment.md) - Production deployment
- [CI/CD Integration](cicd-integration.md) - Automated pipelines
- [API Reference](../reference/api-reference.md) - Full documentation
