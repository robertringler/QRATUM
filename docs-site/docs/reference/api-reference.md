# API Reference

Complete API documentation for QRATUM quantum and simulation modules.

## Core Module

### QuantumConfig

Configuration for quantum backends.

```python
from quasim.quantum.core import QuantumConfig

config = QuantumConfig(
    backend_type: str = "simulator",  # "simulator", "ibmq"
    shots: int = 1024,                 # Measurement shots
    seed: int | None = None,           # Random seed
    ibmq_token: str | None = None,     # IBM Quantum token
    max_qubits: int = 20               # Maximum qubit count
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend_type` | `str` | `"simulator"` | Backend: `"simulator"` or `"ibmq"` |
| `shots` | `int` | `1024` | Number of measurement shots |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `ibmq_token` | `str \| None` | `None` | IBM Quantum API token |
| `max_qubits` | `int` | `20` | Maximum qubits for simulation |

#### Example

```python
# Simulator backend
config = QuantumConfig(
    backend_type="simulator",
    shots=2048,
    seed=42
)

# IBM Quantum backend
config = QuantumConfig(
    backend_type="ibmq",
    ibmq_token="your_api_token",
    shots=1024
)
```

---

## VQE Module

### MolecularVQE

Variational Quantum Eigensolver for molecular ground states.

```python
from quasim.quantum.vqe_molecule import MolecularVQE

vqe = MolecularVQE(config: QuantumConfig)
```

#### Methods

##### compute_h2_energy

Compute H₂ molecule ground state energy.

```python
result = vqe.compute_h2_energy(
    bond_length: float,              # Bond length in Angstroms
    basis: str = "sto3g",            # Basis set
    use_classical_reference: bool = True,
    max_iterations: int = 100,
    optimizer: str = "COBYLA",
    ansatz_layers: int = 2
)
```

**Returns:** `VQEResult`

```python
@dataclass
class VQEResult:
    energy: float              # VQE computed energy (Hartree)
    classical_energy: float    # Classical reference energy
    error_vs_classical: float  # Absolute error
    iterations: int            # Optimization iterations
    converged: bool            # Convergence status
    parameters: np.ndarray     # Optimal variational parameters
```

#### Full Example

```python
from quasim.quantum.core import QuantumConfig
from quasim.quantum.vqe_molecule import MolecularVQE

config = QuantumConfig(backend_type="simulator", shots=1024, seed=42)
vqe = MolecularVQE(config)

result = vqe.compute_h2_energy(
    bond_length=0.735,
    basis="sto3g",
    use_classical_reference=True,
    max_iterations=100
)

print(f"Energy: {result.energy:.6f} Ha")
print(f"Error: {result.error_vs_classical:.6f} Ha")
```

---

## QAOA Module

### QAOA

Quantum Approximate Optimization Algorithm.

```python
from quasim.quantum.qaoa_optimization import QAOA

qaoa = QAOA(
    config: QuantumConfig,
    p_layers: int = 3     # Number of QAOA layers
)
```

#### Methods

##### solve_maxcut

Solve MaxCut graph partitioning problem.

```python
result = qaoa.solve_maxcut(
    edges: list[tuple[int, int]],   # Graph edges
    max_iterations: int = 100,
    classical_reference: bool = True,
    optimizer: str = "COBYLA",
    initial_params: np.ndarray | None = None
)
```

**Returns:** `QAOAResult`

```python
@dataclass
class QAOAResult:
    solution: str              # Binary partition string
    energy: float              # QAOA objective value
    classical_optimal: float   # Classical optimal value
    approximation_ratio: float # QAOA / classical ratio
    iterations: int            # Optimization iterations
    parameters: np.ndarray     # Optimal (gamma, beta) parameters
```

##### solve_ising

Solve Ising model ground state.

```python
result = qaoa.solve_ising(
    coupling_matrix: np.ndarray,    # J_ij coupling matrix
    max_iterations: int = 100
)
```

**Returns:** `QAOAResult`

#### Example

```python
from quasim.quantum.core import QuantumConfig
from quasim.quantum.qaoa_optimization import QAOA

config = QuantumConfig(backend_type="simulator", shots=1024)
qaoa = QAOA(config, p_layers=3)

edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]

result = qaoa.solve_maxcut(edges, max_iterations=100, classical_reference=True)

print(f"Partition: {result.solution}")
print(f"Cut value: {abs(result.energy):.0f}")
print(f"Approximation: {result.approximation_ratio:.2%}")
```

---

## REST API

### Endpoints

#### Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "version": "2.0.0"
}
```

#### Run Kernel

```http
POST /kernel
Content-Type: application/json

{
  "seed": 42,
  "scale": 1.0
}
```

**Response:**

```json
{
  "state_vector": [...],
  "energy": -1.137,
  "convergence": true,
  "execution_time_ms": 1234
}
```

#### Metrics

```http
GET /metrics
```

Returns Prometheus-format metrics.

### Python Client

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Run kernel
response = requests.post(
    "http://localhost:8000/kernel",
    json={"seed": 42, "scale": 1.0}
)
result = response.json()
print(f"Energy: {result['energy']}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Run kernel
curl -X POST http://localhost:8000/kernel \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "scale": 1.0}'

# Get metrics
curl http://localhost:8000/metrics
```

---

## Error Handling

### Common Exceptions

```python
from quasim.quantum.exceptions import (
    QuantumBackendError,
    ConfigurationError,
    ConvergenceError
)

try:
    result = vqe.compute_h2_energy(bond_length=0.735)
except QuantumBackendError as e:
    print(f"Backend error: {e}")
except ConvergenceError as e:
    print(f"Did not converge: {e}")
except ConfigurationError as e:
    print(f"Invalid configuration: {e}")
```

### Error Codes

| Error | Code | Description |
|-------|------|-------------|
| `QuantumBackendError` | 1001 | Backend unavailable |
| `ConfigurationError` | 1002 | Invalid configuration |
| `ConvergenceError` | 1003 | Optimization did not converge |
| `QubitLimitError` | 1004 | Exceeded qubit limit |

---

## Type Annotations

All QRATUM modules use Python type hints:

```python
from typing import Optional
import numpy as np
from numpy.typing import NDArray

def compute_energy(
    parameters: NDArray[np.float64],
    shots: int = 1024,
    seed: Optional[int] = None
) -> float:
    ...
```
