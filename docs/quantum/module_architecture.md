# Quantum Module Architecture

## Overview

This document describes the architecture of QRATUM's quantum computing modules, including backend abstraction, VQE/QAOA implementations, and platform integration.

See full documentation: [QRATUM GitHub](https://github.com/robertringler/QRATUM)

## Quick Start

```python
from qratum import create_platform

# Create platform with simulator backend
platform = create_platform(quantum_backend="simulator", seed=42)

# Execute VQE
vqe_result = platform.execute_vqe("H2", 0.735)
print(f"H2 energy: {vqe_result['energy']:.6f} Hartree")

# Execute QAOA
edges = [(0,1), (1,2), (2,3), (3,0)]
qaoa_result = platform.execute_qaoa("maxcut", {"edges": edges})
print(f"MaxCut solution: {qaoa_result['solution']}")
```

## Architecture Components

### 1. Backend Abstraction
- **AbstractQuantumBackend**: Base class for all backends
- **QiskitAerBackend**: Local simulator (deterministic)
- **IBMQBackend**: IBM Quantum hardware
- **cuQuantumBackend**: NVIDIA GPU (Phase 2)

### 2. Algorithms
- **VQE**: Molecular ground state energies (H2, LiH, BeH2)
- **QAOA**: Combinatorial optimization (MaxCut, Ising, TSP)

### 3. Classical Fallback
- Exact solvers for small problems
- Heuristics for large problems
- PySCF for quantum chemistry

### 4. Platform Integration
- Unified API via `create_platform()`
- Configuration management
- Result formatting

## Performance

| Algorithm | Problem Size | Runtime (Sim) | Accuracy |
|-----------|-------------|---------------|----------|
| VQE (H2)  | 2 qubits    | 30-60s        | <5% error|
| QAOA      | 4-12 nodes  | 20-120s       | >75% ratio|

## References

- Examples: `/examples/quantum_*.py`
- Tests: `/tests/quantum/test_*.py`
- API Reference: See module docstrings
