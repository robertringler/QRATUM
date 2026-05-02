# VQE for H₂ Molecule

This tutorial covers the Variational Quantum Eigensolver (VQE) algorithm for computing molecular ground state energies.

## Overview

VQE is a hybrid quantum-classical algorithm that:

1. Prepares a parameterized quantum state (ansatz)
2. Measures the expectation value of the molecular Hamiltonian
3. Uses classical optimization to minimize the energy
4. Finds the ground state energy of the molecule

## Theory

The molecular Hamiltonian for H₂ in second quantization:

$$
H = \sum_{pq} h_{pq} a_p^\dagger a_q + \frac{1}{2} \sum_{pqrs} h_{pqrs} a_p^\dagger a_q^\dagger a_r a_s
$$

VQE minimizes:

$$
E(\theta) = \langle \psi(\theta) | H | \psi(\theta) \rangle
$$

where $|\psi(\theta)\rangle$ is the parameterized ansatz.

## Basic Example

```python
from quasim.quantum.core import QuantumConfig
from quasim.quantum.vqe_molecule import MolecularVQE

# Configure quantum backend
config = QuantumConfig(
    backend_type="simulator",
    shots=1024,
    seed=42
)

# Create VQE instance
vqe = MolecularVQE(config)

# Compute H₂ ground state at equilibrium
result = vqe.compute_h2_energy(
    bond_length=0.735,  # Equilibrium distance in Angstroms
    basis="sto3g",
    use_classical_reference=True,
    max_iterations=100
)

print(f"VQE Energy: {result.energy:.6f} Hartree")
print(f"Classical: {result.classical_energy:.6f} Hartree")
print(f"Error: {abs(result.energy - result.classical_energy):.6f} Hartree")
```

## Potential Energy Surface

Compute energy vs. bond length:

```python
import numpy as np
import matplotlib.pyplot as plt

# Bond lengths to scan
bond_lengths = np.linspace(0.4, 2.5, 15)
vqe_energies = []
classical_energies = []

for r in bond_lengths:
    result = vqe.compute_h2_energy(
        bond_length=r,
        basis="sto3g",
        use_classical_reference=True,
        max_iterations=50
    )
    vqe_energies.append(result.energy)
    classical_energies.append(result.classical_energy)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(bond_lengths, classical_energies, 'k-', label='Classical (exact)', linewidth=2)
plt.plot(bond_lengths, vqe_energies, 'bo-', label='VQE', markersize=6)
plt.xlabel('Bond Length (Å)', fontsize=12)
plt.ylabel('Energy (Hartree)', fontsize=12)
plt.title('H₂ Potential Energy Surface', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('h2_pes.png', dpi=150)
plt.show()
```

## Advanced Configuration

### Custom Ansatz Depth

```python
result = vqe.compute_h2_energy(
    bond_length=0.735,
    basis="sto3g",
    ansatz_layers=3,  # More layers = more expressiveness
    max_iterations=200
)
```

### Different Optimizers

```python
result = vqe.compute_h2_energy(
    bond_length=0.735,
    basis="sto3g",
    optimizer="COBYLA",  # Options: COBYLA, L-BFGS-B, SLSQP
    max_iterations=100
)
```

### Increased Shot Count

```python
config = QuantumConfig(
    backend_type="simulator",
    shots=4096,  # More shots = less statistical noise
    seed=42
)
```

## Running on Real Hardware

!!! warning "IBM Quantum Account Required"

    You need an IBM Quantum account and API token.

```python
config = QuantumConfig(
    backend_type="ibmq",
    ibmq_token="YOUR_API_TOKEN",
    shots=1024
)

vqe = MolecularVQE(config)
result = vqe.compute_h2_energy(bond_length=0.735)

# Note: Queue times can be 5-60 minutes
```

## Benchmarks

| Bond Length | Classical (Ha) | VQE (Ha) | Error | Iterations |
|-------------|----------------|----------|-------|------------|
| 0.5 Å | -0.8451 | -0.8440 | 0.13% | 42 |
| 0.735 Å | -1.1373 | -1.1362 | 0.10% | 47 |
| 1.0 Å | -1.1016 | -1.0998 | 0.16% | 51 |
| 1.5 Å | -0.9782 | -0.9761 | 0.21% | 58 |
| 2.0 Å | -0.9046 | -0.9012 | 0.38% | 63 |

## Common Issues

### Poor Convergence

- Increase `max_iterations`
- Try different optimizers
- Increase `ansatz_layers`

### High Variance

- Increase `shots` count
- Use error mitigation (when available)

### Slow Execution

- Reduce `shots` for development
- Use local simulator before hardware

## Next Steps

- [QAOA for MaxCut](qaoa-maxcut.md) - Combinatorial optimization
- [API Reference](../reference/api-reference.md) - Full API documentation
- [Performance Tuning](../advanced/performance-tuning.md) - Optimization tips
