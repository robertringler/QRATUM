# First Simulation

Run your first quantum simulation with QRATUM - a VQE calculation for the H₂ molecule.

## Prerequisites

- QRATUM installed (see [Installation](installation.md))
- Python 3.10+

## Understanding VQE

**Variational Quantum Eigensolver (VQE)** is a hybrid quantum-classical algorithm that finds the ground state energy of molecules:

1. Prepare a parameterized quantum circuit (ansatz)
2. Measure the expectation value of the Hamiltonian
3. Use classical optimization to find optimal parameters
4. Iterate until convergence

## Running the H₂ Simulation

### Option 1: Using the CLI

```bash
python examples/quantum_h2_vqe.py
```

### Option 2: Python Script

Create a file `my_first_simulation.py`:

```python
from quasim.quantum.core import QuantumConfig
from quasim.quantum.vqe_molecule import MolecularVQE

# Configure quantum backend
config = QuantumConfig(
    backend_type="simulator",  # Use "ibmq" for real hardware
    shots=1024,
    seed=42  # For reproducibility
)

# Create VQE instance
vqe = MolecularVQE(config)

# Compute H₂ ground state energy
result = vqe.compute_h2_energy(
    bond_length=0.735,  # Angstroms (equilibrium distance)
    basis="sto3g",
    use_classical_reference=True,
    max_iterations=100
)

# Display results
print(f"Ground state energy: {result.energy:.6f} Hartree")
print(f"Classical reference: {result.classical_energy:.6f} Hartree")
print(f"Error: {result.error_vs_classical:.6f} Hartree")
print(f"Iterations: {result.iterations}")
print(f"Converged: {result.converged}")
```

Run it:

```bash
python my_first_simulation.py
```

### Expected Output

```
Ground state energy: -1.136189 Hartree
Classical reference: -1.137270 Hartree
Error: 0.001081 Hartree
Iterations: 47
Converged: True
```

## Understanding the Results

| Metric | Value | Meaning |
|--------|-------|---------|
| Ground state energy | -1.136 Ha | VQE result |
| Classical reference | -1.137 Ha | Exact Hartree-Fock value |
| Error | ~0.001 Ha | VQE accuracy (~1%) |
| Iterations | 47 | Optimization steps |

!!! info "Why the Error?"

    The VQE result differs from classical reference due to:
    
    - **Shot noise**: Statistical uncertainty from finite measurements
    - **Ansatz limitations**: Parameterized circuit may not reach exact ground state
    - **Optimization**: Classical optimizer may find local minima

## Exploring Bond Length Dependence

Compute the potential energy surface:

```python
import numpy as np
import matplotlib.pyplot as plt

from quasim.quantum.core import QuantumConfig
from quasim.quantum.vqe_molecule import MolecularVQE

config = QuantumConfig(backend_type="simulator", shots=1024, seed=42)
vqe = MolecularVQE(config)

# Scan bond lengths
bond_lengths = np.linspace(0.3, 2.5, 20)
energies = []

for r in bond_lengths:
    result = vqe.compute_h2_energy(
        bond_length=r,
        basis="sto3g",
        max_iterations=50
    )
    energies.append(result.energy)
    print(f"R = {r:.2f} Å, E = {result.energy:.4f} Ha")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(bond_lengths, energies, 'b-o', label='VQE')
plt.xlabel('Bond Length (Å)')
plt.ylabel('Energy (Hartree)')
plt.title('H₂ Potential Energy Surface')
plt.legend()
plt.grid(True)
plt.savefig('h2_pes.png')
plt.show()
```

## What's Next?

<div class="grid cards" markdown>

-   :material-molecule:{ .lg .middle } __QAOA Tutorial__

    ---

    Learn QAOA for combinatorial optimization
    
    [:octicons-arrow-right-24: QAOA MaxCut](../tutorials/qaoa-maxcut.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Explore the full API
    
    [:octicons-arrow-right-24: API Reference](../reference/api-reference.md)

-   :material-book-multiple:{ .lg .middle } __More Tutorials__

    ---

    Advanced quantum simulations
    
    [:octicons-arrow-right-24: All Tutorials](../tutorials/index.md)

</div>

## Benchmarks

| Method | Energy (Hartree) | Error vs. Exact | Runtime |
|--------|------------------|-----------------|---------|
| Classical HF | -1.137 | 0% (reference) | <1s |
| QRATUM VQE (simulator) | -1.12 to -1.14 | 1-5% | 30-60s |
| Real IBM Quantum | -1.0 to -1.2 | 5-15% | 5-10min (queue) |

!!! warning "Simulation Limits"

    Current VQE implementation works for:
    
    - Small molecules (H₂, HeH+)
    - 2-4 qubits
    - Simulator backend only
    
    Larger molecules require more qubits and are limited by NISQ noise.
