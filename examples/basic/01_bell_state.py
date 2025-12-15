#!/usr/bin/env python3
"""Example: Creating and measuring a Bell state.

Bell states are maximally entangled two-qubit states.
This example creates the |Φ+⟩ = (|00⟩ + |11⟩)/√2 state.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import qratum

# Create a 2-qubit circuit
circuit = qratum.Circuit(2)

# Apply Hadamard to first qubit
circuit.h(0)

# Apply CNOT with control=0, target=1
circuit.cnot(0, 1)

print("Bell State Circuit:")
print(circuit)
print("\n" + "=" * 60 + "\n")

# Create simulator
simulator = qratum.Simulator(backend="cpu", seed=42)

# Run simulation and measure
result = simulator.run(circuit, shots=1000)

print("Measurement Results:")
print(result)
print("\n" + "=" * 60 + "\n")

# Get probabilities
probs = result.get_probabilities()
print("Probabilities:")
for state, prob in sorted(probs.items()):
    print(f"  |{state}⟩: {prob:.4f}")

# Get state vector (without measurement)
state = simulator.run_statevector(circuit)
print(f"\n{state}")

# Verify entanglement: both qubits should have equal superposition
print("\n" + "=" * 60)
print("Analysis:")
print(f"  Measured states: {len(result.get_counts())}")
print(f"  Expected: 2 states (|00⟩ and |11⟩)")
print(f"  Circuit depth: {circuit.depth()}")
print(f"  Gate count: {circuit.gate_count()}")
