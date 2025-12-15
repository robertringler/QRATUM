#!/usr/bin/env python3
"""Example: GHZ (Greenberger-Horne-Zeilinger) state preparation.

GHZ states are maximally entangled multi-qubit states.
For 3 qubits: |GHZ⟩ = (|000⟩ + |111⟩)/√2
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import qratum

def create_ghz_state(num_qubits: int) -> qratum.Circuit:
    """Create GHZ state circuit.
    
    Args:
        num_qubits: Number of qubits
        
    Returns:
        Circuit that prepares GHZ state
    """
    circuit = qratum.Circuit(num_qubits)
    
    # Apply Hadamard to first qubit
    circuit.h(0)
    
    # Apply CNOT chain
    for i in range(num_qubits - 1):
        circuit.cnot(i, i + 1)
    
    return circuit

# Create 3-qubit GHZ state
print("Creating 3-qubit GHZ state")
print("=" * 60)

circuit = create_ghz_state(3)
print("\nGHZ Circuit:")
print(circuit)

# Simulate
simulator = qratum.Simulator(backend="cpu", seed=42)
result = simulator.run(circuit, shots=1000)

print("\n" + "=" * 60)
print("Measurement Results:")
print(result)

# Get state vector
state = simulator.run_statevector(circuit)
print(f"\n{state}")

# Analysis
print("\n" + "=" * 60)
print("GHZ State Properties:")
print(f"  Number of qubits: {circuit.num_qubits}")
print(f"  Expected outcomes: 2 (|000⟩ and |111⟩)")
print(f"  Measured outcomes: {len(result.get_counts())}")
print(f"  Circuit depth: {circuit.depth()}")
print(f"  Gate count: {circuit.gate_count()}")

# Verify perfect correlation
counts = result.get_counts()
all_same = sum(count for state, count in counts.items() if state in ['000', '111'])
print(f"  Perfect correlation: {all_same / result.shots:.2%}")
