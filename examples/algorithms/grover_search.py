#!/usr/bin/env python3
"""Example: Grover's search algorithm.

Demonstrates using Grover's algorithm to find marked elements
in an unstructured database with quadratic speedup.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qratum import Simulator
from qratum.algorithms.grover import Grover

print("Grover's Search Algorithm")
print("=" * 60)

# Search in 3-qubit space (8 elements total)
num_qubits = 3
search_space_size = 2**num_qubits

# Mark elements 3 and 5
marked_states = [3, 5]

print(f"\nSearch Parameters:")
print(f"  Search space size: {search_space_size}")
print(f"  Marked elements: {marked_states}")
print(f"  Classical complexity: O({search_space_size})")
print(f"  Quantum complexity: O(√{search_space_size}) ≈ {int(search_space_size**0.5)}")

# Create Grover instance
grover = Grover(num_qubits, marked_states)

print(f"\nGrover Configuration:")
print(f"  {grover}")
print(f"  Optimal iterations: {grover.iterations}")
print(f"  Theoretical success probability: {grover.success_probability():.4f}")

# Build and display circuit
circuit = grover.build_circuit()
print(f"\nCircuit Properties:")
print(f"  Depth: {circuit.depth()}")
print(f"  Gate count: {circuit.gate_count()}")

# Run search
print("\n" + "=" * 60)
print("Running Grover Search...")
simulator = Simulator(backend="cpu", seed=42)
result = grover.run(simulator, shots=1000)

print("\nMeasurement Results:")
print(result)

# Analyze results
print("\n" + "=" * 60)
print("Analysis:")

probs = result.get_probabilities()
print(f"\nProbabilities for all states:")
for i in range(search_space_size):
    state_str = format(i, f"0{num_qubits}b")
    prob = probs.get(state_str, 0.0)
    marker = " ← MARKED" if i in marked_states else ""
    print(f"  |{state_str}⟩ ({i}): {prob:.4f}{marker}")

# Get found states
found = grover.find_marked_states(simulator, shots=1000)
print(f"\nFound states: {found}")
print(f"Expected states: {marked_states}")

success = all(s in marked_states for s in found)
print(f"\nSearch successful: {success}")

# Compare with classical search
print("\n" + "=" * 60)
print("Complexity Comparison:")
print(f"  Classical queries: {search_space_size} (worst case)")
print(f"  Quantum iterations: {grover.iterations}")
print(f"  Speedup factor: {search_space_size / grover.iterations:.2f}x")
