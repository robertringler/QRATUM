#!/usr/bin/env python3
"""Example: QAOA for MaxCut problem.

This script demonstrates QAOA (Quantum Approximate Optimization Algorithm)
for solving the MaxCut graph partitioning problem.

MaxCut is NP-hard, but small instances can be solved exactly for comparison.
QAOA provides approximate solutions using quantum circuits.

Requirements:
- pip install qiskit qiskit-aer

Expected behavior:
- For small graphs (4-8 nodes), QAOA should find good approximate solutions
- Approximation ratio typically 0.7-0.95 depending on p-layers and graph structure
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quasim.quantum.core import QuantumConfig
from quasim.quantum.qaoa_optimization import QAOA


def main():
    """Run QAOA for MaxCut."""

    print("=" * 60)
    print("QRATUM - QAOA for MaxCut")
    print("=" * 60)
    print()

    # Check if quantum libraries available
    try:
        import qiskit

        print(f"✓ Qiskit version: {qiskit.__version__}")
    except ImportError:
        print("✗ Qiskit not installed. Install with:")
        print("  pip install qiskit qiskit-aer")
        return 1

    print()

    # Configure quantum backend
    config = QuantumConfig(backend_type="simulator", shots=1024, seed=42)

    print("Configuration:")
    print(f"  Backend: {config.backend_type}")
    print(f"  Shots: {config.shots}")
    print("  QAOA layers (p): 3")
    print()

    # Create QAOA instance with p=3 layers
    qaoa = QAOA(config, p_layers=3)

    # Define a small graph for MaxCut
    # Example: 4-node cycle graph
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]  # Add diagonal for more interesting structure

    print("-" * 60)
    print("Problem: MaxCut on 4-node graph")
    print(f"Edges: {edges}")
    print("-" * 60)
    print()

    try:
        result = qaoa.solve_maxcut(
            edges=edges,
            optimizer="COBYLA",
            max_iterations=100,
            classical_reference=True,  # Compare to exact solution
        )

        print()
        print("-" * 60)
        print("Results:")
        print("-" * 60)
        print(result)
        print()

        # Interpret result
        print("Solution interpretation:")
        solution = result.solution
        set_a = [i for i, bit in enumerate(solution) if bit == "0"]
        set_b = [i for i, bit in enumerate(solution) if bit == "1"]
        print(f"  Set A: {set_a}")
        print(f"  Set B: {set_b}")
        print(f"  Cut value: {abs(result.energy):.0f} edges")

        if result.approximation_ratio is not None:
            ratio_pct = result.approximation_ratio * 100
            print(f"  Approximation ratio: {ratio_pct:.1f}%")

            if result.approximation_ratio >= 0.9:
                print("  ✓ EXCELLENT - Found near-optimal solution")
            elif result.approximation_ratio >= 0.7:
                print("  ✓ GOOD - Found decent approximation")
            else:
                print("  ⚠ FAIR - Solution quality could be improved with more p-layers")

        print()
        print("=" * 60)
        print("QAOA demonstration complete!")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"Error during QAOA computation: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
