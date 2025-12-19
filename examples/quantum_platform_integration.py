#!/usr/bin/env python3
"""Example: QRATUM Platform Integration with Quantum Modules.

This example demonstrates the unified platform API for executing quantum
algorithms (VQE, QAOA) with backend abstraction and classical fallback.

Requirements:
- pip install qiskit qiskit-aer qiskit-nature pyscf

Expected behavior:
- Create platform with quantum backend
- Execute VQE for H2 molecule
- Execute QAOA for MaxCut problem
- Demonstrate classical fallback
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Run platform integration example."""

    print("=" * 70)
    print("QRATUM Platform - Quantum Module Integration Example")
    print("=" * 70)
    print()

    # Check if quantum modules available
    try:
        from qratum import create_platform
    except ImportError as e:
        print(f"✗ Platform integration not available: {e}")
        print("Install quantum dependencies:")
        print("  pip install qiskit qiskit-aer qiskit-nature pyscf")
        return 1

    print("✓ Platform integration available")
    print()

    # Create platform instance
    print("-" * 70)
    print("Creating platform with quantum backend...")
    print("-" * 70)

    try:
        platform = create_platform(
            quantum_backend="simulator",  # Use Qiskit Aer simulator
            seed=42,  # Deterministic execution
            shots=1024,  # Measurement shots
        )
    except ImportError as e:
        print(f"✗ Quantum backend not available: {e}")
        print("This example requires Qiskit. Install with:")
        print("  pip install qiskit qiskit-aer qiskit-nature pyscf")
        return 1

    # Get backend info
    backend_info = platform.get_backend_info()
    print(f"Backend: {backend_info['backend_name']}")
    print(f"Type: {backend_info['backend_type']}")
    print(f"Shots: {backend_info['shots']}")
    print(f"Seed: {backend_info['seed']}")
    print()

    # Example 1: VQE for H2 molecule
    print("-" * 70)
    print("Example 1: VQE for H2 Molecule Ground State Energy")
    print("-" * 70)

    try:
        vqe_result = platform.execute_vqe(
            molecule="H2",
            bond_length=0.735,  # Equilibrium bond length (Angstroms)
            basis="sto3g",
            use_classical_reference=True,
            max_iterations=10,  # Reduced for demo
        )

        print("✓ VQE completed successfully")
        print(f"  Energy: {vqe_result['energy']:.6f} Hartree")
        print(f"  Iterations: {vqe_result['n_iterations']}")
        print(f"  Convergence: {vqe_result['convergence']}")

        if vqe_result.get("classical_energy"):
            print(f"  Classical reference: {vqe_result['classical_energy']:.6f} Ha")
            print(f"  Error: {vqe_result['error_vs_classical']:.6f} Ha")

    except Exception as e:
        print(f"✗ VQE failed: {e}")

    print()

    # Example 2: QAOA for MaxCut
    print("-" * 70)
    print("Example 2: QAOA for MaxCut Optimization")
    print("-" * 70)

    try:
        # Define a small graph
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]

        qaoa_result = platform.execute_qaoa(
            problem_type="maxcut",
            problem_data={"edges": edges},
            p_layers=2,  # QAOA depth
            max_iterations=10,  # Reduced for demo
            classical_reference=True,
        )

        print("✓ QAOA completed successfully")
        print(f"  Solution: {qaoa_result['solution']}")
        print(f"  Cut value: {abs(qaoa_result['energy']):.0f} edges")
        print(f"  Iterations: {qaoa_result['n_iterations']}")

        if qaoa_result.get("approximation_ratio"):
            ratio = qaoa_result["approximation_ratio"]
            print(f"  Approximation ratio: {ratio:.2%}")

    except Exception as e:
        print(f"✗ QAOA failed: {e}")

    print()

    # Example 3: Classical fallback
    print("-" * 70)
    print("Example 3: Classical Fallback for Large Problems")
    print("-" * 70)

    try:
        from quasim.opt.classical_fallback import ClassicalFallback

        fallback = ClassicalFallback()

        # MaxCut classical solution
        classical_result = fallback.solve_maxcut(edges, method="exact")

        print("✓ Classical MaxCut solution")
        print(f"  Solution: {classical_result['solution']}")
        print(f"  Cut value: {classical_result['cut_value']} edges")
        print(f"  Method: {classical_result['method']}")
        print(f"  Optimal: {classical_result['optimal']}")

    except Exception as e:
        print(f"✗ Classical fallback failed: {e}")

    print()
    print("=" * 70)
    print("Platform integration example complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
