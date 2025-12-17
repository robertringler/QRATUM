#!/usr/bin/env python3
"""Example: VQE for H2 molecule ground state energy.

This script demonstrates a genuine quantum algorithm (VQE) for computing
the ground state energy of a hydrogen molecule (H2).

Requirements:
- pip install qiskit qiskit-aer qiskit-nature pyscf

Expected result:
- H2 ground state energy ~ -1.137 Hartree (STO-3G basis, bond=0.735 Å)
- Should match classical calculation within ~0.01 Hartree on simulator

Note: This runs on a classical simulator. To run on real quantum hardware,
provide IBM Quantum API token.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quasim.quantum.core import QuantumConfig
from quasim.quantum.vqe_molecule import MolecularVQE


def main():
    """Run VQE for H2 molecule."""
    print("=" * 60)
    print("QRATUM - VQE for H2 Molecule")
    print("=" * 60)
    print()

    # Check if quantum libraries are available
    try:
        import qiskit

        print(f"✓ Qiskit version: {qiskit.__version__}")
    except ImportError:
        print("✗ Qiskit not installed. Install with:")
        print("  pip install qiskit qiskit-aer qiskit-nature pyscf")
        return 1

    print()

    # Configure quantum backend
    config = QuantumConfig(
        backend_type="simulator",
        shots=1024,  # Number of measurements
        seed=42,  # For reproducibility
    )

    print("Configuration:")
    print(f"  Backend: {config.backend_type}")
    print(f"  Shots: {config.shots}")
    print(f"  Seed: {config.seed}")
    print()

    # Create VQE instance
    vqe = MolecularVQE(config)

    # Compute H2 ground state energy
    print("-" * 60)
    print("Computing H2 ground state energy...")
    print("-" * 60)

    try:
        result = vqe.compute_h2_energy(
            bond_length=0.735,  # Equilibrium bond length in Angstroms
            basis="sto3g",  # Minimal basis set
            use_classical_reference=True,  # Compare to classical calculation
            optimizer="COBYLA",  # Classical optimizer
            max_iterations=100,  # Max optimization steps
        )

        print()
        print("-" * 60)
        print("Results:")
        print("-" * 60)
        print(result)
        print()

        # Validate against expected value
        if result.classical_energy is not None and abs(result.classical_energy) > 1e-10:
            error_pct = abs(result.error_vs_classical / result.classical_energy) * 100
            print(f"Error vs. classical: {error_pct:.2f}%")

            if error_pct < 1.0:
                print("✓ Result within 1% of classical - EXCELLENT")
            elif error_pct < 5.0:
                print("✓ Result within 5% of classical - GOOD")
            else:
                print("⚠ Result differs from classical by >5% - may need more shots/iterations")

        print()
        print("=" * 60)
        print("VQE demonstration complete!")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"Error during VQE computation: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
