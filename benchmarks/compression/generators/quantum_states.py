"""Quantum state generators for compression benchmarking.

This module provides utilities to generate various types of quantum states
for testing MERA compression across different entanglement structures.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from numpy.typing import NDArray

# Optional quantum dependencies
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumCircuit = None

Array = NDArray[np.complex128]


def generate_random_state(n_qubits: int, seed: Optional[int] = None) -> Array:
    """Generate a normalized random quantum state.

    Creates a random complex vector in the Hilbert space and normalizes it
    to represent a valid quantum state.

    Args:
        n_qubits: Number of qubits (state dimension will be 2^n_qubits)
        seed: Random seed for reproducibility

    Returns:
        Normalized complex state vector of shape (2^n_qubits,)

    Example:
        >>> state = generate_random_state(3, seed=42)
        >>> assert state.shape == (8,)
        >>> assert abs(np.linalg.norm(state) - 1.0) < 1e-10
    """
    if seed is not None:
        np.random.seed(seed)

    dim = 2**n_qubits

    # Generate random complex amplitudes
    real_part = np.random.randn(dim)
    imag_part = np.random.randn(dim)
    state = real_part + 1j * imag_part

    # Normalize to unit norm
    state = state / np.linalg.norm(state)

    return state


def generate_product_state(n_qubits: int, state_type: str = "zero") -> Array:
    """Generate a product (separable) quantum state.

    Product states have no entanglement and should compress extremely well.

    Args:
        n_qubits: Number of qubits
        state_type: Type of product state:
            - "zero": |0⟩^⊗n (all qubits in |0⟩)
            - "one": |1⟩^⊗n (all qubits in |1⟩)
            - "plus": |+⟩^⊗n (all qubits in (|0⟩+|1⟩)/√2)
            - "minus": |-⟩^⊗n (all qubits in (|0⟩-|1⟩)/√2)

    Returns:
        Product state vector of shape (2^n_qubits,)

    Example:
        >>> state = generate_product_state(3, "zero")
        >>> assert state[0] == 1.0
        >>> assert np.sum(np.abs(state[1:])) < 1e-10
    """
    dim = 2**n_qubits

    if state_type == "zero":
        # |000...0⟩
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

    elif state_type == "one":
        # |111...1⟩
        state = np.zeros(dim, dtype=complex)
        state[-1] = 1.0

    elif state_type == "plus":
        # |+⟩^⊗n = (1/√2^n) Σ|x⟩
        state = np.ones(dim, dtype=complex) / np.sqrt(dim)

    elif state_type == "minus":
        # |-⟩^⊗n with alternating signs
        state = np.ones(dim, dtype=complex)
        for i in range(dim):
            # Count number of 1s in binary representation
            if bin(i).count("1") % 2 == 1:
                state[i] = -1.0
        state = state / np.sqrt(dim)

    else:
        raise ValueError(f"Unknown state_type: {state_type}")

    return state


def generate_ghz_state(n_qubits: int) -> Array:
    """Generate a Greenberger-Horne-Zeilinger (GHZ) state.

    GHZ states are maximally entangled: |GHZ⟩ = (|000...0⟩ + |111...1⟩) / √2

    These states have high entanglement but structured correlation, making them
    interesting test cases for compression algorithms.

    Args:
        n_qubits: Number of qubits

    Returns:
        GHZ state vector of shape (2^n_qubits,)

    Example:
        >>> state = generate_ghz_state(3)
        >>> assert abs(state[0] - 1/np.sqrt(2)) < 1e-10
        >>> assert abs(state[7] - 1/np.sqrt(2)) < 1e-10
    """
    dim = 2**n_qubits
    state = np.zeros(dim, dtype=complex)

    # Superposition of |000...0⟩ and |111...1⟩
    state[0] = 1.0 / np.sqrt(2)
    state[-1] = 1.0 / np.sqrt(2)

    return state


def generate_w_state(n_qubits: int) -> Array:
    """Generate a W state.

    W states are symmetric entangled states with equal superposition of all
    single-excitation basis states: |W⟩ = (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √n

    These have different entanglement structure than GHZ states.

    Args:
        n_qubits: Number of qubits

    Returns:
        W state vector of shape (2^n_qubits,)

    Example:
        >>> state = generate_w_state(3)
        >>> # Should have amplitude 1/√3 at positions 1, 2, 4
        >>> assert abs(state[1] - 1/np.sqrt(3)) < 1e-10
    """
    dim = 2**n_qubits
    state = np.zeros(dim, dtype=complex)

    # Add amplitude at each single-qubit excitation position
    for i in range(n_qubits):
        # Position where only qubit i is |1⟩
        position = 2**i
        state[position] = 1.0 / np.sqrt(n_qubits)

    return state


def generate_random_circuit_state(
    n_qubits: int, depth: int = 10, seed: Optional[int] = None
) -> Array:
    """Generate a quantum state from a random quantum circuit.

    Creates a random circuit with Hadamard, CNOT, and rotation gates,
    then simulates it to get the final state vector.

    Args:
        n_qubits: Number of qubits
        depth: Circuit depth (number of gate layers)
        seed: Random seed for reproducibility

    Returns:
        State vector from circuit simulation, shape (2^n_qubits,)

    Example:
        >>> state = generate_random_circuit_state(4, depth=20, seed=42)
        >>> assert state.shape == (16,)
        >>> assert abs(np.linalg.norm(state) - 1.0) < 1e-10
    """
    if not QISKIT_AVAILABLE:
        warnings.warn(
            "Qiskit not available. Using random state instead of circuit state.",
            UserWarning,
        )
        return generate_random_state(n_qubits, seed=seed)

    if seed is not None:
        np.random.seed(seed)

    # Create quantum circuit
    circuit = QuantumCircuit(n_qubits)

    # Add random gate layers
    for _ in range(depth):
        # Random single-qubit gates
        for qubit in range(n_qubits):
            gate_choice = np.random.randint(0, 4)
            if gate_choice == 0:
                circuit.h(qubit)
            elif gate_choice == 1:
                circuit.x(qubit)
            elif gate_choice == 2:
                angle = np.random.uniform(0, 2 * np.pi)
                circuit.rz(angle, qubit)
            else:
                angle = np.random.uniform(0, 2 * np.pi)
                circuit.ry(angle, qubit)

        # Random two-qubit gates (if enough qubits)
        if n_qubits >= 2:
            for _ in range(n_qubits // 2):
                qubit1 = np.random.randint(0, n_qubits)
                qubit2 = np.random.randint(0, n_qubits)
                if qubit1 != qubit2:
                    circuit.cx(qubit1, qubit2)

    # Simulate to get state vector
    simulator = AerSimulator(method="statevector")
    circuit.save_statevector()

    result = simulator.run(circuit, shots=1).result()
    statevector = result.get_statevector(circuit)

    # Convert to numpy array
    state = np.array(statevector.data, dtype=complex)

    return state


def generate_vqe_h2_state(bond_length: float = 0.735) -> Optional[Array]:
    """Generate H2 VQE ground state if quantum modules available.

    This is a helper function to generate realistic VQE states for benchmarking.
    Requires qiskit-nature and pyscf to be installed.

    Args:
        bond_length: H-H bond length in Angstroms

    Returns:
        H2 ground state vector (4-dimensional for 2 qubits), or None if unavailable

    Note:
        This function may take several seconds to compute.
    """
    try:
        from quasim.quantum.core import QuantumConfig
        from quasim.quantum.vqe_molecule import MolecularVQE

        # Run VQE to get H2 state
        config = QuantumConfig(shots=1024, seed=42)
        vqe = MolecularVQE(config)
        _result = vqe.compute_h2_energy(
            bond_length=bond_length, max_iterations=50, optimizer="COBYLA"
        )

        # Extract state vector from optimal parameters
        # This requires re-running the circuit with optimal parameters
        # For now, return None as full implementation would be complex
        warnings.warn(
            "VQE state extraction not fully implemented. Use random circuit instead.",
            UserWarning,
        )
        return None

    except ImportError:
        warnings.warn("Quantum modules not available for VQE state generation.", UserWarning)
        return None


def generate_qaoa_maxcut_state(n_nodes: int, edges: list[tuple[int, int]]) -> Optional[Array]:
    """Generate QAOA MaxCut state if quantum modules available.

    Args:
        n_nodes: Number of graph nodes (number of qubits)
        edges: List of edges as (node_i, node_j) tuples

    Returns:
        QAOA state vector (2^n_nodes dimensional), or None if unavailable
    """
    try:
        from quasim.quantum.core import QuantumConfig
        from quasim.quantum.qaoa_optimization import QAOA

        # Run QAOA to get state
        config = QuantumConfig(shots=1024, seed=42)
        _qaoa = QAOA(config, p_layers=3)

        # For state vector, we'd need to extract from optimal parameters
        # This is complex, so we'll return None for now
        warnings.warn(
            "QAOA state extraction not fully implemented. Use random circuit instead.",
            UserWarning,
        )
        return None

    except ImportError:
        warnings.warn("Quantum modules not available for QAOA state generation.", UserWarning)
        return None
