"""QRATUM density matrix representation.

Provides density matrix representation for mixed quantum states
with support for noise and decoherence modeling.
"""

from typing import List, Optional

import numpy as np


class DensityMatrix:
    """Quantum density matrix representation.

    Represents mixed quantum states using density matrix formalism.

    Attributes:
        data: Complex numpy array representing the density matrix
        num_qubits: Number of qubits
    """

    def __init__(self, data: np.ndarray, num_qubits: Optional[int] = None):
        """Initialize density matrix.

        Args:
            data: Complex numpy array (dim x dim)
            num_qubits: Number of qubits (inferred if not provided)

        Raises:
            ValueError: If matrix is not square or dimension not power of 2
        """

        self.data = np.asarray(data, dtype=complex)
        if self.data.shape[0] != self.data.shape[1]:
            raise ValueError("Density matrix must be square")

        dim = self.data.shape[0]
        if num_qubits is None:
            num_qubits = int(np.log2(dim))
        if 2**num_qubits != dim:
            raise ValueError(f"Dimension {dim} is not a power of 2")

        self.num_qubits = num_qubits

    @classmethod
    def from_statevector(cls, state: np.ndarray) -> "DensityMatrix":
        """Create density matrix from pure state.

        Args:
            state: State vector

        Returns:
            DensityMatrix representing |ψ⟩⟨ψ|
        """

        state = np.asarray(state, dtype=complex)
        density = np.outer(state, np.conj(state))
        num_qubits = int(np.log2(len(state)))
        return cls(density, num_qubits)

    @classmethod
    def zero_state(cls, num_qubits: int) -> "DensityMatrix":
        """Create |0...0⟩⟨0...0| density matrix.

        Args:
            num_qubits: Number of qubits

        Returns:
            Density matrix in |0...0⟩ state
        """

        dim = 2**num_qubits
        density = np.zeros((dim, dim), dtype=complex)
        density[0, 0] = 1.0
        return cls(density, num_qubits)

    @classmethod
    def maximally_mixed(cls, num_qubits: int) -> "DensityMatrix":
        """Create maximally mixed state I/2^n.

        Args:
            num_qubits: Number of qubits

        Returns:
            Maximally mixed density matrix
        """

        dim = 2**num_qubits
        density = np.eye(dim, dtype=complex) / dim
        return cls(density, num_qubits)

    def is_pure(self, tol: float = 1e-10) -> bool:
        """Check if state is pure.

        A state is pure if Tr(ρ²) = 1.

        Args:
            tol: Numerical tolerance

        Returns:
            True if state is pure
        """

        purity = np.trace(self.data @ self.data).real
        return abs(purity - 1.0) < tol

    def purity(self) -> float:
        """Calculate purity Tr(ρ²).

        Returns:
            Purity value (1 for pure states, 1/2^n for maximally mixed)
        """

        return np.trace(self.data @ self.data).real

    def trace(self) -> complex:
        """Calculate trace of density matrix.

        Returns:
            Trace (should be 1 for valid density matrix)
        """

        return np.trace(self.data)

    def expectation_value(self, operator: np.ndarray) -> float:
        """Compute expectation value Tr(ρO).

        Args:
            operator: Observable operator

        Returns:
            Expectation value
        """

        return np.trace(self.data @ operator).real

    def partial_trace(self, qubits_to_trace: List[int]) -> "DensityMatrix":
        """Compute partial trace over specified qubits.

        Args:
            qubits_to_trace: List of qubit indices to trace out

        Returns:
            Reduced density matrix
        """

        # Simplified partial trace implementation
        # Full implementation requires tensor reshaping
        n = self.num_qubits
        remaining_qubits = n - len(qubits_to_trace)

        if remaining_qubits <= 0:
            # Trace out everything - return trace as 1x1 matrix
            trace_val = self.trace()
            return DensityMatrix(np.array([[trace_val]], dtype=complex), 0)

        # For now, return a reduced matrix of appropriate size
        # Full implementation would properly trace out specified qubits
        reduced_dim = 2**remaining_qubits
        reduced = np.zeros((reduced_dim, reduced_dim), dtype=complex)

        # Simplified: take upper-left block (placeholder)
        reduced = self.data[:reduced_dim, :reduced_dim]

        return DensityMatrix(reduced, remaining_qubits)

    def evolve(self, unitary: np.ndarray) -> "DensityMatrix":
        """Evolve density matrix with unitary: ρ → UρU†.

        Args:
            unitary: Unitary operator

        Returns:
            New evolved density matrix
        """

        evolved = unitary @ self.data @ np.conj(unitary.T)
        return DensityMatrix(evolved, self.num_qubits)

    def apply_channel(self, kraus_ops: List[np.ndarray]) -> "DensityMatrix":
        """Apply quantum channel using Kraus operators.

        ρ → Σ_k E_k ρ E_k†

        Args:
            kraus_ops: List of Kraus operators

        Returns:
            New density matrix after channel application
        """

        result = np.zeros_like(self.data)
        for E in kraus_ops:
            result += E @ self.data @ np.conj(E.T)
        return DensityMatrix(result, self.num_qubits)

    def copy(self) -> "DensityMatrix":
        """Create a copy of the density matrix.

        Returns:
            New DensityMatrix with copied data
        """

        return DensityMatrix(self.data.copy(), self.num_qubits)

    def __repr__(self) -> str:
        """String representation."""

        return f"DensityMatrix({self.num_qubits} qubits, purity={self.purity():.4f})"

    def __str__(self) -> str:
        """Human-readable string representation."""

        lines = [f"DensityMatrix with {self.num_qubits} qubits:"]
        lines.append(f"  Purity: {self.purity():.6f}")
        lines.append(f"  Trace: {self.trace():.6f}")
        lines.append(f"  Pure state: {self.is_pure()}")
        return "\n".join(lines)


__all__ = ["DensityMatrix"]
