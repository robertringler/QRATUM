"""QRATUM state vector representation and operations.

Provides state vector representation for quantum states with
efficient operations and measurement support.
"""

from typing import List, Optional

import numpy as np


class StateVector:
    """Quantum state vector representation.

    Attributes:
        data: Complex numpy array representing the state
        num_qubits: Number of qubits in the state
    """

    def __init__(self, data: np.ndarray, num_qubits: Optional[int] = None):
        """Initialize state vector.

        Args:
            data: Complex numpy array representing the state
            num_qubits: Number of qubits (inferred from data if not provided)

        Raises:
            ValueError: If state dimension is not a power of 2
        """

        self.data = np.asarray(data, dtype=complex)
        if num_qubits is None:
            num_qubits = int(np.log2(len(data)))
        if 2**num_qubits != len(data):
            raise ValueError(f"State dimension {len(data)} is not a power of 2")
        self.num_qubits = num_qubits

    @classmethod
    def zero_state(cls, num_qubits: int) -> "StateVector":
        """Create |0...0⟩ state.

        Args:
            num_qubits: Number of qubits

        Returns:
            StateVector in |0...0⟩ state
        """

        data = np.zeros(2**num_qubits, dtype=complex)
        data[0] = 1.0
        return cls(data, num_qubits)

    @classmethod
    def random_state(cls, num_qubits: int, seed: Optional[int] = None) -> "StateVector":
        """Create random normalized state.

        Args:
            num_qubits: Number of qubits
            seed: Random seed for reproducibility

        Returns:
            Random StateVector
        """

        if seed is not None:
            np.random.seed(seed)
        data = np.random.randn(2**num_qubits) + 1j * np.random.randn(2**num_qubits)
        data /= np.linalg.norm(data)
        return cls(data, num_qubits)

    def normalize(self) -> "StateVector":
        """Normalize the state vector.

        Returns:
            Self (for chaining)
        """

        norm = np.linalg.norm(self.data)
        if norm > 0:
            self.data /= norm
        return self

    def inner_product(self, other: "StateVector") -> complex:
        """Compute inner product with another state.

        Args:
            other: Another StateVector

        Returns:
            Complex inner product ⟨self|other⟩

        Raises:
            ValueError: If states have different dimensions
        """

        if self.num_qubits != other.num_qubits:
            raise ValueError("States must have same number of qubits")
        return np.vdot(self.data, other.data)

    def tensor_product(self, other: "StateVector") -> "StateVector":
        """Compute tensor product with another state.

        Args:
            other: Another StateVector

        Returns:
            Tensor product state |self⟩ ⊗ |other⟩
        """

        data = np.kron(self.data, other.data)
        return StateVector(data, self.num_qubits + other.num_qubits)

    def probabilities(self) -> np.ndarray:
        """Get measurement probabilities for all basis states.

        Returns:
            Array of probabilities for each computational basis state
        """

        return np.abs(self.data) ** 2

    def expectation_value(self, operator: np.ndarray) -> float:
        """Compute expectation value of an operator.

        Args:
            operator: Hermitian operator matrix

        Returns:
            Real expectation value ⟨ψ|O|ψ⟩
        """

        return np.real(np.vdot(self.data, operator @ self.data))

    def partial_trace(self, qubits_to_trace: List[int]) -> np.ndarray:
        """Compute partial trace over specified qubits.

        Args:
            qubits_to_trace: List of qubit indices to trace out

        Returns:
            Reduced density matrix
        """

        # Create density matrix
        density = np.outer(self.data, np.conj(self.data))

        # Perform partial trace (simplified implementation)
        # Full implementation would require tensor reshaping
        # For now, return full density matrix
        return density

    def copy(self) -> "StateVector":
        """Create a copy of the state vector.

        Returns:
            New StateVector with copied data
        """

        return StateVector(self.data.copy(), self.num_qubits)

    def __repr__(self) -> str:
        """String representation of state vector."""

        return f"StateVector({self.num_qubits} qubits, dim={len(self.data)})"

    def __str__(self) -> str:
        """Human-readable string representation."""

        lines = [f"StateVector with {self.num_qubits} qubits:"]
        probs = self.probabilities()
        # Show only non-zero amplitudes
        for i, (amp, prob) in enumerate(zip(self.data, probs)):
            if prob > 1e-10:
                basis = format(i, f"0{self.num_qubits}b")
                lines.append(f"  |{basis}⟩: {amp:.4f} (prob: {prob:.4f})")
        return "\n".join(lines)

    def compress(self, fidelity: float = 0.995) -> 'CompressedStateVector':
        """Return AHTC-compressed representation.

        Args:
            fidelity: Target fidelity (default: 0.995 for 99.5%)

        Returns:
            CompressedStateVector object

        Example:
            >>> sv = StateVector.random_state(10)
            >>> compressed = sv.compress(fidelity=0.995)
        """
        from quasim.holo.anti_tensor import compress

        compressed_data, achieved_fidelity, metadata = compress(
            self.data, fidelity=fidelity
        )

        return CompressedStateVector(
            compressed_data=compressed_data,
            num_qubits=self.num_qubits,
            fidelity=achieved_fidelity,
            metadata=metadata,
        )

    @classmethod
    def from_compressed(cls, compressed: 'CompressedStateVector') -> 'StateVector':
        """Reconstruct from compressed format.

        Args:
            compressed: CompressedStateVector object

        Returns:
            Decompressed StateVector

        Example:
            >>> compressed = sv.compress()
            >>> recovered = StateVector.from_compressed(compressed)
        """
        from quasim.holo.anti_tensor import decompress

        data = decompress(compressed.compressed_data)
        return cls(data, compressed.num_qubits)


class CompressedStateVector:
    """Compressed state vector using AHTC.

    Attributes:
        compressed_data: Compressed representation dictionary
        num_qubits: Number of qubits in original state
        fidelity: Achieved compression fidelity
        metadata: Compression metadata
    """

    def __init__(
        self,
        compressed_data: dict,
        num_qubits: int,
        fidelity: float,
        metadata: dict,
    ):
        """Initialize compressed state vector.

        Args:
            compressed_data: Compressed data from AHTC
            num_qubits: Number of qubits
            fidelity: Achieved fidelity
            metadata: Compression metadata
        """
        self.compressed_data = compressed_data
        self.num_qubits = num_qubits
        self.fidelity = fidelity
        self.metadata = metadata

    def decompress(self) -> StateVector:
        """Decompress to StateVector.

        Returns:
            Decompressed StateVector
        """
        return StateVector.from_compressed(self)

    def __repr__(self) -> str:
        """String representation."""
        ratio = self.metadata.get('compression_ratio', 0)
        return (
            f"CompressedStateVector({self.num_qubits} qubits, "
            f"fidelity={self.fidelity:.4f}, ratio={ratio:.2f}x)"
        )


__all__ = ["StateVector", "CompressedStateVector"]
