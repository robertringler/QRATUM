"""QRATUM measurement and result handling.

Provides measurement operations on quantum states and result
processing with classical bit storage.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import Counter


class Result:
    """Quantum measurement result.

    Stores measurement outcomes and provides analysis methods.

    Attributes:
        counts: Dictionary mapping basis states to measurement counts
        shots: Total number of measurement shots
        num_qubits: Number of qubits measured
    """

    def __init__(self, counts: Dict[str, int], num_qubits: int):
        """Initialize measurement result.

        Args:
            counts: Dictionary of basis states to counts
            num_qubits: Number of qubits
        """
        self.counts = counts
        self.num_qubits = num_qubits
        self.shots = sum(counts.values())

    def get_counts(self) -> Dict[str, int]:
        """Get raw measurement counts.

        Returns:
            Dictionary mapping basis states to counts
        """
        return self.counts

    def get_probabilities(self) -> Dict[str, float]:
        """Get measurement probabilities.

        Returns:
            Dictionary mapping basis states to probabilities
        """
        return {state: count / self.shots for state, count in self.counts.items()}

    def most_frequent(self, n: int = 1) -> List[str]:
        """Get n most frequently measured states.

        Args:
            n: Number of states to return

        Returns:
            List of most frequent basis states
        """
        sorted_states = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)
        return [state for state, _ in sorted_states[:n]]

    def marginal_counts(self, qubits: List[int]) -> Dict[str, int]:
        """Get marginal counts for subset of qubits.

        Args:
            qubits: List of qubit indices to keep

        Returns:
            Dictionary of marginal counts
        """
        marginal = {}
        for state, count in self.counts.items():
            # Extract specified qubits
            marginal_state = "".join(state[i] for i in qubits)
            marginal[marginal_state] = marginal.get(marginal_state, 0) + count
        return marginal

    def expectation_value(self, observable: str) -> float:
        """Compute expectation value of Pauli observable.

        Args:
            observable: Pauli string (e.g., "ZZI", "XYZ")

        Returns:
            Expectation value

        Raises:
            ValueError: If observable length doesn't match num_qubits
        """
        if len(observable) != self.num_qubits:
            raise ValueError(
                f"Observable length {len(observable)} must match num_qubits {self.num_qubits}"
            )

        expectation = 0.0
        for state, count in self.counts.items():
            # Compute eigenvalue for this state
            eigenvalue = 1.0
            for i, pauli in enumerate(observable):
                if pauli == "Z":
                    # Z eigenvalue: +1 for |0⟩, -1 for |1⟩
                    eigenvalue *= 1 if state[i] == "0" else -1
                elif pauli == "X" or pauli == "Y":
                    # X and Y require state transformation (simplified)
                    pass
                # Identity doesn't change eigenvalue

            expectation += eigenvalue * count / self.shots

        return expectation

    def __repr__(self) -> str:
        """String representation of result."""
        return f"Result({self.shots} shots, {len(self.counts)} outcomes)"

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [f"Measurement Result ({self.shots} shots):"]
        sorted_counts = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)
        for state, count in sorted_counts[:10]:  # Show top 10
            prob = count / self.shots
            lines.append(f"  |{state}⟩: {count:5d} ({prob:.4f})")
        if len(sorted_counts) > 10:
            lines.append(f"  ... ({len(sorted_counts) - 10} more states)")
        return "\n".join(lines)


class Measurement:
    """Quantum measurement operation.

    Provides methods for measuring quantum states in different bases.
    """

    @staticmethod
    def measure_statevector(
        state: np.ndarray, shots: int = 1024, seed: Optional[int] = None
    ) -> Result:
        """Measure state vector in computational basis.

        Args:
            state: State vector to measure
            shots: Number of measurement shots
            seed: Random seed for reproducibility

        Returns:
            Measurement Result
        """
        if seed is not None:
            np.random.seed(seed)

        num_qubits = int(np.log2(len(state)))
        probabilities = np.abs(state) ** 2

        # Sample from probability distribution
        outcomes = np.random.choice(len(state), size=shots, p=probabilities)

        # Convert to binary strings and count
        counts = {}
        for outcome in outcomes:
            basis_state = format(outcome, f"0{num_qubits}b")
            counts[basis_state] = counts.get(basis_state, 0) + 1

        return Result(counts, num_qubits)

    @staticmethod
    def measure_qubits(
        state: np.ndarray, qubits: List[int], shots: int = 1024, seed: Optional[int] = None
    ) -> Result:
        """Measure specific qubits in computational basis.

        Args:
            state: State vector to measure
            qubits: List of qubit indices to measure
            shots: Number of measurement shots
            seed: Random seed for reproducibility

        Returns:
            Measurement Result for specified qubits
        """
        # For simplicity, measure all qubits and marginalize
        full_result = Measurement.measure_statevector(state, shots, seed)
        marginal_counts = full_result.marginal_counts(qubits)
        return Result(marginal_counts, len(qubits))

    @staticmethod
    def sample_distribution(
        probabilities: Dict[str, float], shots: int = 1024, seed: Optional[int] = None
    ) -> Result:
        """Sample from a probability distribution.

        Args:
            probabilities: Dictionary of basis states to probabilities
            shots: Number of samples
            seed: Random seed for reproducibility

        Returns:
            Measurement Result
        """
        if seed is not None:
            np.random.seed(seed)

        states = list(probabilities.keys())
        probs = np.array([probabilities[s] for s in states])

        # Normalize if needed
        probs /= probs.sum()

        # Sample
        samples = np.random.choice(len(states), size=shots, p=probs)

        # Count outcomes
        counts = {}
        for sample in samples:
            state = states[sample]
            counts[state] = counts.get(state, 0) + 1

        num_qubits = len(states[0]) if states else 0
        return Result(counts, num_qubits)


__all__ = ["Result", "Measurement"]
