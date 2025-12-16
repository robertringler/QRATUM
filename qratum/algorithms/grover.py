"""Grover's search algorithm implementation.

Grover's algorithm provides quadratic speedup for unstructured search,
finding a marked element in O(√N) queries instead of O(N).
"""

from typing import List, Optional

import numpy as np

from qratum.core.circuit import Circuit
from qratum.core.measurement import Result
from qratum.core.simulator import Simulator


class Grover:
    """Grover's quantum search algorithm.

    Attributes:
        num_qubits: Number of qubits (search space size = 2^n)
        marked_states: List of marked state indices to find
        iterations: Number of Grover iterations
    """

    def __init__(self, num_qubits: int, marked_states: List[int], iterations: Optional[int] = None):
        """Initialize Grover search.

        Args:
            num_qubits: Number of qubits
            marked_states: List of indices of marked states
            iterations: Number of Grover iterations (auto-calculated if None)
        """
        self.num_qubits = num_qubits
        self.marked_states = marked_states
        self.search_space_size = 2**num_qubits

        # Calculate optimal number of iterations if not provided
        if iterations is None:
            # Optimal: π/4 * sqrt(N/M) where N = search space, M = marked items
            m = len(marked_states)
            self.iterations = int(np.pi / 4 * np.sqrt(self.search_space_size / m))
        else:
            self.iterations = iterations

    def build_circuit(self) -> Circuit:
        """Build Grover search circuit.

        Returns:
            Complete Grover circuit
        """
        circuit = Circuit(self.num_qubits)

        # Step 1: Initialize to uniform superposition
        for i in range(self.num_qubits):
            circuit.h(i)

        # Step 2: Apply Grover iterations
        for _ in range(self.iterations):
            # Oracle: mark target states
            circuit = self._apply_oracle(circuit)

            # Diffusion operator: amplify marked states
            circuit = self._apply_diffusion(circuit)

        return circuit

    def _apply_oracle(self, circuit: Circuit) -> Circuit:
        """Apply oracle that marks target states.

        This simplified implementation marks states by applying Z gates.
        A full implementation would use controlled multi-qubit operations.

        Args:
            circuit: Circuit to add oracle to

        Returns:
            Circuit with oracle applied
        """
        # For each marked state, apply phase flip
        for marked in self.marked_states:
            # Convert index to binary representation
            binary = format(marked, f"0{self.num_qubits}b")

            # Apply X gates to flip qubits that should be 0
            for i, bit in enumerate(binary):
                if bit == "0":
                    circuit.x(i)

            # Apply multi-controlled Z (simplified: single Z on last qubit)
            # Full implementation would use multi-controlled Z
            circuit.z(self.num_qubits - 1)

            # Undo X gates
            for i, bit in enumerate(binary):
                if bit == "0":
                    circuit.x(i)

        return circuit

    def _apply_diffusion(self, circuit: Circuit) -> Circuit:
        """Apply diffusion operator (inversion about average).

        The diffusion operator is: 2|s⟩⟨s| - I
        where |s⟩ is the uniform superposition.

        Args:
            circuit: Circuit to add diffusion to

        Returns:
            Circuit with diffusion applied
        """
        # H on all qubits
        for i in range(self.num_qubits):
            circuit.h(i)

        # X on all qubits
        for i in range(self.num_qubits):
            circuit.x(i)

        # Multi-controlled Z (simplified)
        # Full implementation uses multi-controlled Z
        circuit.z(self.num_qubits - 1)

        # X on all qubits
        for i in range(self.num_qubits):
            circuit.x(i)

        # H on all qubits
        for i in range(self.num_qubits):
            circuit.h(i)

        return circuit

    def run(self, simulator: Simulator, shots: int = 1024) -> Result:
        """Run Grover's algorithm and measure results.

        Args:
            simulator: Quantum simulator
            shots: Number of measurement shots

        Returns:
            Measurement results
        """
        circuit = self.build_circuit()
        result = simulator.run(circuit, shots=shots)
        return result

    def find_marked_states(self, simulator: Simulator, shots: int = 1024) -> List[int]:
        """Run search and return most likely marked states.

        Args:
            simulator: Quantum simulator
            shots: Number of measurement shots

        Returns:
            List of found state indices (most frequently measured)
        """
        result = self.run(simulator, shots=shots)

        # Get most frequent states
        counts = result.get_counts()
        sorted_states = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        # Convert binary strings to integers
        found_states = []
        for state_str, _ in sorted_states[: len(self.marked_states)]:
            state_int = int(state_str, 2)
            found_states.append(state_int)

        return found_states

    def success_probability(self) -> float:
        """Calculate theoretical success probability.

        Returns:
            Probability of measuring a marked state
        """
        m = len(self.marked_states)
        n = self.search_space_size
        theta = np.arcsin(np.sqrt(m / n))
        prob = np.sin((2 * self.iterations + 1) * theta) ** 2
        return prob

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Grover({self.num_qubits} qubits, "
            f"{len(self.marked_states)} marked, "
            f"{self.iterations} iterations)"
        )


def grover_search(
    num_qubits: int,
    marked_states: List[int],
    backend: str = "cpu",
    shots: int = 1024,
    seed: Optional[int] = None,
) -> Result:
    """Convenience function to run Grover search.

    Args:
        num_qubits: Number of qubits
        marked_states: List of marked state indices
        backend: Simulation backend
        shots: Number of measurements
        seed: Random seed

    Returns:
        Measurement results
    """
    grover = Grover(num_qubits, marked_states)
    simulator = Simulator(backend=backend, seed=seed)
    return grover.run(simulator, shots=shots)


__all__ = ["Grover", "grover_search"]
