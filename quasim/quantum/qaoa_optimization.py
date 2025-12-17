"""Quantum Approximate Optimization Algorithm (QAOA) for combinatorial problems.

QAOA is a hybrid quantum-classical algorithm for solving combinatorial optimization
problems on NISQ-era quantum computers.

Problem types supported:
- MaxCut: Graph partitioning problem
- Ising models: Spin glass optimization (materials science proxy)
- Traveling Salesman: Small instances only

Limitations:
- Circuit depth grows with problem size and p-layers
- NISQ noise limits practical problem sizes to ~10-20 variables
- Approximation quality depends on p (typically p=3-10)
- May not outperform classical algorithms for small problems

References:
- Farhi et al., "A Quantum Approximate Optimization Algorithm" (2014)
- Guerreschi & Smelyanskiy, "Practical optimization for hybrid algorithms" (2017)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Optional dependencies
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.primitives import Sampler
    from qiskit.quantum_info import SparsePauliOp

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

from .core import QuantumBackend, QuantumConfig


@dataclass
class QAOAResult:
    """Result of QAOA optimization.

    Attributes:
        solution: Best solution bitstring
        energy: Energy (cost) of best solution
        optimal_params: Optimal QAOA parameters (gammas, betas)
        approximation_ratio: Ratio to optimal solution (if known)
        n_iterations: Number of optimization iterations
        success: Whether optimization converged
        classical_optimal: Classical optimal energy (if available)
        prob_distribution: Probability distribution over solutions
    """

    solution: str
    energy: float
    optimal_params: np.ndarray
    approximation_ratio: float | None = None
    n_iterations: int = 0
    success: bool = True
    classical_optimal: float | None = None
    prob_distribution: dict[str, float] | None = None

    def __repr__(self) -> str:
        lines = [
            f"QAOAResult(solution={self.solution}",
            f"  energy={self.energy:.4f}",
            f"  iterations={self.n_iterations}",
        ]
        if self.approximation_ratio is not None:
            lines.append(f"  approx_ratio={self.approximation_ratio:.4f}")
        if self.classical_optimal is not None:
            lines.append(f"  classical_optimal={self.classical_optimal:.4f}")
        return "\n".join(lines) + ")"


class QAOA:
    """Quantum Approximate Optimization Algorithm implementation.

    QAOA alternates between applying:
    1. Problem Hamiltonian (cost function encoding)
    2. Mixer Hamiltonian (exploration operator)

    The circuit depth grows with p (number of layers), and parameters
    are optimized classically to minimize the cost function expectation.

    Example:
        >>> config = QuantumConfig(shots=1024)
        >>> qaoa = QAOA(config, p_layers=3)
        >>> # Solve MaxCut on small graph
        >>> edges = [(0,1), (1,2), (2,3), (3,0)]
        >>> result = qaoa.solve_maxcut(edges)
        >>> print(f"Best cut: {result.solution}, value: {result.energy}")
    """

    def __init__(self, config: QuantumConfig, p_layers: int = 3):
        """Initialize QAOA solver.

        Args:
            config: Quantum configuration
            p_layers: Number of QAOA layers (depth). Typical range: 1-10

        Raises:
            ImportError: If Qiskit not available
        """

        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required. Install with: pip install qiskit qiskit-aer")

        if p_layers < 1:
            raise ValueError("p_layers must be at least 1")

        self.config = config
        self.backend = QuantumBackend(config)
        self.p_layers = p_layers

        # Use Sampler primitive for bitstring sampling
        self.sampler = Sampler()

    def solve_maxcut(
        self,
        edges: list[tuple[int, int]],
        optimizer: str = "COBYLA",
        max_iterations: int = 100,
        classical_reference: bool = True,
    ) -> QAOAResult:
        """Solve MaxCut problem using QAOA.

        MaxCut: Partition graph nodes into two sets to maximize edges between sets.
        This is NP-hard for general graphs but solvable for small instances.

        Args:
            edges: List of (node_i, node_j) edges
            optimizer: Classical optimizer name
            max_iterations: Maximum optimization iterations
            classical_reference: Compute classical optimal for comparison

        Returns:
            QAOAResult with best solution and optimization details
        """

        # Determine number of nodes
        n_nodes = max(max(e) for e in edges) + 1

        print(f"MaxCut problem: {n_nodes} nodes, {len(edges)} edges")
        print(f"QAOA layers: {self.p_layers}")
        print(f"Backend: {self.backend.backend_name}")

        # Build MaxCut Hamiltonian
        hamiltonian = self._build_maxcut_hamiltonian(edges, n_nodes)

        # Get classical optimal if requested
        classical_optimal = None
        if classical_reference:
            classical_optimal = self._solve_maxcut_classical(edges, n_nodes)
            print(f"Classical optimal cut value: {classical_optimal:.4f}")

        # Create QAOA circuit
        circuit = self._create_qaoa_circuit(hamiltonian, n_nodes)

        # Optimize parameters
        result = self._optimize_qaoa(
            circuit, hamiltonian, n_nodes, optimizer=optimizer, max_iterations=max_iterations
        )

        # Add classical reference
        if classical_optimal is not None:
            result.classical_optimal = classical_optimal
            # MaxCut: higher is better, so ratio = qaoa_value / classical_value
            result.approximation_ratio = abs(result.energy) / abs(classical_optimal)

        return result

    def solve_ising(
        self,
        coupling_matrix: np.ndarray,
        external_field: np.ndarray | None = None,
        optimizer: str = "COBYLA",
        max_iterations: int = 100,
    ) -> QAOAResult:
        """Solve Ising spin glass problem using QAOA.

        Ising model: H = -Σ J_ij σ_i σ_j - Σ h_i σ_i
        where σ_i ∈ {-1, +1}

        This is a proxy for materials science problems like:
        - Lattice defect optimization
        - Spin glass ground states
        - Magnetic ordering

        Args:
            coupling_matrix: J_ij coupling coefficients (NxN symmetric)
            external_field: h_i external field (length N), optional
            optimizer: Classical optimizer
            max_iterations: Maximum iterations

        Returns:
            QAOAResult with best spin configuration
        """

        n_spins = len(coupling_matrix)

        print(f"Ising model: {n_spins} spins")
        print(f"QAOA layers: {self.p_layers}")

        # Build Ising Hamiltonian
        hamiltonian = self._build_ising_hamiltonian(coupling_matrix, external_field)

        # Create QAOA circuit
        circuit = self._create_qaoa_circuit(hamiltonian, n_spins)

        # Optimize
        result = self._optimize_qaoa(
            circuit, hamiltonian, n_spins, optimizer=optimizer, max_iterations=max_iterations
        )

        return result

    def _build_maxcut_hamiltonian(
        self, edges: list[tuple[int, int]], n_nodes: int
    ) -> SparsePauliOp:
        """Build MaxCut Hamiltonian as Pauli operator.

        MaxCut Hamiltonian: H = Σ_{(i,j) ∈ E} 0.5 * (1 - Z_i Z_j)
        Minimizing this maximizes the cut value.
        """

        pauli_list = []

        for i, j in edges:
            # Add Z_i Z_j term
            pauli_str = ["I"] * n_nodes
            pauli_str[i] = "Z"
            pauli_str[j] = "Z"
            pauli_list.append(("".join(pauli_str), -0.5))

            # Add constant offset (contributes 0.5 per edge)
            pauli_list.append(("I" * n_nodes, 0.5))

        return SparsePauliOp.from_list(pauli_list)

    def _build_ising_hamiltonian(
        self, coupling_matrix: np.ndarray, external_field: np.ndarray | None = None
    ) -> SparsePauliOp:
        """Build Ising Hamiltonian as Pauli operator.

        H = -Σ J_ij Z_i Z_j - Σ h_i Z_i
        """

        n = len(coupling_matrix)
        pauli_list = []

        # Coupling terms
        for i in range(n):
            for j in range(i + 1, n):
                if coupling_matrix[i, j] != 0:
                    pauli_str = ["I"] * n
                    pauli_str[i] = "Z"
                    pauli_str[j] = "Z"
                    pauli_list.append(("".join(pauli_str), -coupling_matrix[i, j]))

        # External field terms
        if external_field is not None:
            for i, h_i in enumerate(external_field):
                if h_i != 0:
                    pauli_str = ["I"] * n
                    pauli_str[i] = "Z"
                    pauli_list.append(("".join(pauli_str), -h_i))

        return (
            SparsePauliOp.from_list(pauli_list)
            if pauli_list
            else SparsePauliOp.from_list([("I" * n, 0.0)])
        )

    def _create_qaoa_circuit(self, hamiltonian: SparsePauliOp, n_qubits: int) -> QuantumCircuit:
        """Create QAOA ansatz circuit with p layers.

        QAOA circuit structure:
        1. Initial state: uniform superposition |+>^n
        2. For each layer p:
           a. Apply problem Hamiltonian with parameter γ_p
           b. Apply mixer Hamiltonian with parameter β_p
        3. Measure all qubits
        """

        circuit = QuantumCircuit(n_qubits)

        # Initial state: uniform superposition
        circuit.h(range(n_qubits))

        # QAOA layers
        gammas = [Parameter(f"γ_{p}") for p in range(self.p_layers)]
        betas = [Parameter(f"β_{p}") for p in range(self.p_layers)]

        for p in range(self.p_layers):
            # Problem Hamiltonian (cost) layer
            self._apply_hamiltonian_layer(circuit, hamiltonian, gammas[p])

            # Mixer Hamiltonian layer (X rotations)
            for i in range(n_qubits):
                circuit.rx(2 * betas[p], i)

        # Measurements
        circuit.measure_all()

        return circuit

    def _apply_hamiltonian_layer(
        self, circuit: QuantumCircuit, hamiltonian: SparsePauliOp, gamma: Parameter
    ) -> None:
        """Apply problem Hamiltonian evolution e^(-iγH)."""

        for pauli_str, coeff in hamiltonian.to_list():
            if pauli_str == "I" * len(pauli_str):
                # Identity term contributes only global phase
                continue

            # Decompose Pauli string into gates
            self._apply_pauli_evolution(circuit, pauli_str, gamma * coeff)

    def _apply_pauli_evolution(self, circuit: QuantumCircuit, pauli_str: str, angle: Any) -> None:
        """Apply evolution under Pauli string."""

        n = len(pauli_str)

        # Find qubits with non-identity operators
        z_qubits = [i for i in range(n) if pauli_str[i] == "Z"]
        x_qubits = [i for i in range(n) if pauli_str[i] == "X"]
        y_qubits = [i for i in range(n) if pauli_str[i] == "Y"]

        # Convert X and Y to Z using basis rotations
        for i in x_qubits:
            circuit.h(i)
        for i in y_qubits:
            circuit.sdg(i)
            circuit.h(i)

        # Apply ZZ...Z rotation
        active_qubits = z_qubits + x_qubits + y_qubits
        if len(active_qubits) > 1:
            for i in range(len(active_qubits) - 1):
                circuit.cx(active_qubits[i], active_qubits[i + 1])

        if active_qubits:
            circuit.rz(2 * angle, active_qubits[-1])

        if len(active_qubits) > 1:
            for i in range(len(active_qubits) - 2, -1, -1):
                circuit.cx(active_qubits[i], active_qubits[i + 1])

        # Reverse basis rotations
        for i in y_qubits:
            circuit.h(i)
            circuit.s(i)
        for i in x_qubits:
            circuit.h(i)

    def _optimize_qaoa(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        n_qubits: int,
        optimizer: str = "COBYLA",
        max_iterations: int = 100,
    ) -> QAOAResult:
        """Optimize QAOA parameters to minimize cost function."""

        evaluation_count = [0]
        best_result = {"energy": float("inf"), "bitstring": None, "counts": None}

        def cost_function(params: np.ndarray) -> float:
            """Evaluate cost function expectation."""

            evaluation_count[0] += 1

            # Run circuit with current parameters
            job = self.sampler.run(circuit, params)
            result = job.result()

            # Get measurement counts
            quasi_dists = result.quasi_dists[0]

            # Compute expectation value
            energy = 0.0
            for bitstring_int, prob in quasi_dists.items():
                bitstring = format(bitstring_int, f"0{n_qubits}b")
                # Evaluate Hamiltonian for this bitstring
                bitstring_energy = self._evaluate_bitstring_energy(bitstring, hamiltonian)
                energy += prob * bitstring_energy

            # Track best solution
            if bitstring_energy < best_result["energy"]:
                best_result["energy"] = bitstring_energy
                best_result["bitstring"] = bitstring
                best_result["counts"] = quasi_dists

            if evaluation_count[0] % 10 == 0:
                print(f"  Evaluation {evaluation_count[0]}: E = {energy:.4f}")

            return energy

        # Initial parameters
        np.random.seed(self.config.seed)
        initial_params = np.random.uniform(-np.pi, np.pi, 2 * self.p_layers)

        # Optimize
        from scipy.optimize import minimize

        result = minimize(
            cost_function, initial_params, method=optimizer, options={"maxiter": max_iterations}
        )

        return QAOAResult(
            solution=best_result["bitstring"],
            energy=best_result["energy"],
            optimal_params=result.x,
            n_iterations=evaluation_count[0],
            success=result.success,
            prob_distribution=best_result["counts"],
        )

    def _evaluate_bitstring_energy(self, bitstring: str, hamiltonian: SparsePauliOp) -> float:
        """Evaluate Hamiltonian energy for a classical bitstring.

        Convert bitstring to spin configuration and compute energy.
        """

        # Convert bitstring to spin values (0 -> +1, 1 -> -1)
        spins = np.array([1 if b == "0" else -1 for b in bitstring])

        energy = 0.0
        for pauli_str, coeff in hamiltonian.to_list():
            term_value = 1.0
            for i, pauli in enumerate(pauli_str):
                if pauli == "Z":
                    term_value *= spins[i]
                elif pauli == "X":
                    # For computational basis states, X gives 0
                    term_value = 0.0
                    break
                # 'I' contributes 1.0
            energy += coeff * term_value

        return energy

    def _solve_maxcut_classical(self, edges: list[tuple[int, int]], n_nodes: int) -> float:
        """Solve MaxCut exactly using brute force (small instances only)."""

        max_cut = 0

        # Try all 2^n partitions
        for partition in range(2**n_nodes):
            # Partition is encoded as binary
            nodes_in_set_a = [i for i in range(n_nodes) if (partition >> i) & 1]

            # Count edges crossing partition
            cut_size = sum(1 for i, j in edges if ((i in nodes_in_set_a) != (j in nodes_in_set_a)))

            max_cut = max(max_cut, cut_size)

        return float(max_cut)


# Export main classes
__all__ = ["QAOA", "QAOAResult"]
