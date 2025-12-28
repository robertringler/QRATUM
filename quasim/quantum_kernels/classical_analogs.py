"""Classical Analogs of Quantum Algorithms.

This module provides classical implementations of quantum algorithms
that capture some of the algorithmic structure while remaining
fully verifiable and deterministic.

Key algorithms:
- ClassicalVQE: Classical analog of Variational Quantum Eigensolver
- ClassicalQAOA: Classical analog of Quantum Approximate Optimization Algorithm

These implementations are useful for:
- Benchmarking quantum advantage
- Prototyping before quantum execution
- Providing fallback for NISQ-limited problems
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class ClassicalVQEResult:
    """Result of classical VQE computation.

    Attributes:
        energy: Ground state energy estimate
        optimal_params: Optimal variational parameters
        n_iterations: Number of optimization iterations
        n_evaluations: Total function evaluations
        success: Whether optimization converged
        execution_time: Total execution time in seconds
        method: Optimization method used
        execution_id: Unique identifier
    """

    energy: float
    optimal_params: np.ndarray
    n_iterations: int
    n_evaluations: int
    success: bool
    execution_time: float = 0.0
    method: str = "classical_vqe"
    execution_id: str = ""

    def __post_init__(self) -> None:
        if not self.execution_id:
            self.execution_id = str(uuid.uuid4())


@dataclass
class ClassicalQAOAResult:
    """Result of classical QAOA computation.

    Attributes:
        solution: Best solution found
        cost: Cost/energy of best solution
        optimal_params: Optimal QAOA-like parameters
        approximation_ratio: Ratio to optimal (if known)
        n_iterations: Number of optimization iterations
        success: Whether optimization converged
        execution_time: Total execution time
        method: Method used
        execution_id: Unique identifier
    """

    solution: np.ndarray
    cost: float
    optimal_params: np.ndarray
    approximation_ratio: float | None = None
    n_iterations: int = 0
    success: bool = True
    execution_time: float = 0.0
    method: str = "classical_qaoa"
    execution_id: str = ""

    def __post_init__(self) -> None:
        if not self.execution_id:
            self.execution_id = str(uuid.uuid4())


class ClassicalVQE:
    """Classical analog of Variational Quantum Eigensolver.

    This class implements a classical optimization approach that mimics
    VQE's variational structure without quantum circuits. It uses
    parameterized classical ansätze and gradient-based optimization.

    Applications:
    - Benchmarking quantum VQE implementations
    - Prototyping molecular simulations
    - Fallback for problems too large for NISQ devices

    Example:
        >>> vqe = ClassicalVQE(seed=42)
        >>> # Define Hamiltonian as matrix
        >>> H = np.array([[1, 0.5], [0.5, -1]])
        >>> result = vqe.minimize_energy(H, n_params=4)
        >>> print(f"Ground state energy: {result.energy:.6f}")
    """

    def __init__(
        self,
        seed: int | None = 42,
        optimizer: str = "L-BFGS-B",
        max_iterations: int = 1000,
    ):
        """Initialize classical VQE.

        Args:
            seed: Random seed for reproducibility
            optimizer: Scipy optimizer name
            max_iterations: Maximum optimization iterations
        """
        self.seed = seed
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self._rng = np.random.default_rng(seed)

    def minimize_energy(
        self,
        hamiltonian: np.ndarray,
        n_params: int = 10,
        initial_params: np.ndarray | None = None,
    ) -> ClassicalVQEResult:
        """Find minimum energy using variational optimization.

        Args:
            hamiltonian: Hermitian Hamiltonian matrix
            n_params: Number of variational parameters
            initial_params: Initial parameter values

        Returns:
            ClassicalVQEResult with optimized energy
        """
        from scipy.optimize import minimize

        start_time = time.time()

        # Initialize parameters
        if initial_params is None:
            initial_params = self._rng.uniform(-np.pi, np.pi, n_params)

        # System size from Hamiltonian
        n = hamiltonian.shape[0]

        # Track evaluations
        evaluation_count = [0]

        def energy_function(params: np.ndarray) -> float:
            """Compute energy expectation value."""
            evaluation_count[0] += 1

            # Generate state from parameters using classical ansatz
            state = self._parameterized_state(params, n)

            # Compute expectation value
            energy = np.real(state.conj() @ hamiltonian @ state)

            return float(energy)

        # Optimize
        result = minimize(
            energy_function,
            initial_params,
            method=self.optimizer,
            options={"maxiter": self.max_iterations},
        )

        execution_time = time.time() - start_time

        return ClassicalVQEResult(
            energy=result.fun,
            optimal_params=result.x,
            n_iterations=result.nit,
            n_evaluations=evaluation_count[0],
            success=result.success,
            execution_time=execution_time,
        )

    def compute_molecular_energy(
        self,
        one_body: np.ndarray,
        two_body: np.ndarray | None = None,
        n_electrons: int = 2,
    ) -> ClassicalVQEResult:
        """Compute molecular energy using Hartree-Fock-like ansatz.

        Args:
            one_body: One-body integrals (h_pq)
            two_body: Two-body integrals (g_pqrs), optional
            n_electrons: Number of electrons

        Returns:
            ClassicalVQEResult with molecular energy
        """
        start_time = time.time()

        n_orbitals = one_body.shape[0]

        # Simple Hartree-Fock: diagonalize one-body part
        eigenvalues, eigenvectors = np.linalg.eigh(one_body)

        # Occupy lowest orbitals
        occupied = eigenvalues[:n_electrons]
        hf_energy = np.sum(occupied)

        # Add two-body contribution (simplified)
        if two_body is not None:
            # Simplified: just count mean-field contribution
            for i in range(n_electrons):
                for j in range(n_electrons):
                    # Coulomb - Exchange
                    hf_energy += 0.5 * (two_body[i, j, i, j] - two_body[i, j, j, i])

        execution_time = time.time() - start_time

        return ClassicalVQEResult(
            energy=float(hf_energy),
            optimal_params=eigenvectors[:, :n_electrons].flatten(),
            n_iterations=1,
            n_evaluations=1,
            success=True,
            execution_time=execution_time,
            method="hartree_fock",
        )

    def _parameterized_state(self, params: np.ndarray, dim: int) -> np.ndarray:
        """Generate parameterized quantum-like state.

        Uses product of Givens rotations to generate arbitrary state.

        Args:
            params: Variational parameters
            dim: Hilbert space dimension

        Returns:
            Normalized state vector
        """
        # Start with |0⟩ state
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0

        # Apply parameterized rotations
        n_rotations = len(params) // 2

        for i in range(min(n_rotations, dim - 1)):
            theta = params[2 * i]
            phi = params[2 * i + 1] if 2 * i + 1 < len(params) else 0

            # Givens rotation between states i and i+1
            c = np.cos(theta)
            s = np.sin(theta) * np.exp(1j * phi)

            new_i = c * state[i] - np.conj(s) * state[i + 1]
            new_ip1 = s * state[i] + c * state[i + 1]

            state[i] = new_i
            state[i + 1] = new_ip1

        return state / np.linalg.norm(state)


class ClassicalQAOA:
    """Classical analog of Quantum Approximate Optimization Algorithm.

    This class implements classical optimization approaches that mimic
    QAOA's alternating structure for combinatorial optimization.

    Applications:
    - MaxCut on graphs
    - Ising model optimization
    - Portfolio optimization
    - Constraint satisfaction

    Example:
        >>> qaoa = ClassicalQAOA(seed=42)
        >>> # Define MaxCut problem
        >>> edges = [(0, 1), (1, 2), (2, 0)]
        >>> result = qaoa.solve_maxcut(3, edges)
        >>> print(f"Best cut: {result.solution}")
    """

    def __init__(
        self,
        seed: int | None = 42,
        p_layers: int = 3,
        n_samples: int = 1000,
    ):
        """Initialize classical QAOA.

        Args:
            seed: Random seed for reproducibility
            p_layers: Number of QAOA-like layers
            n_samples: Number of samples for optimization
        """
        self.seed = seed
        self.p_layers = p_layers
        self.n_samples = n_samples
        self._rng = np.random.default_rng(seed)

    def solve_maxcut(
        self,
        n_nodes: int,
        edges: list[tuple[int, int]],
        weights: list[float] | None = None,
    ) -> ClassicalQAOAResult:
        """Solve MaxCut problem.

        Args:
            n_nodes: Number of graph nodes
            edges: List of (i, j) edges
            weights: Edge weights (default: all 1.0)

        Returns:
            ClassicalQAOAResult with best partition
        """
        from scipy.optimize import minimize

        start_time = time.time()

        if weights is None:
            weights = [1.0] * len(edges)

        # Build adjacency matrix
        adj = np.zeros((n_nodes, n_nodes))
        for (i, j), w in zip(edges, weights):
            adj[i, j] = w
            adj[j, i] = w

        # Cost function: cut value (negative for minimization)
        def cut_cost(assignment: np.ndarray) -> float:
            """Compute cut value (negative for minimization)."""
            cut = 0.0
            for (i, j), w in zip(edges, weights):
                if assignment[i] != assignment[j]:
                    cut += w
            return -cut  # Negative because we minimize

        # Brute force for small problems
        if n_nodes <= 20:
            best_assignment, best_cost = self._brute_force_maxcut(
                n_nodes, edges, weights
            )
        else:
            # QAOA-inspired classical heuristic
            best_assignment, best_cost = self._qaoa_heuristic(
                n_nodes, adj, self.p_layers
            )

        # Compute approximation ratio
        optimal_cost = self._compute_optimal_maxcut(n_nodes, edges, weights)
        approx_ratio = abs(best_cost) / abs(optimal_cost) if optimal_cost != 0 else 1.0

        execution_time = time.time() - start_time

        return ClassicalQAOAResult(
            solution=best_assignment,
            cost=abs(best_cost),
            optimal_params=np.zeros(2 * self.p_layers),  # Placeholder
            approximation_ratio=approx_ratio,
            n_iterations=self.n_samples,
            success=True,
            execution_time=execution_time,
        )

    def solve_ising(
        self,
        J: np.ndarray,
        h: np.ndarray | None = None,
    ) -> ClassicalQAOAResult:
        """Solve Ising model optimization.

        H = -Σ J_ij s_i s_j - Σ h_i s_i

        Args:
            J: Coupling matrix
            h: External field (optional)

        Returns:
            ClassicalQAOAResult with best spin configuration
        """
        start_time = time.time()
        n = J.shape[0]

        if h is None:
            h = np.zeros(n)

        # Cost function
        def ising_energy(spins: np.ndarray) -> float:
            """Compute Ising energy."""
            coupling = -np.sum(J * np.outer(spins, spins)) / 2
            field = -np.sum(h * spins)
            return coupling + field

        # Try multiple random starts
        best_energy = float("inf")
        best_spins = None

        for _ in range(self.n_samples):
            # Random spin configuration
            spins = self._rng.choice([-1, 1], size=n).astype(float)

            # Local optimization
            spins = self._local_ising_optimize(spins, J, h)

            energy = ising_energy(spins)
            if energy < best_energy:
                best_energy = energy
                best_spins = spins.copy()

        execution_time = time.time() - start_time

        return ClassicalQAOAResult(
            solution=best_spins if best_spins is not None else np.zeros(n),
            cost=best_energy,
            optimal_params=np.zeros(2 * self.p_layers),
            n_iterations=self.n_samples,
            success=True,
            execution_time=execution_time,
        )

    def _brute_force_maxcut(
        self,
        n_nodes: int,
        edges: list[tuple[int, int]],
        weights: list[float],
    ) -> tuple[np.ndarray, float]:
        """Brute force MaxCut for small instances."""
        best_cut = 0.0
        best_assignment = np.zeros(n_nodes)

        for bits in range(2**n_nodes):
            assignment = np.array([(bits >> i) & 1 for i in range(n_nodes)])

            cut = 0.0
            for (i, j), w in zip(edges, weights):
                if assignment[i] != assignment[j]:
                    cut += w

            if cut > best_cut:
                best_cut = cut
                best_assignment = assignment.copy()

        return best_assignment, -best_cut

    def _qaoa_heuristic(
        self,
        n_nodes: int,
        adj: np.ndarray,
        p: int,
    ) -> tuple[np.ndarray, float]:
        """QAOA-inspired classical heuristic.

        Simulates QAOA by:
        1. Starting from superposition-like state (random)
        2. Applying problem-Hamiltonian-like evolution (greedy improvement)
        3. Applying mixer-like evolution (local perturbation)
        """
        best_cut = 0.0
        best_assignment = np.zeros(n_nodes)

        for _ in range(self.n_samples):
            # Initial random assignment
            assignment = self._rng.choice([0, 1], size=n_nodes)

            # QAOA-like iterations
            for layer in range(p):
                # Problem Hamiltonian layer: greedy improvement
                for i in range(n_nodes):
                    gain = 0.0
                    for j in range(n_nodes):
                        if adj[i, j] > 0:
                            if assignment[i] != assignment[j]:
                                gain -= adj[i, j]
                            else:
                                gain += adj[i, j]
                    if gain > 0:
                        assignment[i] = 1 - assignment[i]

                # Mixer layer: random bit flip with decreasing probability
                flip_prob = 0.5 / (layer + 1)
                for i in range(n_nodes):
                    if self._rng.random() < flip_prob:
                        assignment[i] = 1 - assignment[i]

            # Compute cut value
            cut = 0.0
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if adj[i, j] > 0 and assignment[i] != assignment[j]:
                        cut += adj[i, j]

            if cut > best_cut:
                best_cut = cut
                best_assignment = assignment.copy()

        return best_assignment, -best_cut

    def _compute_optimal_maxcut(
        self,
        n_nodes: int,
        edges: list[tuple[int, int]],
        weights: list[float],
    ) -> float:
        """Compute optimal MaxCut value (brute force for small instances)."""
        if n_nodes > 20:
            # For large instances, return estimate
            return sum(weights)

        _, cost = self._brute_force_maxcut(n_nodes, edges, weights)
        return abs(cost)

    def _local_ising_optimize(
        self,
        spins: np.ndarray,
        J: np.ndarray,
        h: np.ndarray,
        max_steps: int = 100,
    ) -> np.ndarray:
        """Local optimization for Ising model."""
        n = len(spins)

        for _ in range(max_steps):
            improved = False

            for i in range(n):
                # Compute energy change if spin i is flipped
                delta_E = 2 * spins[i] * (np.sum(J[i, :] * spins) + h[i])

                if delta_E < 0:
                    spins[i] *= -1
                    improved = True

            if not improved:
                break

        return spins
