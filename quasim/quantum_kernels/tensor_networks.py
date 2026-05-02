"""Tensor Network Methods for Quantum-Inspired Simulation.

This module implements tensor network methods that provide efficient
classical simulation of quantum systems, particularly for systems
with limited entanglement.

Supported methods:
- MPS (Matrix Product States): Efficient for 1D systems
- PEPS (Projected Entangled Pair States): For 2D systems (stub)
- MERA (Multi-scale Entanglement Renormalization Ansatz): For scale-invariant systems (planned)

Applications:
- Molecular simulation (limited bond dimension)
- Materials science (lattice models)
- Climate modeling (tensor decomposition)
- Financial modeling (high-dimensional PDEs)

References:
- Schollwöck, "The density-matrix renormalization group in the age of MPS" (2011)
- Verstraete et al., "Matrix Product States, PEPS, and beyond" (2008)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np


class TensorNetworkType(Enum):
    """Types of tensor networks."""

    MPS = "mps"
    PEPS = "peps"
    MERA = "mera"
    TTN = "ttn"  # Tree Tensor Network


@dataclass
class TensorNetworkConfig:
    """Configuration for tensor network simulations.

    Attributes:
        network_type: Type of tensor network (mps, peps, mera, ttn)
        bond_dimension: Maximum bond dimension (controls accuracy vs. cost)
        physical_dimension: Dimension of physical indices (usually 2 for qubits)
        num_sites: Number of sites/qubits
        convergence_threshold: Convergence threshold for iterative methods
        max_iterations: Maximum iterations for optimization
        seed: Random seed for reproducibility
        use_svd_cutoff: Whether to use SVD truncation with cutoff
        svd_cutoff: SVD singular value cutoff (relative)
    """

    network_type: Literal["mps", "peps", "mera", "ttn"] = "mps"
    bond_dimension: int = 64
    physical_dimension: int = 2
    num_sites: int = 10
    convergence_threshold: float = 1e-10
    max_iterations: int = 100
    seed: int | None = 42
    use_svd_cutoff: bool = True
    svd_cutoff: float = 1e-12

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.bond_dimension < 1:
            raise ValueError("bond_dimension must be at least 1")

        if self.num_sites < 2:
            raise ValueError("num_sites must be at least 2")

        if self.bond_dimension > 1000:
            warnings.warn(
                f"bond_dimension={self.bond_dimension} is large. "
                "Memory usage scales as O(D^3) per site.",
                UserWarning,
                stacklevel=2,
            )


@dataclass
class MPSState:
    """Matrix Product State representation.

    An MPS represents a quantum state as:
    |ψ⟩ = Σ A[1]_{i1} A[2]_{i2} ... A[n]_{in} |i1 i2 ... in⟩

    where A[k] are rank-3 tensors (left_bond, physical, right_bond).

    Attributes:
        tensors: List of MPS tensors
        center: Position of orthogonality center (-1 if not canonical)
        bond_dimensions: List of bond dimensions
        norm: Norm of the state
    """

    tensors: list[np.ndarray]
    center: int = -1
    bond_dimensions: list[int] = field(default_factory=list)
    norm: float = 1.0

    def __post_init__(self) -> None:
        """Compute bond dimensions if not provided."""
        if not self.bond_dimensions and self.tensors:
            self.bond_dimensions = [t.shape[0] for t in self.tensors[:-1]]
            self.bond_dimensions.append(self.tensors[-1].shape[2] if self.tensors else 1)

    @property
    def num_sites(self) -> int:
        """Number of sites in MPS."""
        return len(self.tensors)

    @property
    def max_bond_dimension(self) -> int:
        """Maximum bond dimension."""
        return max(self.bond_dimensions) if self.bond_dimensions else 1


class MPSSimulator:
    """Matrix Product State simulator for quantum systems.

    This simulator provides efficient classical simulation of quantum
    systems with limited entanglement, using MPS representation.

    Key operations:
    - Ground state finding via DMRG
    - Time evolution via TEBD
    - Expectation value computation
    - Entanglement entropy calculation

    Example:
        >>> config = TensorNetworkConfig(num_sites=20, bond_dimension=64)
        >>> simulator = MPSSimulator(config)
        >>> # Create Heisenberg Hamiltonian
        >>> hamiltonian = simulator.create_heisenberg_hamiltonian(J=1.0)
        >>> # Find ground state
        >>> energy, state = simulator.find_ground_state(hamiltonian)
        >>> print(f"Ground state energy: {energy:.6f}")
    """

    def __init__(self, config: TensorNetworkConfig):
        """Initialize MPS simulator.

        Args:
            config: Tensor network configuration
        """
        if config.network_type != "mps":
            warnings.warn(
                f"MPSSimulator received network_type={config.network_type}, using MPS.",
                UserWarning,
                stacklevel=2,
            )

        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def create_product_state(self, local_states: list[np.ndarray] | None = None) -> MPSState:
        """Create product state MPS.

        Args:
            local_states: List of local state vectors (default: all |0⟩)

        Returns:
            MPSState in product state form
        """
        n = self.config.num_sites
        d = self.config.physical_dimension

        if local_states is None:
            # Default: all |0⟩ states
            local_states = [np.array([1.0, 0.0]) for _ in range(n)]

        tensors = []
        for i, state in enumerate(local_states):
            # Rank-3 tensor with bond dimension 1
            tensor = state.reshape(1, d, 1)
            tensors.append(tensor)

        return MPSState(tensors=tensors, center=0)

    def create_random_mps(self, bond_dimension: int | None = None) -> MPSState:
        """Create random MPS state.

        Args:
            bond_dimension: Bond dimension (default from config)

        Returns:
            Random MPSState
        """
        n = self.config.num_sites
        d = self.config.physical_dimension
        D = bond_dimension or self.config.bond_dimension

        tensors = []

        # Compute bond dimensions ensuring consistency
        # Bond dims grow from left, then shrink from right
        bond_dims = []
        for i in range(n - 1):
            left_max = d ** (i + 1)  # Maximum from left
            right_max = d ** (n - i - 1)  # Maximum from right
            bond_dims.append(min(D, left_max, right_max))

        # First site: (1, d, D_1)
        tensor = self._rng.standard_normal((1, d, bond_dims[0] if bond_dims else 1))
        tensors.append(tensor)

        # Middle sites: (D_i, d, D_{i+1})
        for i in range(1, n - 1):
            left_dim = bond_dims[i - 1]
            right_dim = bond_dims[i]
            tensor = self._rng.standard_normal((left_dim, d, right_dim))
            tensors.append(tensor)

        # Last site: (D_{n-1}, d, 1)
        if n > 1:
            left_dim = bond_dims[-1] if bond_dims else 1
            tensor = self._rng.standard_normal((left_dim, d, 1))
            tensors.append(tensor)

        state = MPSState(tensors=tensors)

        # Normalize
        return self.normalize(state)

    def normalize(self, state: MPSState) -> MPSState:
        """Normalize MPS state.

        Args:
            state: MPS state to normalize

        Returns:
            Normalized MPS state
        """
        norm = self.compute_norm(state)
        if abs(norm) < 1e-15:
            return state

        # Scale first tensor
        new_tensors = [state.tensors[0] / np.sqrt(norm)] + list(state.tensors[1:])

        return MPSState(tensors=new_tensors, center=state.center, norm=1.0)

    def compute_norm(self, state: MPSState) -> float:
        """Compute norm of MPS state.

        Args:
            state: MPS state

        Returns:
            Norm squared ⟨ψ|ψ⟩
        """
        # Contract from left using transfer matrices
        # For MPS tensor A with shape (D_left, d, D_right):
        # Transfer matrix T = Σ_s A[s] ⊗ A*[s] has shape (D_left² , D_right²)
        # We contract: T_1 @ T_2 @ ... @ T_n

        left = np.array([[1.0]])  # Initial boundary (1x1 matrix)

        for tensor in state.tensors:
            # tensor shape: (D_left, d, D_right)
            D_left, d, D_right = tensor.shape

            # Contract: left @ (A ⊗ A*)
            # left has shape (D_left_prev, D_left_prev) where D_left_prev matches D_left
            # Result should have shape (D_right, D_right)

            # Compute transfer matrix contraction
            # T_{αβ,γδ} = Σ_s A_{αsγ} A*_{βsδ}
            # New left = old_left_{αβ} T_{αβ,γδ} = Σ_{αβs} left_{αβ} A_{αsγ} A*_{βsδ}

            new_left = np.einsum("ab,asc,bsd->cd", left, tensor, np.conj(tensor))
            left = new_left

        return float(np.real(left[0, 0]))

    def compute_expectation_value(
        self,
        state: MPSState,
        operator: np.ndarray,
        site: int,
    ) -> float:
        """Compute expectation value of local operator.

        Args:
            state: MPS state
            operator: Local operator matrix
            site: Site index

        Returns:
            ⟨ψ|O|ψ⟩
        """
        # Contract from left up to site
        left = np.array([[1.0]])
        for i, tensor in enumerate(state.tensors):
            if i < site:
                left = np.einsum("ab,asc,bsd->cd", left, tensor, np.conj(tensor))
            elif i == site:
                # Insert operator
                left = np.einsum(
                    "ab,asc,st,btd->cd",
                    left,
                    tensor,
                    operator,
                    np.conj(tensor),
                )
            else:
                left = np.einsum("ab,asc,bsd->cd", left, tensor, np.conj(tensor))

        return float(np.real(left[0, 0]))

    def compute_entanglement_entropy(self, state: MPSState, cut_position: int) -> float:
        """Compute entanglement entropy at bipartition.

        Args:
            state: MPS state
            cut_position: Position of bipartition cut

        Returns:
            von Neumann entropy S = -Σ λ² log(λ²)
        """
        # Contract from left to cut position
        left = np.eye(1)
        for i in range(cut_position + 1):
            tensor = state.tensors[i]
            left = np.einsum("ab,acd,bce->de", left, tensor, np.conj(tensor))

        # Eigenvalues give squared Schmidt coefficients
        eigenvalues = np.linalg.eigvalsh(left)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]  # Remove numerical zeros

        # von Neumann entropy
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))

        return float(entropy)

    def apply_two_site_gate(
        self,
        state: MPSState,
        gate: np.ndarray,
        site: int,
        truncate: bool = True,
    ) -> MPSState:
        """Apply two-site gate to MPS.

        Args:
            state: MPS state
            gate: Two-site gate (d²×d² matrix)
            site: Left site index
            truncate: Whether to truncate to max bond dimension

        Returns:
            Updated MPS state
        """
        d = self.config.physical_dimension
        D = self.config.bond_dimension

        # Get the two tensors
        A = state.tensors[site]  # (D_L, d, D_M)
        B = state.tensors[site + 1]  # (D_M, d, D_R)

        # Contract tensors
        # theta_{L,i,j,R} = A_{L,i,M} B_{M,j,R}
        theta = np.einsum("aim,mjr->aijr", A, B)

        # Reshape for gate application
        D_L, D_R = theta.shape[0], theta.shape[3]
        theta = theta.reshape(D_L, d * d, D_R)

        # Apply gate
        gate_reshaped = gate.reshape(d * d, d * d)
        theta = np.einsum("aib,ij->ajb", theta, gate_reshaped)

        # Reshape back
        theta = theta.reshape(D_L * d, d * D_R)

        # SVD to split back into two tensors
        U, S, Vh = np.linalg.svd(theta, full_matrices=False)

        # Truncate
        if truncate:
            D_new = min(len(S), D)
            U = U[:, :D_new]
            S = S[:D_new]
            Vh = Vh[:D_new, :]

        # Absorb singular values into U
        U = U @ np.diag(S)

        # Reshape back to tensors
        new_A = U.reshape(D_L, d, -1)
        new_B = Vh.reshape(-1, d, D_R)

        # Update state
        new_tensors = list(state.tensors)
        new_tensors[site] = new_A
        new_tensors[site + 1] = new_B

        return MPSState(tensors=new_tensors, center=site)

    def create_heisenberg_hamiltonian(
        self,
        J: float = 1.0,
        h: float = 0.0,
    ) -> list[tuple[np.ndarray, int, int]]:
        """Create Heisenberg model Hamiltonian terms.

        H = J Σ (S_i · S_{i+1}) + h Σ S_i^z

        Args:
            J: Exchange coupling
            h: External field strength

        Returns:
            List of (operator, site1, site2) tuples
        """
        n = self.config.num_sites

        # Pauli matrices
        sx = np.array([[0, 1], [1, 0]], dtype=complex) / 2
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
        sz = np.array([[1, 0], [0, -1]], dtype=complex) / 2
        identity = np.eye(2, dtype=complex)

        terms = []

        # Exchange terms
        for i in range(n - 1):
            # XX + YY + ZZ interaction
            hxx = np.kron(sx, sx)
            hyy = np.kron(sy, sy)
            hzz = np.kron(sz, sz)
            h_bond = J * (hxx + hyy + hzz)
            terms.append((h_bond, i, i + 1))

        # Field terms (as single-site terms)
        if h != 0:
            for i in range(n):
                terms.append((h * sz, i, i))

        return terms

    def find_ground_state_simple(
        self,
        hamiltonian_terms: list[tuple[np.ndarray, int, int]],
        max_sweeps: int = 10,
    ) -> tuple[float, MPSState]:
        """Find ground state using simplified variational method.

        This is a simplified ground state finder suitable for demonstration.
        For production use, implement full DMRG.

        Args:
            hamiltonian_terms: Hamiltonian as list of (operator, site1, site2)
            max_sweeps: Maximum optimization sweeps

        Returns:
            (ground_state_energy, ground_state_mps)
        """
        # Start with random state
        state = self.create_random_mps()

        best_energy = float("inf")
        best_state = state

        for sweep in range(max_sweeps):
            # Compute energy
            energy = self._compute_energy(state, hamiltonian_terms)

            if energy < best_energy:
                best_energy = energy
                best_state = state

            # Simple update: apply imaginary time evolution
            state = self._imaginary_time_step(state, hamiltonian_terms, dt=0.1)

        return best_energy, best_state

    def _compute_energy(
        self,
        state: MPSState,
        hamiltonian_terms: list[tuple[np.ndarray, int, int]],
    ) -> float:
        """Compute energy expectation value."""
        energy = 0.0

        for op, site1, site2 in hamiltonian_terms:
            if site1 == site2:
                # Single-site term
                energy += self.compute_expectation_value(state, op, site1)
            else:
                # Two-site term - simplified evaluation
                # This is approximate; full implementation needs proper contraction
                d = self.config.physical_dimension
                local_energy = self._compute_two_site_expectation(state, op, site1)
                energy += np.real(local_energy)

        return float(energy)

    def _compute_two_site_expectation(
        self,
        state: MPSState,
        operator: np.ndarray,
        site: int,
    ) -> float:
        """Compute two-site expectation value (simplified)."""
        d = self.config.physical_dimension

        # Contract state around the two sites
        A = state.tensors[site]
        B = state.tensors[site + 1]

        # Contract into two-site tensor
        theta = np.einsum("aim,mjr->aijr", A, B)

        # Reshape
        D_L, D_R = theta.shape[0], theta.shape[3]
        theta_flat = theta.reshape(D_L, d * d, D_R)

        # Apply operator and contract
        op_reshaped = operator.reshape(d * d, d * d)
        theta_op = np.einsum("aib,ij->ajb", theta_flat, op_reshaped)

        # Contract with conjugate
        result = np.einsum("aib,aib->", theta_flat, np.conj(theta_op))

        return float(np.real(result))

    def _imaginary_time_step(
        self,
        state: MPSState,
        hamiltonian_terms: list[tuple[np.ndarray, int, int]],
        dt: float,
    ) -> MPSState:
        """Apply imaginary time evolution step."""
        d = self.config.physical_dimension

        for op, site1, site2 in hamiltonian_terms:
            if site1 != site2:
                # Two-site gate: exp(-dt * H)
                # Simplified: use first-order approximation
                gate = np.eye(d * d) - dt * op.reshape(d * d, d * d)
                state = self.apply_two_site_gate(state, gate.reshape(d, d, d, d), site1)

        return self.normalize(state)


class PEPSSimulator:
    """Projected Entangled Pair States simulator (stub).

    PEPS extends MPS to 2D systems, enabling simulation of
    2D quantum systems with area-law entanglement.

    Note: Full PEPS contraction is computationally expensive (#P-hard).
    This implementation provides basic operations; for production,
    consider approximate contraction methods.

    Status: Stub implementation - full version planned for Phase 2.
    """

    def __init__(self, config: TensorNetworkConfig):
        """Initialize PEPS simulator.

        Args:
            config: Tensor network configuration
        """
        if config.network_type != "peps":
            warnings.warn(
                f"PEPSSimulator received network_type={config.network_type}, using PEPS.",
                UserWarning,
                stacklevel=2,
            )

        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def create_product_state_2d(
        self,
        lattice_size: tuple[int, int],
        local_states: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Create 2D product state PEPS (stub).

        Args:
            lattice_size: (rows, cols) of 2D lattice
            local_states: Local state array (optional)

        Returns:
            PEPS state representation
        """
        rows, cols = lattice_size
        d = self.config.physical_dimension

        # Stub: return metadata only
        return {
            "type": "peps",
            "lattice_size": lattice_size,
            "bond_dimension": 1,
            "physical_dimension": d,
            "status": "stub_implementation",
            "note": "Full PEPS implementation planned for Phase 2",
        }

    def approximate_contraction(
        self,
        state: dict[str, Any],
        boundary_mps_dim: int = 32,
    ) -> float:
        """Approximate PEPS contraction using boundary MPS (stub).

        Args:
            state: PEPS state
            boundary_mps_dim: Bond dimension for boundary MPS

        Returns:
            Approximate norm
        """
        warnings.warn(
            "PEPS contraction is a stub. Returning 1.0",
            UserWarning,
            stacklevel=2,
        )
        return 1.0
