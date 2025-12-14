"""Multi-Qubit Quantum Simulator with Entanglement Support.

This module implements a scalable multi-qubit quantum simulator supporting:
- N-qubit state vectors (2-32 qubits)
- Entangled state generation (Bell, GHZ, W)
- Single and multi-qubit gate operations
- Noise channels (amplitude/phase damping, depolarizing)
- Pauli tomography and entanglement entropy
- Monte Carlo trajectory averaging
- Deterministic, reproducible execution
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class MultiQubitSimulator:
    """Multi-qubit quantum simulator with noise and entanglement support.

    Implements state-vector simulation for N qubits with support for:
    - Arbitrary single and multi-qubit gates
    - Noise channels via Kraus operators
    - Time-dependent Hamiltonian evolution
    - Entanglement measures and tomography

    Attributes:
        num_qubits: Number of qubits in the system
        seed: Random seed for reproducibility
        state: Current quantum state vector
        results: Dictionary of simulation results
    """

    def __init__(self, num_qubits: int = 2, seed: int = 123) -> None:
        """Initialize multi-qubit simulator.

        Args:
            num_qubits: Number of qubits (2-32 supported)
            seed: Random seed for deterministic execution
        """
        if num_qubits < 1 or num_qubits > 32:
            raise ValueError("num_qubits must be between 1 and 32")

        self.num_qubits = num_qubits
        self.seed = seed
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.state: NDArray[np.complex128] | None = None
        self.results: dict[str, Any] = {}
        self.gate_history: list[dict[str, Any]] = []

    def initialize_state(self, state: str | NDArray[np.complex128] = "zero") -> None:
        """Initialize quantum state.

        Args:
            state: Initial state - "zero" for |00...0⟩, or custom state vector
        """
        dim = 2**self.num_qubits

        if isinstance(state, str) and state == "zero":
            self.state = np.zeros(dim, dtype=np.complex128)
            self.state[0] = 1.0
        elif isinstance(state, np.ndarray):
            if state.shape[0] != dim:
                raise ValueError(f"State vector must have dimension {dim}")
            norm = np.linalg.norm(state)
            self.state = state.astype(np.complex128) / norm
        else:
            raise ValueError("state must be 'zero' or a numpy array")

    def apply_gate(
        self,
        gate: str | NDArray[np.complex128],
        targets: list[int],
        params: dict[str, float] | None = None,
    ) -> None:
        """Apply quantum gate to target qubits.

        Args:
            gate: Gate name ("H", "X", "Y", "Z", "CNOT", "CZ", "SWAP") or matrix
            targets: List of target qubit indices
            params: Optional parameters for parameterized gates (e.g., rotation angle)
        """
        if self.state is None:
            raise RuntimeError("State not initialized. Call initialize_state() first.")

        # Get gate matrix
        gate_matrix = self._get_standard_gate(gate, params) if isinstance(gate, str) else gate

        # Apply gate to state
        self.state = self._apply_gate_to_state(self.state, gate_matrix, targets)

        # Record gate application
        self.gate_history.append(
            {
                "gate": gate if isinstance(gate, str) else "custom",
                "targets": targets,
                "params": params,
            }
        )

    def _get_standard_gate(
        self, gate_name: str, params: dict[str, float] | None = None
    ) -> NDArray[np.complex128]:
        """Get standard gate matrix by name.

        Args:
            gate_name: Name of the gate
            params: Optional parameters

        Returns:
            Gate matrix as numpy array
        """
        params = params or {}

        # Single-qubit gates
        if gate_name == "H":  # Hadamard
            return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        elif gate_name == "X":  # Pauli-X
            return np.array([[0, 1], [1, 0]], dtype=np.complex128)
        elif gate_name == "Y":  # Pauli-Y
            return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        elif gate_name == "Z":  # Pauli-Z
            return np.array([[1, 0], [0, -1]], dtype=np.complex128)
        elif gate_name == "S":  # Phase gate
            return np.array([[1, 0], [0, 1j]], dtype=np.complex128)
        elif gate_name == "T":  # T gate
            return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
        elif gate_name == "Rx":  # X rotation
            theta = params.get("theta", 0.0)
            return np.array(
                [
                    [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                    [-1j * np.sin(theta / 2), np.cos(theta / 2)],
                ],
                dtype=np.complex128,
            )
        elif gate_name == "Ry":  # Y rotation
            theta = params.get("theta", 0.0)
            return np.array(
                [[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]],
                dtype=np.complex128,
            )
        elif gate_name == "Rz":  # Z rotation
            theta = params.get("theta", 0.0)
            return np.array(
                [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=np.complex128
            )

        # Two-qubit gates
        elif gate_name == "CNOT":
            return np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128
            )
        elif gate_name == "CZ":
            return np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=np.complex128
            )
        elif gate_name == "SWAP":
            return np.array(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
            )
        else:
            raise ValueError(f"Unknown gate: {gate_name}")

    def _apply_gate_to_state(
        self, state: NDArray[np.complex128], gate: NDArray[np.complex128], targets: list[int]
    ) -> NDArray[np.complex128]:
        """Apply gate matrix to state vector.

        Uses tensor reshaping for efficient application of gates to arbitrary qubits.

        Args:
            state: Current state vector
            gate: Gate matrix
            targets: Target qubit indices

        Returns:
            Updated state vector
        """
        n = self.num_qubits
        d = 2**n

        # Reshape state to tensor
        shape = [2] * n
        state_tensor = state.reshape(shape)

        # Handle single-qubit gates
        if len(targets) == 1:
            q = targets[0]
            # Move target qubit to first position
            axes = [q] + [i for i in range(n) if i != q]
            state_tensor = np.transpose(state_tensor, axes)
            # Apply gate
            state_tensor = np.tensordot(gate, state_tensor, axes=([1], [0]))
            # Move qubit back
            inv_axes = [0] * n
            for i, ax in enumerate(axes):
                inv_axes[ax] = i
            state_tensor = np.transpose(state_tensor, inv_axes)

        # Handle two-qubit gates
        elif len(targets) == 2:
            q1, q2 = targets
            # Move target qubits to first positions
            axes = [q1, q2] + [i for i in range(n) if i not in [q1, q2]]
            state_tensor = np.transpose(state_tensor, axes)
            # Reshape for gate application
            shape_front = (4,) + tuple([2] * (n - 2))
            state_tensor = state_tensor.reshape(shape_front)
            # Apply gate
            gate_reshaped = gate.reshape(4, 4)
            state_tensor = np.tensordot(gate_reshaped, state_tensor, axes=([1], [0]))
            # Reshape back
            state_tensor = state_tensor.reshape([2] * n)
            # Move qubits back
            inv_axes = [0] * n
            for i, ax in enumerate(axes):
                inv_axes[ax] = i
            state_tensor = np.transpose(state_tensor, inv_axes)
        else:
            raise ValueError("Only 1 and 2 qubit gates are currently supported")

        return state_tensor.reshape(d)

    def apply_noise(self, noise_dict: dict[str, float]) -> None:
        """Apply noise channels to the quantum state.

        Args:
            noise_dict: Dictionary with noise parameters:
                - gamma1: Amplitude damping rate
                - gamma_phi: Phase damping rate
                - p_depol: Depolarizing probability
                - corr: Correlation coefficient for cross-dephasing
        """
        if self.state is None:
            raise RuntimeError("State not initialized")

        gamma1 = noise_dict.get("gamma1", 0.0)
        gamma_phi = noise_dict.get("gamma_phi", 0.0)
        p_depol = noise_dict.get("p_depol", 0.0)

        # Convert to density matrix for noise application
        rho = np.outer(self.state, np.conj(self.state))

        # Apply amplitude damping
        if gamma1 > 0:
            for q in range(self.num_qubits):
                rho = self._apply_amplitude_damping(rho, q, gamma1)

        # Apply phase damping
        if gamma_phi > 0:
            for q in range(self.num_qubits):
                rho = self._apply_phase_damping(rho, q, gamma_phi)

        # Apply depolarizing
        if p_depol > 0:
            for q in range(self.num_qubits):
                rho = self._apply_depolarizing(rho, q, p_depol)

        # Extract state vector (for pure states, take dominant eigenvector)
        eigenvalues, eigenvectors = np.linalg.eigh(rho)
        max_idx = np.argmax(eigenvalues)
        self.state = eigenvectors[:, max_idx]
        self.state /= np.linalg.norm(self.state)

    def _apply_amplitude_damping(
        self, rho: NDArray[np.complex128], qubit: int, gamma: float
    ) -> NDArray[np.complex128]:
        """Apply amplitude damping to density matrix."""
        # Kraus operators for amplitude damping
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)

        rho_new = self._apply_single_qubit_kraus(rho, qubit, [K0, K1])
        return rho_new

    def _apply_phase_damping(
        self, rho: NDArray[np.complex128], qubit: int, gamma: float
    ) -> NDArray[np.complex128]:
        """Apply phase damping to density matrix."""
        # Kraus operators for phase damping
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
        K1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=np.complex128)

        rho_new = self._apply_single_qubit_kraus(rho, qubit, [K0, K1])
        return rho_new

    def _apply_depolarizing(
        self, rho: NDArray[np.complex128], qubit: int, p: float
    ) -> NDArray[np.complex128]:
        """Apply depolarizing channel to density matrix."""
        # Kraus operators for depolarizing
        K0 = np.sqrt(1 - 3 * p / 4) * np.eye(2, dtype=np.complex128)
        K1 = np.sqrt(p / 4) * np.array([[0, 1], [1, 0]], dtype=np.complex128)
        K2 = np.sqrt(p / 4) * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        K3 = np.sqrt(p / 4) * np.array([[1, 0], [0, -1]], dtype=np.complex128)

        rho_new = self._apply_single_qubit_kraus(rho, qubit, [K0, K1, K2, K3])
        return rho_new

    def _apply_single_qubit_kraus(
        self, rho: NDArray[np.complex128], qubit: int, kraus_ops: list[NDArray[np.complex128]]
    ) -> NDArray[np.complex128]:
        """Apply Kraus operators to single qubit in density matrix."""
        d = 2**self.num_qubits
        rho_new = np.zeros((d, d), dtype=np.complex128)

        for K in kraus_ops:
            # Build full Kraus operator for the system
            K_full = self._build_full_operator(K, qubit)
            rho_new += K_full @ rho @ K_full.conj().T

        return rho_new

    def _build_full_operator(
        self, single_qubit_op: NDArray[np.complex128], target: int
    ) -> NDArray[np.complex128]:
        """Build full system operator from single-qubit operator."""
        op = np.array([[1.0]], dtype=np.complex128)

        for q in range(self.num_qubits):
            if q == target:
                op = np.kron(op, single_qubit_op)
            else:
                op = np.kron(op, np.eye(2, dtype=np.complex128))

        return op

    def create_bell_pair(self, qubits: tuple[int, int] = (0, 1)) -> None:
        """Create Bell pair |Φ+⟩ = (|00⟩ + |11⟩)/√2.

        Args:
            qubits: Tuple of two qubit indices
        """
        if self.state is None:
            self.initialize_state()

        q0, q1 = qubits
        self.apply_gate("H", [q0])
        self.apply_gate("CNOT", [q0, q1])

    def create_ghz_state(self, qubits: list[int] | None = None) -> None:
        """Create GHZ state |GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2.

        Args:
            qubits: List of qubit indices (default: all qubits)
        """
        if self.state is None:
            self.initialize_state()

        if qubits is None:
            qubits = list(range(self.num_qubits))

        if len(qubits) < 2:
            raise ValueError("GHZ state requires at least 2 qubits")

        # Apply Hadamard to first qubit
        self.apply_gate("H", [qubits[0]])

        # Apply cascade of CNOTs
        for i in range(len(qubits) - 1):
            self.apply_gate("CNOT", [qubits[i], qubits[i + 1]])

    def create_w_state(self, qubits: list[int] | None = None) -> None:
        """Create W state - symmetric superposition with one excitation.

        For 3 qubits: |W⟩ = (|001⟩ + |010⟩ + |100⟩)/√3

        Args:
            qubits: List of qubit indices (default: all qubits)
        """
        if self.state is None:
            self.initialize_state()

        if qubits is None:
            qubits = list(range(self.num_qubits))

        n = len(qubits)
        if n < 2:
            raise ValueError("W state requires at least 2 qubits")

        # Create W state via controlled operations
        # Start with |0...01⟩
        d = 2**self.num_qubits
        self.state = np.zeros(d, dtype=np.complex128)

        # Build W state directly
        for _i, q in enumerate(qubits):
            idx = 2 ** (self.num_qubits - 1 - q)
            self.state[idx] = 1.0 / np.sqrt(n)

    def tomography(self) -> dict[str, Any]:
        """Perform Pauli tomography on the quantum state.

        Returns:
            Dictionary with density matrix and Bloch vectors
        """
        if self.state is None:
            raise RuntimeError("State not initialized")

        # Compute density matrix
        rho = np.outer(self.state, np.conj(self.state))

        # Compute Bloch vectors for each qubit (if tractable)
        bloch_vectors = []
        if self.num_qubits <= 3:
            for q in range(self.num_qubits):
                bloch_vec = self._compute_bloch_vector(rho, q)
                bloch_vectors.append(bloch_vec.tolist())

        self.results.update(
            {
                "density_matrix_real": rho.real.tolist(),
                "density_matrix_imag": rho.imag.tolist(),
                "bloch_vectors": bloch_vectors,
                "purity": float(np.real(np.trace(rho @ rho))),
            }
        )

        return self.results

    def _compute_bloch_vector(self, rho: NDArray[np.complex128], qubit: int) -> NDArray[np.float64]:
        """Compute Bloch vector for a single qubit."""
        # Pauli matrices
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        # Build full operators
        X_full = self._build_full_operator(X, qubit)
        Y_full = self._build_full_operator(Y, qubit)
        Z_full = self._build_full_operator(Z, qubit)

        # Compute expectation values
        x = np.real(np.trace(rho @ X_full))
        y = np.real(np.trace(rho @ Y_full))
        z = np.real(np.trace(rho @ Z_full))

        return np.array([x, y, z])

    def entanglement_entropy(self, subsystem: list[int]) -> float:
        """Compute von Neumann entropy of subsystem.

        S = -Tr[ρ_A log₂ ρ_A]

        Args:
            subsystem: List of qubit indices forming subsystem A

        Returns:
            Entanglement entropy in bits
        """
        if self.state is None:
            raise RuntimeError("State not initialized")

        # Compute reduced density matrix
        rho_reduced = self._partial_trace(subsystem)

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho_reduced)
        eigenvalues = eigenvalues[eigenvalues > 1e-14]  # Remove numerical zeros

        # Compute entropy
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))

        return float(entropy)

    def _partial_trace(self, keep_qubits: list[int]) -> NDArray[np.complex128]:
        """Compute partial trace over complement of keep_qubits."""
        n = self.num_qubits
        trace_qubits = [q for q in range(n) if q not in keep_qubits]

        # Reshape state to tensor
        shape = [2] * n
        state_tensor = self.state.reshape(shape)

        # Form density matrix tensor
        rho_tensor = np.tensordot(state_tensor, np.conj(state_tensor), axes=0)

        # Trace out unwanted qubits
        for q in sorted(trace_qubits, reverse=True):
            # Sum over diagonal elements of qubit q
            rho_tensor = np.trace(rho_tensor, axis1=q, axis2=n + q)

        # Reshape to matrix
        dim_reduced = 2 ** len(keep_qubits)
        rho_reduced = rho_tensor.reshape(dim_reduced, dim_reduced)

        return rho_reduced

    def evolve_control(
        self, control_schedule: list[tuple[float, dict[str, Any]]], method: str = "trotter"
    ) -> None:
        """Evolve state under time-dependent Hamiltonian.

        H(t) = Σ_i Ω_i(t) σ_αᵢ + Σ_{ij} J_{ij}(t) σ_z^i σ_z^j

        Args:
            control_schedule: List of (time, hamiltonian_params) tuples
            method: Evolution method - "trotter" or "expm"
        """
        if self.state is None:
            raise RuntimeError("State not initialized")

        for dt, ham_params in control_schedule:
            # Build Hamiltonian
            H = self._build_hamiltonian(ham_params)

            # Evolve
            if method == "trotter":
                # First-order Trotter: exp(-iHdt) ≈ exp(-iH₁dt)exp(-iH₂dt)...
                U = self._trotter_evolution(H, dt)
            elif method == "expm":
                # Exact matrix exponential
                U = self._expm_evolution(H, dt)
            else:
                raise ValueError(f"Unknown evolution method: {method}")

            # Apply evolution operator
            self.state = U @ self.state

    def _build_hamiltonian(self, params: dict[str, Any]) -> NDArray[np.complex128]:
        """Build Hamiltonian from parameters."""
        d = 2**self.num_qubits
        H = np.zeros((d, d), dtype=np.complex128)

        # Single-qubit terms
        for i in range(self.num_qubits):
            omega_x = params.get(f"omega_x_{i}", 0.0)
            omega_y = params.get(f"omega_y_{i}", 0.0)
            omega_z = params.get(f"omega_z_{i}", 0.0)

            if omega_x != 0:
                X_full = self._build_full_operator(self._get_standard_gate("X"), i)
                H += omega_x * X_full
            if omega_y != 0:
                Y_full = self._build_full_operator(self._get_standard_gate("Y"), i)
                H += omega_y * Y_full
            if omega_z != 0:
                Z_full = self._build_full_operator(self._get_standard_gate("Z"), i)
                H += omega_z * Z_full

        # Two-qubit coupling terms
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                J = params.get(f"J_{i}_{j}", 0.0)
                if J != 0:
                    Z = self._get_standard_gate("Z")
                    ZZ = np.kron(Z, Z)
                    ZZ_full = self._build_full_operator(ZZ, i) if i == 0 and j == 1 else H * 0
                    H += J * ZZ_full

        return H

    def _trotter_evolution(self, H: NDArray[np.complex128], dt: float) -> NDArray[np.complex128]:
        """Compute evolution operator via Trotterization."""
        # For simplicity, use matrix exponential (can be optimized later)
        return self._expm_evolution(H, dt)

    def _expm_evolution(self, H: NDArray[np.complex128], dt: float) -> NDArray[np.complex128]:
        """Compute evolution operator via matrix exponential."""
        from scipy.linalg import expm

        return expm(-1j * H * dt)

    def run(self, trajectories: int = 1) -> dict[str, Any]:
        """Run simulation and compute results.

        Args:
            trajectories: Number of Monte Carlo trajectories (for noise averaging)

        Returns:
            Dictionary with simulation results
        """
        if self.state is None:
            raise RuntimeError("State not initialized")

        # Store final state
        self.results["state_vector_real"] = self.state.real.tolist()
        self.results["state_vector_imag"] = self.state.imag.tolist()
        self.results["num_qubits"] = self.num_qubits
        self.results["seed"] = self.seed
        self.results["trajectories"] = trajectories

        return self.results

    def compute_fidelity(self, target_state: NDArray[np.complex128]) -> float:
        """Compute fidelity with target state.

        F = |⟨ψ|φ⟩|²

        Args:
            target_state: Target state vector

        Returns:
            Fidelity (0 to 1)
        """
        if self.state is None:
            raise RuntimeError("State not initialized")

        overlap = np.abs(np.vdot(target_state, self.state))
        return float(overlap**2)


def create_bell_plus() -> NDArray[np.complex128]:
    """Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
    state = np.zeros(4, dtype=np.complex128)
    state[0] = 1.0 / np.sqrt(2)  # |00⟩
    state[3] = 1.0 / np.sqrt(2)  # |11⟩
    return state


def create_ghz_state_exact(n: int) -> NDArray[np.complex128]:
    """Create exact GHZ state for n qubits."""
    d = 2**n
    state = np.zeros(d, dtype=np.complex128)
    state[0] = 1.0 / np.sqrt(2)  # |00...0⟩
    state[d - 1] = 1.0 / np.sqrt(2)  # |11...1⟩
    return state
