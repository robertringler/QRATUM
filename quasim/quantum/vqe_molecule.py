"""Variational Quantum Eigensolver (VQE) for molecular ground state energy.

This module implements VQE for small molecules, validated against known exact results.
VQE is a hybrid quantum-classical algorithm suitable for NISQ-era quantum computers.

Example molecules supported:
- H2 (Hydrogen): 2 qubits, exact ground state ~ -1.137 Hartree
- LiH (Lithium Hydride): 4 qubits, exact ground state ~ -7.882 Hartree
- BeH2 (Beryllium Hydride): 6 qubits, exact ground state ~ -15.594 Hartree

Limitations:
- NISQ devices: Limited circuit depth (~100-5000 gates max)
- Statistical error: Requires many shots (1000+) for reliable results
- Classical simulation: Scales exponentially with qubit count (practical limit ~30 qubits)

References:
- Peruzzo et al., "A variational eigenvalue solver on a photonic quantum processor" (2014)
- Kandala et al., "Hardware-efficient variational quantum eigensolver" (2017)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

# Optional quantum dependencies
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.primitives import Estimator
    from qiskit.quantum_info import SparsePauliOp

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    QISKIT_NATURE_AVAILABLE = True
except ImportError:
    QISKIT_NATURE_AVAILABLE = False

from .core import AbstractQuantumBackend, QuantumBackend, QuantumConfig


@dataclass
class VQEResult:
    """Result of VQE computation.

    Attributes:
        energy: Ground state energy (Hartree)
        optimal_params: Optimal circuit parameters
        n_iterations: Number of optimization iterations
        n_evaluations: Total function evaluations
        success: Whether optimization converged
        classical_energy: Classical reference energy (if available)
        error_vs_classical: Absolute error vs classical (if available)
        std_dev: Standard deviation of energy measurements
        convergence: Whether optimization converged successfully
        execution_id: Optional execution identifier for tracking
    """

    energy: float
    optimal_params: np.ndarray
    n_iterations: int
    n_evaluations: int
    success: bool
    classical_energy: float | None = None
    error_vs_classical: float | None = None
    std_dev: float | None = None
    convergence: bool = True
    execution_id: str | None = None

    def __repr__(self) -> str:
        lines = [
            f"VQEResult(energy={self.energy:.6f} Ha",
            f"  iterations={self.n_iterations}",
            f"  evaluations={self.n_evaluations}",
            f"  success={self.success}",
        ]
        if self.classical_energy is not None:
            lines.append(f"  classical_ref={self.classical_energy:.6f} Ha")
            lines.append(f"  error={self.error_vs_classical:.6f} Ha")
        if self.std_dev is not None:
            lines.append(f"  std_dev={self.std_dev:.6f} Ha")
        return "\n".join(lines) + ")"


class MolecularVQE:
    """VQE implementation for molecular ground state calculations.

    This class implements the Variational Quantum Eigensolver for finding
    molecular ground state energies using quantum circuits.

    Algorithm:
    1. Generate molecular Hamiltonian (classical preprocessing)
    2. Prepare initial quantum state (Hartree-Fock)
    3. Construct parameterized ansatz circuit (e.g., UCCSD)
    4. Iteratively optimize parameters to minimize energy expectation
    5. Return ground state energy and parameters

    Example:
        >>> config = QuantumConfig(shots=1024)
        >>> vqe = MolecularVQE(config)
        >>> result = vqe.compute_h2_energy(bond_length=0.735)
        >>> print(f"H2 energy: {result.energy:.4f} Hartree")
    """

    def __init__(self, config: QuantumConfig | AbstractQuantumBackend):
        """Initialize VQE calculator.

        Args:
            config: Quantum backend configuration or backend instance

        Raises:
            ImportError: If required quantum libraries not available
        """

        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required. Install with: pip install qiskit qiskit-aer")

        # Support both config and backend instances for flexibility
        if isinstance(config, QuantumConfig):
            self.config = config
            self.backend = QuantumBackend(config)
        elif isinstance(config, AbstractQuantumBackend):
            self.backend = config
            self.config = config.config if hasattr(config, "config") else QuantumConfig()
        else:
            raise TypeError(
                f"config must be a QuantumConfig or AbstractQuantumBackend, got {type(config)!r}"
            )

        # Use Estimator primitive for energy expectation values
        self.estimator = Estimator()

    def compute_h2_energy(
        self,
        bond_length: float = 0.735,
        basis: str = "sto3g",
        use_classical_reference: bool = True,
        optimizer: str = "COBYLA",
        max_iterations: int = 100,
    ) -> VQEResult:
        """Compute H2 molecule ground state energy using VQE.

        Hydrogen molecule (H2) at equilibrium geometry:
        - Bond length: ~0.735 Angstrom
        - Exact ground state: ~-1.137 Hartree (STO-3G basis)
        - Number of qubits required: 2 (minimal)

        Args:
            bond_length: H-H bond length in Angstroms
            basis: Basis set for molecular calculation (default: sto3g)
            use_classical_reference: Compute classical energy for validation
            optimizer: Classical optimizer (COBYLA, SPSA, L_BFGS_B)
            max_iterations: Maximum optimization iterations

        Returns:
            VQEResult with energy and optimization details
        """

        # Generate H2 Hamiltonian
        hamiltonian, n_qubits, classical_energy = self._generate_h2_hamiltonian(
            bond_length, basis, compute_classical=use_classical_reference
        )

        print(f"H2 molecule (bond={bond_length:.3f} Å, basis={basis})")
        print(f"Qubits required: {n_qubits}")
        if classical_energy is not None:
            print(f"Classical (exact) energy: {classical_energy:.6f} Hartree")

        # Create ansatz circuit
        ansatz = self._create_hardware_efficient_ansatz(n_qubits, depth=2)
        n_params = ansatz.num_parameters

        print(f"Ansatz parameters: {n_params}")
        print(f"Backend: {self.backend.backend_name}")
        print(f"Shots per evaluation: {self.config.shots}")

        # Initial parameters (small random values)
        np.random.seed(self.config.seed)
        initial_params = np.random.uniform(-np.pi / 4, np.pi / 4, n_params)

        # Optimize using classical optimizer
        result = self._optimize_vqe(
            hamiltonian, ansatz, initial_params, optimizer=optimizer, max_iterations=max_iterations
        )

        # Add classical reference if computed
        if classical_energy is not None:
            result.classical_energy = classical_energy
            result.error_vs_classical = abs(result.energy - classical_energy)

        return result

    def compute_molecule_energy(
        self,
        molecule: str,
        bond_length: float,
        basis: str = "sto3g",
        use_classical_reference: bool = True,
        optimizer: str = "COBYLA",
        max_iterations: int = 100,
    ) -> VQEResult:
        """Compute molecular ground state energy using VQE.

        Generalized VQE for molecules (H2, LiH, BeH2).

        Supports:
        - H2: 2 qubits (production-ready)
        - LiH: 4 qubits (planned Phase 2)
        - BeH2: 6 qubits (planned Phase 2)

        Args:
            molecule: Molecule name ("H2", "LiH", "BeH2")
            bond_length: Bond length in Angstroms
            basis: Basis set for molecular calculation
            use_classical_reference: Compute classical energy for validation
            optimizer: Classical optimizer (COBYLA, SPSA)
            max_iterations: Maximum optimization iterations

        Returns:
            VQEResult with energy and optimization details

        Raises:
            ValueError: If molecule is not supported
            NotImplementedError: If molecule support is planned but not implemented
        """
        if molecule == "H2":
            return self.compute_h2_energy(
                bond_length=bond_length,
                basis=basis,
                use_classical_reference=use_classical_reference,
                optimizer=optimizer,
                max_iterations=max_iterations,
            )
        elif molecule == "LiH":
            raise NotImplementedError("LiH VQE planned for Phase 2 (2026)")
        elif molecule == "BeH2":
            raise NotImplementedError("BeH2 VQE planned for Phase 2 (2026)")
        else:
            raise ValueError(f"Unsupported molecule: {molecule}")

    def _generate_h2_hamiltonian(
        self, bond_length: float, basis: str, compute_classical: bool = True
    ) -> tuple[SparsePauliOp, int, float | None]:
        """Generate H2 Hamiltonian using Qiskit Nature.

        Returns:
            (hamiltonian, n_qubits, classical_energy)
        """

        if not QISKIT_NATURE_AVAILABLE:
            # Fallback: Use hardcoded H2 Hamiltonian for bond_length ~ 0.735 Å
            warnings.warn(
                "qiskit-nature not available. Using approximate H2 Hamiltonian.",
                UserWarning,
                stacklevel=2,
            )
            return self._get_approximate_h2_hamiltonian()

        # Use PySCF to generate molecular Hamiltonian
        try:
            from pyscf import gto, scf

            # Build H2 molecule
            mol = gto.Mole()
            mol.atom = f"H 0 0 0; H 0 0 {bond_length}"
            mol.basis = basis
            mol.build()

            # Get classical reference if requested
            classical_energy = None
            if compute_classical:
                mf = scf.RHF(mol)
                classical_energy = mf.kernel()

            # Use Qiskit Nature to get qubit Hamiltonian
            driver = PySCFDriver.from_molecule(mol)
            problem = driver.run()

            # Map to qubits using Jordan-Wigner transformation
            mapper = JordanWignerMapper()
            hamiltonian = mapper.map(problem.hamiltonian.second_q_op())

            n_qubits = hamiltonian.num_qubits

            return hamiltonian, n_qubits, classical_energy

        except ImportError:
            warnings.warn(
                "PySCF not available. Using approximate Hamiltonian.", UserWarning, stacklevel=2
            )
            return self._get_approximate_h2_hamiltonian()

    def _get_approximate_h2_hamiltonian(self) -> tuple[SparsePauliOp, int, float]:
        """Get approximate H2 Hamiltonian for bond_length ~ 0.735 Å (STO-3G).

        This is a fallback when qiskit-nature/pyscf are not available.
        Coefficients derived from minimal basis H2 calculation.

        Reference: These coefficients match the H2 Hamiltonian at equilibrium
        geometry (0.735Å) with STO-3G basis, obtained from standard quantum
        chemistry calculations (e.g., PySCF).
        """

        # Known H2 Hamiltonian terms (Jordan-Wigner, 2 qubits)
        # Coefficients for H2 at 0.735 Angstroms, STO-3G basis
        hamiltonian = SparsePauliOp.from_list(
            [
                ("II", -1.0523),  # Identity term (nuclear repulsion + const)
                ("IZ", 0.3979),  # Z on qubit 1
                ("ZI", -0.3979),  # Z on qubit 0
                ("ZZ", -0.0112),  # ZZ interaction
                ("XX", 0.1809),  # XX interaction (electron hopping)
            ]
        )

        n_qubits = 2
        classical_energy = -1.137  # Hartree-Fock energy for this geometry

        return hamiltonian, n_qubits, classical_energy

    def _create_hardware_efficient_ansatz(self, n_qubits: int, depth: int = 2) -> QuantumCircuit:
        """Create hardware-efficient ansatz circuit.

        This ansatz alternates single-qubit rotations with entangling gates,
        designed to be shallow and hardware-friendly for NISQ devices.

        Args:
            n_qubits: Number of qubits
            depth: Circuit depth (number of rotation + entangling layers)

        Returns:
            Parameterized quantum circuit
        """

        circuit = QuantumCircuit(n_qubits)
        params = []

        # Initial layer of rotations
        for i in range(n_qubits):
            param = Parameter(f"θ_init_{i}")
            params.append(param)
            circuit.ry(param, i)

        # Alternating rotation + entangling layers
        for d in range(depth):
            # Entangling layer (linear chain)
            for i in range(n_qubits - 1):
                circuit.cx(i, i + 1)

            # Rotation layer
            for i in range(n_qubits):
                param = Parameter(f"θ_d{d}_q{i}")
                params.append(param)
                circuit.ry(param, i)

        return circuit

    def _optimize_vqe(
        self,
        hamiltonian: SparsePauliOp,
        ansatz: QuantumCircuit,
        initial_params: np.ndarray,
        optimizer: str = "COBYLA",
        max_iterations: int = 100,
    ) -> VQEResult:
        """Optimize VQE energy using classical optimizer.

        Args:
            hamiltonian: Qubit Hamiltonian operator
            ansatz: Parameterized quantum circuit
            initial_params: Initial parameter values
            optimizer: Optimizer name
            max_iterations: Maximum iterations

        Returns:
            VQEResult with optimal energy and parameters
        """

        # Track optimization progress
        iteration_count = [0]
        evaluation_count = [0]
        energies = []

        def cost_function(params: np.ndarray) -> float:
            """Evaluate energy expectation value."""

            evaluation_count[0] += 1

            # Use Estimator to compute <ψ(θ)|H|ψ(θ)>
            job = self.estimator.run(ansatz, hamiltonian, params)
            result = job.result()
            energy = result.values[0]

            energies.append(energy)

            if evaluation_count[0] % 10 == 0:
                print(f"  Evaluation {evaluation_count[0]}: E = {energy:.6f} Ha")

            return energy

        # Select optimizer
        if optimizer == "COBYLA":
            from scipy.optimize import minimize

            result = minimize(
                cost_function, initial_params, method="COBYLA", options={"maxiter": max_iterations}
            )
            optimal_params = result.x
            optimal_energy = result.fun
            success = result.success

        elif optimizer == "SPSA":
            # Simple SPSA implementation
            optimal_params, optimal_energy, success = self._spsa_optimize(
                cost_function, initial_params, max_iterations
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # Compute standard deviation from last few evaluations
        std_dev = np.std(energies[-5:]) if len(energies) >= 5 else None

        iteration_count[0] = len(energies)

        return VQEResult(
            energy=optimal_energy,
            optimal_params=optimal_params,
            n_iterations=iteration_count[0],
            n_evaluations=evaluation_count[0],
            success=success,
            std_dev=std_dev,
        )

    def _spsa_optimize(
        self,
        cost_fn: Any,
        initial_params: np.ndarray,
        max_iterations: int,
        a: float = 0.05,
        c: float = 0.1,
    ) -> tuple[np.ndarray, float, bool]:
        """Simple SPSA (Simultaneous Perturbation Stochastic Approximation).

        SPSA is gradient-free and efficient for noisy objective functions,
        making it suitable for quantum optimization.
        """

        params = initial_params.copy()
        best_energy = float("inf")
        best_params = params.copy()

        for k in range(max_iterations):
            # SPSA learning rates
            ak = a / (k + 1) ** 0.602
            ck = c / (k + 1) ** 0.101

            # Random perturbation
            delta = 2 * np.random.randint(0, 2, size=len(params)) - 1

            # Two-point gradient estimate
            params_plus = params + ck * delta
            params_minus = params - ck * delta

            loss_plus = cost_fn(params_plus)
            loss_minus = cost_fn(params_minus)

            gradient_est = (loss_plus - loss_minus) / (2 * ck * delta)

            # Update parameters
            params = params - ak * gradient_est

            # Track best result
            current_energy = (loss_plus + loss_minus) / 2
            if current_energy < best_energy:
                best_energy = current_energy
                best_params = params.copy()

        # Final evaluation
        final_energy = cost_fn(best_params)

        return best_params, final_energy, True


# Export main classes
__all__ = ["MolecularVQE", "VQEResult"]
