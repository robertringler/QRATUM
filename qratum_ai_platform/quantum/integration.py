"""Integration adapters connecting quantum modules to main platform.

This module provides adapters that connect the quantum computing modules
(VQE, QAOA) to the QRATUM unified platform, handling configuration,
execution, and result formatting.
"""Quantum backend integration adapters.

Adapters for integrating quasim.quantum modules with platform layer.
"""

from __future__ import annotations

from typing import Any

from quasim.quantum.core import QuantumConfig, create_backend
from quasim.quantum.qaoa_optimization import QAOA
from quasim.quantum.vqe_molecule import MolecularVQE


class QuantumModuleAdapter:
    """Adapter layer between platform and quantum modules.

    This adapter provides a unified interface for executing quantum algorithms
    (VQE, QAOA) from the platform layer, handling configuration mapping and
    result formatting.

    Example:
        >>> platform_config = {
        >>>     'quantum_backend': 'simulator',
        >>>     'shots': 1024,
        >>>     'seed': 42
        >>> }
        >>> adapter = QuantumModuleAdapter(platform_config)
        >>> result = adapter.execute_vqe('H2', 0.735)
    """

    def __init__(self, platform_config: dict[str, Any]):
        """Initialize quantum adapter with platform configuration.

        Args:
            platform_config: Platform configuration dict with keys:
                - quantum_backend: Backend type ('simulator', 'ibmq')
                - shots: Number of measurement shots
                - seed: Random seed for reproducibility
                - ibmq_token: IBM Quantum token (optional)
        """
        # Map platform config to quantum config
        self.quantum_config = QuantumConfig(
            backend_type=platform_config.get("quantum_backend", "simulator"),
            shots=platform_config.get("shots", 1024),
            seed=platform_config.get("seed", 42),
            ibmq_token=platform_config.get("ibmq_token"),
            device_name=platform_config.get("device_name"),
        )

        # Create backend instance
        self.backend = create_backend(self.quantum_config)

        # Initialize quantum algorithm instances
        self.vqe = MolecularVQE(self.quantum_config)
        self.qaoa = None  # Lazy initialization with p_layers

    def execute_vqe(
        self,
        molecule: str,
        bond_length: float,
        basis: str = "sto3g",
        use_classical_reference: bool = True,
        optimizer: str = "COBYLA",
        max_iterations: int = 100,
    ) -> dict[str, Any]:
        """Execute VQE with platform context.

        Args:
            molecule: Molecule name ("H2", "LiH", "BeH2")
            bond_length: Bond length in Angstroms
            basis: Basis set
            use_classical_reference: Compare to classical calculation
            optimizer: Classical optimizer
            max_iterations: Max optimization iterations

        Returns:
            Dict with VQE results (energy, params, convergence, etc.)
        """
        result = self.vqe.compute_molecule_energy(
            molecule=molecule,
            bond_length=bond_length,
            basis=basis,
            use_classical_reference=use_classical_reference,
            optimizer=optimizer,
            max_iterations=max_iterations,
        )

        # Convert dataclass to dict for platform consumption
        return {
            "energy": result.energy,
            "optimal_params": result.optimal_params.tolist(),
            "n_iterations": result.n_iterations,
            "n_evaluations": result.n_evaluations,
            "success": result.success,
            "classical_energy": result.classical_energy,
            "error_vs_classical": result.error_vs_classical,
            "std_dev": result.std_dev,
            "convergence": result.convergence,
            "execution_id": result.execution_id,
        }

    def execute_qaoa(
        self,
        problem_type: str,
        problem_data: dict[str, Any],
        p_layers: int = 3,
        optimizer: str = "COBYLA",
        max_iterations: int = 100,
        classical_reference: bool = True,
    ) -> dict[str, Any]:
        """Execute QAOA with platform context.

        Args:
            problem_type: Problem type ('maxcut', 'ising')
            problem_data: Problem-specific data
            p_layers: Number of QAOA layers
            optimizer: Classical optimizer
            max_iterations: Max iterations
            classical_reference: Compute classical solution

        Returns:
            Dict with QAOA results (solution, energy, approximation_ratio, etc.)
        """
        # Lazy initialize QAOA with p_layers
        if self.qaoa is None or self.qaoa.p_layers != p_layers:
            self.qaoa = QAOA(self.quantum_config, p_layers=p_layers)

        # Execute based on problem type
        if problem_type == "maxcut":
            result = self.qaoa.solve_maxcut(
                edges=problem_data["edges"],
                optimizer=optimizer,
                max_iterations=max_iterations,
                classical_reference=classical_reference,
            )
        elif problem_type == "ising":
            result = self.qaoa.solve_ising(
                coupling_matrix=problem_data["coupling_matrix"],
                external_field=problem_data.get("external_field"),
                optimizer=optimizer,
                max_iterations=max_iterations,
            )
        else:
            raise ValueError(f"Unsupported QAOA problem type: {problem_type}")

        # Convert dataclass to dict
        return {
            "solution": result.solution,
            "energy": result.energy,
            "optimal_params": result.optimal_params.tolist(),
            "approximation_ratio": result.approximation_ratio,
            "n_iterations": result.n_iterations,
            "success": result.success,
            "classical_optimal": result.classical_optimal,
            "prob_distribution": result.prob_distribution,
            "execution_id": result.execution_id,
        }

    def get_backend_info(self) -> dict[str, Any]:
        """Get information about the current quantum backend.

        Returns:
            dict with backend information
        """
        return {
            "backend_name": self.backend.backend_name,
            "backend_type": self.quantum_config.backend_type,
            "shots": self.quantum_config.shots,
            "seed": self.quantum_config.seed,
            "is_simulator": self.quantum_config.is_simulator,
        }


__all__ = ["QuantumModuleAdapter"]
from typing import Any, Dict, Optional

__all__ = ["QuantumBackendAdapter"]


class QuantumBackendAdapter:
    """Adapter for quantum backend operations.

    Wraps quasim.quantum modules without modifying them.
    """

    def __init__(
        self,
        backend_type: str = "simulator",
        seed: Optional[int] = None,
        shots: int = 1024,
    ):
        """Initialize quantum backend adapter.

        Args:
            backend_type: Backend type ('simulator', 'ibmq', 'cuquantum')
            seed: Random seed for reproducibility
            shots: Number of shots
        """
        self.backend_type = backend_type
        self.seed = seed
        self.shots = shots

    def get_config(self) -> Dict[str, Any]:
        """Get backend configuration.

        Returns:
            Configuration dictionary
        """
        return {
            "backend_type": self.backend_type,
            "seed": self.seed,
            "shots": self.shots,
        }

    def validate_backend(self) -> bool:
        """Validate backend is available.

        Returns:
            True if backend is available
        """
        # Stub implementation - in production, check actual backend availability
        return True
