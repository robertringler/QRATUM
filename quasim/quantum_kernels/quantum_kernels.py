"""Quantum Kernel Methods for Machine Learning.

This module provides quantum-inspired kernel methods for ML enhancement,
suitable for bounded verticals within QRATUM's trust architecture.

Key features:
- Quantum-inspired feature maps
- Kernel computation for SVM-like algorithms
- Integration with classical ML pipelines
- Verification hooks for kernel validity

Applications:
- Classification with quantum-like feature spaces
- Regression with entanglement-inspired kernels
- Anomaly detection in high-dimensional spaces
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal

import numpy as np


class KernelType(Enum):
    """Types of quantum-inspired kernels."""

    RBF_QUANTUM = "rbf_quantum"  # Quantum-enhanced RBF
    FIDELITY = "fidelity"  # Quantum state fidelity
    PROJECTED = "projected"  # Projected quantum kernel
    IQP = "iqp"  # Instantaneous Quantum Polynomial


@dataclass
class QuantumKernelConfig:
    """Configuration for quantum kernel computation.

    Attributes:
        kernel_type: Type of quantum kernel
        feature_dimension: Dimension of quantum feature space
        n_layers: Number of encoding layers
        gamma: Kernel width parameter (for RBF-like kernels)
        seed: Random seed for reproducibility
        use_entanglement: Whether to use entanglement-like correlations
        regularization: Regularization parameter
    """

    kernel_type: Literal["rbf_quantum", "fidelity", "projected", "iqp"] = "rbf_quantum"
    feature_dimension: int = 2
    n_layers: int = 2
    gamma: float = 1.0
    seed: int | None = 42
    use_entanglement: bool = True
    regularization: float = 1e-8


class QuantumKernel:
    """Quantum-inspired kernel for machine learning.

    This class implements quantum-inspired kernels that can be used
    with classical kernel methods (SVM, GP, etc.) while capturing
    some quantum-like feature map properties.

    The kernels are designed to:
    - Be efficiently computable classically
    - Capture quantum-like correlations
    - Integrate with sklearn and similar frameworks
    - Support verification of kernel properties

    Example:
        >>> kernel = QuantumKernel(QuantumKernelConfig(feature_dimension=4))
        >>> X = np.random.randn(100, 4)
        >>> K = kernel.compute_kernel_matrix(X)
        >>> # Use K with SVM or other kernel method
    """

    def __init__(self, config: QuantumKernelConfig):
        """Initialize quantum kernel.

        Args:
            config: Kernel configuration
        """
        self.config = config
        self._rng = np.random.default_rng(config.seed)

        # Initialize feature map parameters
        self._init_feature_map()

    def _init_feature_map(self) -> None:
        """Initialize feature map parameters."""
        n_features = self.config.feature_dimension
        n_layers = self.config.n_layers

        # Random rotation angles for feature map
        self._rotation_params = self._rng.uniform(
            -np.pi, np.pi, size=(n_layers, n_features)
        )

        # Entanglement pattern
        if self.config.use_entanglement:
            self._entanglement_params = self._rng.uniform(
                -np.pi / 4, np.pi / 4, size=(n_layers, n_features - 1)
            )
        else:
            self._entanglement_params = np.zeros((n_layers, max(1, n_features - 1)))

    def compute_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel value between two data points.

        Args:
            x1: First data point
            x2: Second data point

        Returns:
            Kernel value k(x1, x2)
        """
        if self.config.kernel_type == "rbf_quantum":
            return self._rbf_quantum_kernel(x1, x2)
        elif self.config.kernel_type == "fidelity":
            return self._fidelity_kernel(x1, x2)
        elif self.config.kernel_type == "projected":
            return self._projected_kernel(x1, x2)
        elif self.config.kernel_type == "iqp":
            return self._iqp_kernel(x1, x2)
        else:
            raise ValueError(f"Unknown kernel type: {self.config.kernel_type}")

    def compute_kernel_matrix(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        """Compute kernel matrix for data.

        Args:
            X: Data matrix (n_samples, n_features)
            Y: Optional second data matrix (default: X)

        Returns:
            Kernel matrix K[i,j] = k(X[i], Y[j])
        """
        if Y is None:
            Y = X

        n_x = X.shape[0]
        n_y = Y.shape[0]

        K = np.zeros((n_x, n_y))

        for i in range(n_x):
            for j in range(n_y):
                K[i, j] = self.compute_kernel(X[i], Y[j])

        # Add regularization for numerical stability
        if Y is X:
            K += self.config.regularization * np.eye(n_x)

        return K

    def feature_map(self, x: np.ndarray) -> np.ndarray:
        """Map data point to quantum feature space.

        Args:
            x: Data point

        Returns:
            Feature vector in quantum-inspired space
        """
        d = self.config.feature_dimension
        n_layers = self.config.n_layers

        # Initialize state (like |0⟩^n)
        state_dim = 2**d
        state = np.zeros(state_dim, dtype=complex)
        state[0] = 1.0

        # Apply parameterized quantum-like evolution
        for layer in range(n_layers):
            # Data encoding (like RZ rotations)
            state = self._apply_encoding_layer(state, x, layer)

            # Entanglement (like CNOT layer)
            if self.config.use_entanglement:
                state = self._apply_entanglement_layer(state, layer)

        return state

    def _rbf_quantum_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute quantum-enhanced RBF kernel.

        This kernel combines RBF with quantum-inspired correlations.
        """
        gamma = self.config.gamma

        # Standard RBF component
        diff = x1 - x2
        rbf_component = np.exp(-gamma * np.dot(diff, diff))

        # Quantum-inspired component: interference pattern
        sum_vec = x1 + x2
        interference = np.cos(gamma * np.linalg.norm(sum_vec))

        # Combined kernel
        return float(0.7 * rbf_component + 0.3 * (1 + interference) / 2)

    def _fidelity_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute quantum state fidelity kernel.

        k(x1, x2) = |⟨φ(x1)|φ(x2)⟩|²
        """
        state1 = self.feature_map(x1)
        state2 = self.feature_map(x2)

        overlap = np.abs(np.vdot(state1, state2)) ** 2

        return float(overlap)

    def _projected_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute projected quantum kernel.

        Projects quantum states and computes classical kernel.
        """
        state1 = self.feature_map(x1)
        state2 = self.feature_map(x2)

        # Project to probability distribution
        prob1 = np.abs(state1) ** 2
        prob2 = np.abs(state2) ** 2

        # Hellinger kernel on probability distributions
        hellinger = np.sum(np.sqrt(prob1 * prob2))

        return float(hellinger)

    def _iqp_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute IQP-inspired kernel.

        Based on Instantaneous Quantum Polynomial circuits.
        """
        d = len(x1)

        # Linear terms
        linear = np.sum(x1 * x2)

        # Quadratic interaction terms
        quadratic = 0.0
        for i in range(d):
            for j in range(i + 1, d):
                quadratic += x1[i] * x1[j] * x2[i] * x2[j]

        # Combine with cosine for periodic structure
        combined = linear + 0.5 * quadratic
        kernel_value = (1 + np.cos(self.config.gamma * combined)) / 2

        return float(kernel_value)

    def _apply_encoding_layer(
        self,
        state: np.ndarray,
        x: np.ndarray,
        layer: int,
    ) -> np.ndarray:
        """Apply data encoding layer to state."""
        d = min(len(x), self.config.feature_dimension)
        n_qubits = int(np.log2(len(state)))

        # Apply RZ-like rotations encoding data
        new_state = state.copy()

        for i in range(min(n_qubits, d)):
            angle = x[i] * self.config.gamma + self._rotation_params[layer, i]

            # RZ rotation on qubit i
            for basis_idx in range(len(state)):
                if (basis_idx >> i) & 1:
                    new_state[basis_idx] *= np.exp(1j * angle)
                else:
                    new_state[basis_idx] *= np.exp(-1j * angle)

        return new_state

    def _apply_entanglement_layer(
        self,
        state: np.ndarray,
        layer: int,
    ) -> np.ndarray:
        """Apply entanglement layer to state."""
        n_qubits = int(np.log2(len(state)))
        new_state = state.copy()

        # Apply controlled-Z like operations
        for i in range(n_qubits - 1):
            angle = self._entanglement_params[layer, i]

            for basis_idx in range(len(state)):
                # Apply phase if both qubits i and i+1 are |1⟩
                if ((basis_idx >> i) & 1) and ((basis_idx >> (i + 1)) & 1):
                    new_state[basis_idx] *= np.exp(1j * angle)

        return new_state

    def verify_kernel_validity(self, X: np.ndarray) -> dict[str, Any]:
        """Verify kernel matrix satisfies required properties.

        Checks:
        - Symmetry
        - Positive semi-definiteness
        - Numerical stability

        Args:
            X: Data matrix

        Returns:
            Dictionary with verification results
        """
        K = self.compute_kernel_matrix(X)

        # Check symmetry
        is_symmetric = np.allclose(K, K.T)

        # Check positive semi-definiteness
        eigenvalues = np.linalg.eigvalsh(K)
        min_eigenvalue = float(np.min(eigenvalues))
        is_psd = min_eigenvalue >= -1e-10

        # Check diagonal values (should be 1 for normalized kernels)
        diag = np.diag(K)
        max_diag = float(np.max(diag))
        min_diag = float(np.min(diag))

        return {
            "is_symmetric": is_symmetric,
            "is_positive_semidefinite": is_psd,
            "min_eigenvalue": min_eigenvalue,
            "max_diagonal": max_diag,
            "min_diagonal": min_diag,
            "condition_number": float(np.max(eigenvalues) / max(abs(min_eigenvalue), 1e-15)),
            "valid": is_symmetric and is_psd,
        }

    def get_sklearn_kernel(self) -> Callable:
        """Get kernel function compatible with sklearn.

        Returns:
            Callable kernel function
        """

        def sklearn_kernel(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
            return self.compute_kernel_matrix(X, Y)

        return sklearn_kernel


class QuantumFeatureEncoder:
    """Encoder for quantum-inspired feature extraction.

    This encoder transforms classical data into quantum-inspired
    feature representations suitable for ML pipelines.

    Example:
        >>> encoder = QuantumFeatureEncoder(output_dim=16)
        >>> X_transformed = encoder.fit_transform(X)
    """

    def __init__(
        self,
        output_dim: int = 16,
        n_layers: int = 2,
        seed: int | None = 42,
    ):
        """Initialize feature encoder.

        Args:
            output_dim: Output feature dimension
            n_layers: Number of encoding layers
            seed: Random seed
        """
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._fitted = False
        self._params: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "QuantumFeatureEncoder":
        """Fit encoder to data.

        Args:
            X: Training data

        Returns:
            self
        """
        input_dim = X.shape[1]

        # Initialize encoding parameters
        self._params = self._rng.uniform(
            -np.pi,
            np.pi,
            size=(self.n_layers, input_dim, self.output_dim),
        )

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to quantum feature space.

        Args:
            X: Input data

        Returns:
            Transformed features
        """
        if not self._fitted or self._params is None:
            raise ValueError("Encoder not fitted. Call fit() first.")

        n_samples = X.shape[0]
        output = np.zeros((n_samples, self.output_dim))

        for i in range(n_samples):
            output[i] = self._encode_sample(X[i])

        return output

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data.

        Args:
            X: Input data

        Returns:
            Transformed features
        """
        return self.fit(X).transform(X)

    def _encode_sample(self, x: np.ndarray) -> np.ndarray:
        """Encode single sample."""
        if self._params is None:
            raise ValueError("Encoder not fitted")

        features = np.zeros(self.output_dim)

        for layer in range(self.n_layers):
            # Linear combination with non-linearity
            linear = np.dot(x, self._params[layer])

            # Quantum-inspired non-linearity (cosine)
            features += np.cos(linear + layer * np.pi / 4)

        # Normalize
        norm = np.linalg.norm(features)
        if norm > 1e-10:
            features /= norm

        return features
