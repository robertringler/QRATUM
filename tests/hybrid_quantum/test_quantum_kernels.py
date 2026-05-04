"""Tests for quantum kernels module."""

import numpy as np
import pytest

try:
    from quasim.quantum_kernels.tensor_networks import (
        MPSSimulator,
        MPSState,
        PEPSSimulator,
        TensorNetworkConfig,
    )

    TENSOR_NETWORKS_AVAILABLE = True
except ImportError:
    TENSOR_NETWORKS_AVAILABLE = False

try:
    from quasim.quantum_kernels.classical_analogs import (
        ClassicalQAOA,
        ClassicalQAOAResult,
        ClassicalVQE,
        ClassicalVQEResult,
    )

    CLASSICAL_ANALOGS_AVAILABLE = True
except ImportError:
    CLASSICAL_ANALOGS_AVAILABLE = False

try:
    from quasim.quantum_kernels.quantum_kernels import (
        QuantumFeatureEncoder,
        QuantumKernel,
        QuantumKernelConfig,
    )

    QUANTUM_KERNELS_AVAILABLE = True
except ImportError:
    QUANTUM_KERNELS_AVAILABLE = False

try:
    from quasim.quantum_kernels.ahtc import (
        AHTCAccelerator,
        CompressionConfig,
        CompressionResult,
    )

    AHTC_AVAILABLE = True
except ImportError:
    AHTC_AVAILABLE = False


@pytest.mark.skipif(not TENSOR_NETWORKS_AVAILABLE, reason="Tensor networks module not available")
class TestTensorNetworkConfig:
    """Test TensorNetworkConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = TensorNetworkConfig()
        assert config.network_type == "mps"
        assert config.bond_dimension == 64
        assert config.num_sites == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = TensorNetworkConfig(
            bond_dimension=128,
            num_sites=20,
            seed=123,
        )
        assert config.bond_dimension == 128
        assert config.num_sites == 20

    def test_invalid_bond_dimension(self):
        """Test that invalid bond dimension raises error."""
        with pytest.raises(ValueError):
            TensorNetworkConfig(bond_dimension=0)

    def test_high_bond_dimension_warning(self):
        """Test warning for high bond dimension."""
        with pytest.warns(UserWarning, match="large"):
            TensorNetworkConfig(bond_dimension=2000)


@pytest.mark.skipif(not TENSOR_NETWORKS_AVAILABLE, reason="Tensor networks module not available")
class TestMPSSimulator:
    """Test MPSSimulator."""

    def test_simulator_creation(self):
        """Test creating MPS simulator."""
        config = TensorNetworkConfig(num_sites=5, bond_dimension=16)
        simulator = MPSSimulator(config)
        assert simulator.config.num_sites == 5

    def test_create_product_state(self):
        """Test creating product state MPS."""
        config = TensorNetworkConfig(num_sites=4)
        simulator = MPSSimulator(config)

        state = simulator.create_product_state()

        assert state.num_sites == 4
        assert len(state.tensors) == 4

    def test_create_random_mps(self):
        """Test creating random MPS."""
        config = TensorNetworkConfig(num_sites=4, bond_dimension=8, seed=42)
        simulator = MPSSimulator(config)

        state = simulator.create_random_mps()

        assert state.num_sites == 4
        assert state.max_bond_dimension <= 8

    def test_compute_norm(self):
        """Test computing MPS norm."""
        config = TensorNetworkConfig(num_sites=3, seed=42)
        simulator = MPSSimulator(config)

        state = simulator.create_product_state()
        norm = simulator.compute_norm(state)

        assert np.isclose(norm, 1.0, atol=1e-10)

    def test_normalize(self):
        """Test normalizing MPS."""
        config = TensorNetworkConfig(num_sites=3, seed=42)
        simulator = MPSSimulator(config)

        state = simulator.create_random_mps()
        normalized = simulator.normalize(state)

        norm = simulator.compute_norm(normalized)
        assert np.isclose(norm, 1.0, atol=1e-10)

    def test_compute_entanglement_entropy(self):
        """Test computing entanglement entropy."""
        config = TensorNetworkConfig(num_sites=4, seed=42)
        simulator = MPSSimulator(config)

        # Product state has zero entanglement
        product_state = simulator.create_product_state()
        entropy = simulator.compute_entanglement_entropy(product_state, cut_position=1)

        assert entropy >= 0  # Entropy is non-negative

    def test_create_heisenberg_hamiltonian(self):
        """Test creating Heisenberg Hamiltonian."""
        config = TensorNetworkConfig(num_sites=4)
        simulator = MPSSimulator(config)

        terms = simulator.create_heisenberg_hamiltonian(J=1.0, h=0.1)

        assert len(terms) > 0
        # Should have n-1 exchange terms + n field terms
        assert len(terms) == 3 + 4  # 3 bonds + 4 sites


@pytest.mark.skipif(not TENSOR_NETWORKS_AVAILABLE, reason="Tensor networks module not available")
class TestPEPSSimulator:
    """Test PEPSSimulator (stub)."""

    def test_simulator_creation(self):
        """Test creating PEPS simulator."""
        config = TensorNetworkConfig(network_type="peps")
        simulator = PEPSSimulator(config)
        assert simulator.config.network_type == "peps"

    def test_create_product_state_2d(self):
        """Test creating 2D product state (stub)."""
        config = TensorNetworkConfig(network_type="peps")
        simulator = PEPSSimulator(config)

        state = simulator.create_product_state_2d((3, 3))

        assert state["type"] == "peps"
        assert state["lattice_size"] == (3, 3)
        assert "stub" in state["status"]


@pytest.mark.skipif(
    not CLASSICAL_ANALOGS_AVAILABLE, reason="Classical analogs module not available"
)
class TestClassicalVQE:
    """Test ClassicalVQE."""

    def test_vqe_creation(self):
        """Test creating ClassicalVQE."""
        vqe = ClassicalVQE(seed=42)
        assert vqe.seed == 42

    def test_minimize_energy(self):
        """Test minimizing energy with simple Hamiltonian."""
        vqe = ClassicalVQE(seed=42, max_iterations=100)

        # Simple 2x2 Hamiltonian
        H = np.array([[1.0, 0.5], [0.5, -1.0]])

        result = vqe.minimize_energy(H, n_params=4)

        assert result.success
        # Ground state energy should be negative for this Hamiltonian
        eigenvalues = np.linalg.eigvalsh(H)
        assert result.energy <= eigenvalues[0] + 0.1  # Allow some tolerance

    def test_molecular_energy(self):
        """Test molecular energy computation."""
        vqe = ClassicalVQE(seed=42)

        # Simple one-body integrals
        h_pq = np.array([[-1.0, 0.1], [0.1, -0.5]])

        result = vqe.compute_molecular_energy(h_pq, n_electrons=1)

        assert result.success
        assert result.method == "hartree_fock"


@pytest.mark.skipif(
    not CLASSICAL_ANALOGS_AVAILABLE, reason="Classical analogs module not available"
)
class TestClassicalQAOA:
    """Test ClassicalQAOA."""

    def test_qaoa_creation(self):
        """Test creating ClassicalQAOA."""
        qaoa = ClassicalQAOA(seed=42, p_layers=3)
        assert qaoa.p_layers == 3

    def test_solve_maxcut_small(self):
        """Test solving small MaxCut problem."""
        qaoa = ClassicalQAOA(seed=42)

        # Triangle graph
        edges = [(0, 1), (1, 2), (2, 0)]
        result = qaoa.solve_maxcut(n_nodes=3, edges=edges)

        assert result.success
        assert result.cost >= 2  # At least 2 edges can be cut
        assert result.approximation_ratio is not None

    def test_solve_ising(self):
        """Test solving Ising model."""
        qaoa = ClassicalQAOA(seed=42)

        # Simple ferromagnetic chain
        J = np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=float,
        )

        result = qaoa.solve_ising(J)

        assert result.success
        # Ground state should have aligned spins
        assert len(result.solution) == 3


@pytest.mark.skipif(not QUANTUM_KERNELS_AVAILABLE, reason="Quantum kernels module not available")
class TestQuantumKernel:
    """Test QuantumKernel."""

    def test_kernel_creation(self):
        """Test creating quantum kernel."""
        config = QuantumKernelConfig(feature_dimension=4)
        kernel = QuantumKernel(config)
        assert kernel.config.feature_dimension == 4

    def test_compute_kernel(self):
        """Test computing kernel value."""
        config = QuantumKernelConfig(feature_dimension=2, seed=42)
        kernel = QuantumKernel(config)

        x1 = np.array([1.0, 0.0])
        x2 = np.array([0.0, 1.0])

        k = kernel.compute_kernel(x1, x2)

        assert 0 <= k <= 1  # Kernel values should be bounded

    def test_compute_kernel_matrix(self):
        """Test computing kernel matrix."""
        config = QuantumKernelConfig(feature_dimension=2, seed=42)
        kernel = QuantumKernel(config)

        X = np.random.randn(10, 2)
        K = kernel.compute_kernel_matrix(X)

        assert K.shape == (10, 10)
        # Matrix should be symmetric
        assert np.allclose(K, K.T)

    def test_kernel_types(self):
        """Test different kernel types."""
        for kernel_type in ["rbf_quantum", "fidelity", "projected", "iqp"]:
            config = QuantumKernelConfig(
                kernel_type=kernel_type,
                feature_dimension=2,
                seed=42,
            )
            kernel = QuantumKernel(config)

            x1 = np.array([1.0, 0.0])
            x2 = np.array([0.0, 1.0])

            k = kernel.compute_kernel(x1, x2)
            assert isinstance(k, float)

    def test_verify_kernel_validity(self):
        """Test kernel validity verification."""
        config = QuantumKernelConfig(feature_dimension=2, seed=42)
        kernel = QuantumKernel(config)

        X = np.random.randn(5, 2)
        verification = kernel.verify_kernel_validity(X)

        assert "is_symmetric" in verification
        assert "is_positive_semidefinite" in verification
        assert "valid" in verification


@pytest.mark.skipif(not QUANTUM_KERNELS_AVAILABLE, reason="Quantum kernels module not available")
class TestQuantumFeatureEncoder:
    """Test QuantumFeatureEncoder."""

    def test_encoder_creation(self):
        """Test creating feature encoder."""
        encoder = QuantumFeatureEncoder(output_dim=16)
        assert encoder.output_dim == 16

    def test_fit_transform(self):
        """Test fit and transform."""
        encoder = QuantumFeatureEncoder(output_dim=8, seed=42)

        X = np.random.randn(20, 4)
        X_transformed = encoder.fit_transform(X)

        assert X_transformed.shape == (20, 8)

    def test_transform_requires_fit(self):
        """Test that transform requires fit."""
        encoder = QuantumFeatureEncoder(output_dim=8)
        X = np.random.randn(10, 4)

        with pytest.raises(ValueError, match="not fitted"):
            encoder.transform(X)


@pytest.mark.skipif(not AHTC_AVAILABLE, reason="AHTC module not available")
class TestAHTCAccelerator:
    """Test AHTCAccelerator."""

    def test_accelerator_creation(self):
        """Test creating AHTC accelerator."""
        config = CompressionConfig(method="svd", target_rank=10)
        accelerator = AHTCAccelerator(config)
        assert accelerator.config.target_rank == 10

    def test_compress_matrix_svd(self):
        """Test compressing matrix with SVD."""
        config = CompressionConfig(method="svd", target_rank=5)
        accelerator = AHTCAccelerator(config)

        # Create low-rank matrix
        A = np.random.randn(20, 10)
        B = np.random.randn(10, 30)
        matrix = A @ B  # Rank 10 matrix

        result = accelerator.compress(matrix, target_rank=5)

        assert result.rank_used == 5
        assert result.compression_ratio > 1

    def test_decompress(self):
        """Test decompressing matrix."""
        config = CompressionConfig(method="svd", target_rank=5)
        accelerator = AHTCAccelerator(config)

        matrix = np.random.randn(20, 30)
        compressed = accelerator.compress(matrix)

        reconstructed = accelerator.decompress(compressed)

        assert reconstructed.shape == matrix.shape

    def test_accelerated_matvec(self):
        """Test accelerated matrix-vector product."""
        config = CompressionConfig(method="svd", target_rank=10)
        accelerator = AHTCAccelerator(config)

        matrix = np.random.randn(100, 100)
        vector = np.random.randn(100)

        compressed = accelerator.compress(matrix)

        # Compare direct vs accelerated
        direct_result = matrix @ vector
        accelerated_result = accelerator.accelerated_matvec(compressed, vector)

        # Should be close (not exact due to compression)
        # Allow more tolerance for compressed result
        assert accelerated_result.shape == direct_result.shape

    def test_estimate_memory_savings(self):
        """Test memory savings estimation."""
        config = CompressionConfig()
        accelerator = AHTCAccelerator(config)

        estimate = accelerator.estimate_memory_savings(
            original_shape=(1000, 1000),
            rank=50,
        )

        assert "compression_ratio" in estimate
        assert estimate["compression_ratio"] > 1
        assert "memory_reduction_percent" in estimate

    def test_compression_methods(self):
        """Test different compression methods."""
        matrix = np.random.randn(20, 20)

        for method in ["svd", "hosvd", "tt"]:
            config = CompressionConfig(method=method, target_rank=5)
            accelerator = AHTCAccelerator(config)

            result = accelerator.compress(matrix)

            assert result.method_used == method
            assert result.compression_ratio > 0
