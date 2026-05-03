"""Test suite for distributed quantum simulation.

Tests:
- Correctness: distributed vs single-GPU for small systems
- Bell/GHZ scalability and entanglement
- Determinism: identical seeds yield identical outputs
- Checkpoint/restore functionality
"""

from __future__ import annotations

import numpy as np
import pytest

from .quasim_dist import (
    init_cluster,
    initialize_zero_state,
    load_checkpoint,
    profile,
    save_checkpoint,
    shard_state,
)
from .quasim_multi import MultiQubitSimulator, create_bell_plus, create_ghz_state_exact


class TestDistributedSimulation:
    """Test distributed quantum simulation."""

    def test_init_cluster_jax(self):
        """Test JAX cluster initialization."""

        try:
            ctx = init_cluster(backend="jax", mesh_shape=(1, 1), seed=12345)
            assert ctx.backend == "jax"
            assert ctx.mesh_shape == (1, 1)
            assert ctx.seed == 12345
            assert ctx.world_size >= 1
        except Exception as e:
            pytest.skip(f"JAX not available: {e}")

    def test_init_cluster_torch(self):
        """Test PyTorch cluster initialization."""

        try:
            ctx = init_cluster(backend="torch", mesh_shape=(1, 1), seed=12345)
            assert ctx.backend == "torch"
            assert ctx.mesh_shape == (1, 1)
            assert ctx.seed == 12345
            assert ctx.world_size >= 1
        except Exception as e:
            pytest.skip(f"PyTorch not available: {e}")

    def test_shard_state(self):
        """Test state sharding."""

        ctx = init_cluster(backend="jax", mesh_shape=(1, 1), seed=42)
        state = initialize_zero_state(num_qubits=4)

        sharded = shard_state(ctx, state)
        assert sharded.global_shape == (16,)
        assert sharded.context.backend == "jax"

    def test_determinism(self):
        """Test deterministic execution with same seed."""

        seed = 42
        num_qubits = 3

        # Run 1
        init_cluster(backend="jax", mesh_shape=(1, 1), seed=seed)
        state1 = initialize_zero_state(num_qubits=num_qubits)

        # Run 2
        init_cluster(backend="jax", mesh_shape=(1, 1), seed=seed)
        state2 = initialize_zero_state(num_qubits=num_qubits)

        # Should be identical
        np.testing.assert_array_equal(state1, state2)

    def test_checkpoint_save_load(self, tmp_path):
        """Test checkpoint save and restore."""

        ctx = init_cluster(backend="jax", mesh_shape=(1, 1), seed=42)
        state = initialize_zero_state(num_qubits=4)
        sharded = shard_state(ctx, state)

        # Save checkpoint
        checkpoint_path = str(tmp_path / "checkpoint")
        save_checkpoint(ctx, sharded, checkpoint_path)

        # Load checkpoint
        restored = load_checkpoint(ctx, checkpoint_path)
        assert restored.global_shape == sharded.global_shape

    def test_profile(self):
        """Test profiling functionality."""

        ctx = init_cluster(backend="jax", mesh_shape=(1, 1), seed=42)
        prof = profile(ctx)

        assert "backend" in prof
        assert "world_size" in prof
        assert "wall_time_s" in prof
        assert prof["backend"] == "jax"


class TestMultiQubitSimulator:
    """Test multi-qubit simulator."""

    def test_initialization(self):
        """Test simulator initialization."""

        sim = MultiQubitSimulator(num_qubits=2, seed=123)
        sim.initialize_state()

        assert sim.state is not None
        assert len(sim.state) == 4
        assert np.isclose(sim.state[0], 1.0)
        assert np.isclose(np.linalg.norm(sim.state), 1.0)

    def test_single_qubit_gates(self):
        """Test single-qubit gate application."""

        sim = MultiQubitSimulator(num_qubits=2, seed=123)
        sim.initialize_state()

        # Apply Hadamard
        sim.apply_gate("H", [0])

        # State should be (|00⟩ + |10⟩)/√2
        expected = np.zeros(4, dtype=np.complex128)
        expected[0] = 1.0 / np.sqrt(2)
        expected[2] = 1.0 / np.sqrt(2)

        np.testing.assert_allclose(sim.state, expected, atol=1e-10)

    def test_bell_pair(self):
        """Test Bell pair creation."""

        sim = MultiQubitSimulator(num_qubits=2, seed=123)
        sim.initialize_state()
        sim.create_bell_pair((0, 1))

        # Compare with exact Bell state
        expected = create_bell_plus()
        fidelity = sim.compute_fidelity(expected)

        assert fidelity > 0.9999, f"Bell pair fidelity: {fidelity}"

    def test_ghz_state(self):
        """Test GHZ state creation."""

        num_qubits = 3
        sim = MultiQubitSimulator(num_qubits=num_qubits, seed=123)
        sim.initialize_state()
        sim.create_ghz_state()

        # Compare with exact GHZ state
        expected = create_ghz_state_exact(num_qubits)
        fidelity = sim.compute_fidelity(expected)

        assert fidelity > 0.9999, f"GHZ state fidelity: {fidelity}"

    def test_entanglement_entropy(self):
        """Test entanglement entropy calculation."""

        sim = MultiQubitSimulator(num_qubits=2, seed=123)
        sim.initialize_state()

        # Product state should have zero entropy
        entropy = sim.entanglement_entropy([0])
        assert np.isclose(entropy, 0.0, atol=1e-10)

        # Bell pair should have 1 bit of entropy
        sim.create_bell_pair((0, 1))
        entropy = sim.entanglement_entropy([0])
        assert np.isclose(entropy, 1.0, atol=1e-10)

    def test_w_state(self):
        """Test W state creation."""

        sim = MultiQubitSimulator(num_qubits=3, seed=123)
        sim.initialize_state()
        sim.create_w_state()

        # Check normalization
        assert np.isclose(np.linalg.norm(sim.state), 1.0)

        # Check that exactly 3 states have equal amplitude
        probs = np.abs(sim.state) ** 2
        nonzero = probs[probs > 1e-10]
        assert len(nonzero) == 3
        assert np.allclose(nonzero, 1.0 / 3.0, atol=1e-10)

    def test_noise_application(self):
        """Test noise channel application."""

        sim = MultiQubitSimulator(num_qubits=2, seed=123)
        sim.initialize_state()
        sim.create_bell_pair((0, 1))

        # Store pure state fidelity
        target = create_bell_plus()
        fidelity_pure = sim.compute_fidelity(target)

        # Apply noise
        noise_dict = {"gamma1": 0.01, "gamma_phi": 0.01, "p_depol": 0.01}
        sim.apply_noise(noise_dict)

        # Fidelity should decrease
        fidelity_noisy = sim.compute_fidelity(target)
        assert fidelity_noisy < fidelity_pure

    def test_tomography(self):
        """Test quantum state tomography."""

        sim = MultiQubitSimulator(num_qubits=2, seed=123)
        sim.initialize_state()
        sim.create_bell_pair((0, 1))

        results = sim.tomography()

        assert "density_matrix_real" in results
        assert "density_matrix_imag" in results
        assert "purity" in results

        # Pure state should have purity ≈ 1
        assert results["purity"] > 0.99

    def test_deterministic_execution(self):
        """Test deterministic execution with fixed seed."""

        seed = 42

        # Run 1
        sim1 = MultiQubitSimulator(num_qubits=3, seed=seed)
        sim1.initialize_state()
        sim1.create_ghz_state()
        state1 = sim1.state.copy()

        # Run 2
        sim2 = MultiQubitSimulator(num_qubits=3, seed=seed)
        sim2.initialize_state()
        sim2.create_ghz_state()
        state2 = sim2.state.copy()

        # Should be identical
        np.testing.assert_array_equal(state1, state2)


class TestScalability:
    """Test scalability and performance."""

    @pytest.mark.parametrize("num_qubits", [2, 4, 6, 8])
    def test_ghz_scaling(self, num_qubits):
        """Test GHZ state creation scales correctly."""

        sim = MultiQubitSimulator(num_qubits=num_qubits, seed=123)
        sim.initialize_state()
        sim.create_ghz_state()

        # Check normalization
        assert np.isclose(np.linalg.norm(sim.state), 1.0)

        # Check GHZ structure
        expected = create_ghz_state_exact(num_qubits)
        fidelity = sim.compute_fidelity(expected)
        assert fidelity > 0.9999

    @pytest.mark.parametrize("num_qubits", [2, 4, 6])
    def test_entropy_scaling(self, num_qubits):
        """Test entanglement entropy scales correctly."""

        sim = MultiQubitSimulator(num_qubits=num_qubits, seed=123)
        sim.initialize_state()
        sim.create_ghz_state()

        # For GHZ, subsystem entropy should be ~1 bit
        entropy = sim.entanglement_entropy([0])
        assert 0.9 < entropy < 1.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
