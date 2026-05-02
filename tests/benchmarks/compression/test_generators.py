"""Tests for quantum state generators."""

import numpy as np

from benchmarks.compression.generators.quantum_states import (
    generate_ghz_state, generate_random_state)


class TestRandomStateGenerator:
    """Tests for random state generation."""

    def test_correct_dimension(self):
        """Random state should have correct dimension."""

        state = generate_random_state(4, seed=42)
        assert state.shape == (16,)

    def test_normalized(self):
        """Random state should be normalized."""

        state = generate_random_state(5, seed=42)
        norm = np.linalg.norm(state)
        assert abs(norm - 1.0) < 1e-10

    def test_deterministic_with_seed(self):
        """Same seed should produce same state."""

        state1 = generate_random_state(3, seed=42)
        state2 = generate_random_state(3, seed=42)
        assert np.allclose(state1, state2)


class TestGHZStateGenerator:
    """Tests for GHZ state generation."""

    def test_correct_structure(self):
        """GHZ state should be (|000⟩ + |111⟩)/√2."""

        state = generate_ghz_state(3)
        assert state.shape == (8,)

        # Check |000⟩ and |111⟩ amplitudes
        expected = 1.0 / np.sqrt(2)
        assert abs(state[0] - expected) < 1e-10
        assert abs(state[7] - expected) < 1e-10

    def test_normalized(self):
        """GHZ state should be normalized."""

        state = generate_ghz_state(5)
        norm = np.linalg.norm(state)
        assert abs(norm - 1.0) < 1e-10
