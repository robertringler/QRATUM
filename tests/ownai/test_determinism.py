"""Tests for determinism utilities."""

import numpy as np

from quasim.ownai.determinism import (
    hash_array,
    hash_preds,
    set_seed,
    verify_determinism,
)


def test_set_seed_numpy():
    """Test that set_seed produces deterministic numpy output."""

    set_seed(42)
    result1 = np.random.randn(10)

    set_seed(42)
    result2 = np.random.randn(10)

    np.testing.assert_array_equal(result1, result2)


def test_hash_array():
    """Test array hashing."""

    arr = np.array([1, 2, 3], dtype=np.float32)
    hash1 = hash_array(arr)
    hash2 = hash_array(arr)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 produces 64 hex chars


def test_hash_preds():
    """Test prediction hashing."""

    preds = np.array([0.1, 0.9, 0.3], dtype=np.float32)
    hash1 = hash_preds(preds)
    hash2 = hash_preds(preds)

    assert hash1 == hash2
    assert len(hash1) == 64


def test_verify_determinism():
    """Test determinism verification."""

    def deterministic_func():
        return np.random.randn(10)

    # Should be deterministic with seed
    assert verify_determinism(deterministic_func, seed=42, n_runs=3)


def test_hash_preds_different_arrays():
    """Test that different arrays produce different hashes."""

    preds1 = np.array([0.1, 0.9], dtype=np.float32)
    preds2 = np.array([0.2, 0.8], dtype=np.float32)

    hash1 = hash_preds(preds1)
    hash2 = hash_preds(preds2)

    assert hash1 != hash2
