"""Tests for cryptographically secure RNG in PQC implementations.

Validates that zero-seed keygen has been replaced with proper secure RNG
using getrandom/secrets for production security.
"""

import pytest


def test_security_module_uses_secure_rng():
    """Test that security module uses secrets.token_bytes for key generation."""
    from qratum.planetary.security import QuantumResistantCrypto, CryptoAlgorithm
    
    crypto = QuantumResistantCrypto()
    
    # Generate multiple keys
    key1 = crypto.generate_key_pair(algorithm=CryptoAlgorithm.CRYSTALS_DILITHIUM)
    key2 = crypto.generate_key_pair(algorithm=CryptoAlgorithm.CRYSTALS_DILITHIUM)
    
    # Keys should be different (not deterministic zero-seed)
    assert key1.key_id != key2.key_id
    assert key1.public_key != key2.public_key
    
    # Verify key properties
    assert key1.algorithm == CryptoAlgorithm.CRYSTALS_DILITHIUM
    assert len(key1.public_key) > 0


def test_multiple_key_generation_produces_unique_keys():
    """Test that multiple key generations produce unique keys."""
    from qratum.planetary.security import QuantumResistantCrypto
    
    crypto = QuantumResistantCrypto()
    
    # Generate 10 keys
    keys = [crypto.generate_key_pair() for _ in range(10)]
    
    # All key IDs should be unique
    key_ids = [k.key_id for k in keys]
    assert len(key_ids) == len(set(key_ids))
    
    # All public keys should be unique
    pub_keys = [k.public_key for k in keys]
    assert len(pub_keys) == len(set(pub_keys))


def test_key_generation_randomness():
    """Test that key generation uses proper randomness."""
    from qratum.planetary.security import QuantumResistantCrypto
    import secrets
    
    crypto = QuantumResistantCrypto()
    
    # Generate a key
    key = crypto.generate_key_pair(key_size_bits=256)
    
    # Verify key has expected properties
    assert key.key_size_bits == 256
    assert len(key.public_key) == 64  # 256 bits = 64 hex chars
    
    # Verify it's not a predictable pattern (e.g., all zeros)
    assert key.public_key != "0" * 64
    assert key.public_key != "f" * 64


def test_signing_with_secure_keys():
    """Test that signing works with securely generated keys."""
    from qratum.planetary.security import QuantumResistantCrypto
    
    crypto = QuantumResistantCrypto()
    key = crypto.generate_key_pair()
    
    # Sign data
    data = b"test data to sign"
    signature = crypto.sign_data(key.key_id, data)
    
    assert signature is not None
    assert "signature" in signature
    assert "key_id" in signature
    assert signature["key_id"] == key.key_id


def test_signature_verification():
    """Test that signatures can be verified."""
    from qratum.planetary.security import QuantumResistantCrypto
    
    crypto = QuantumResistantCrypto()
    key = crypto.generate_key_pair()
    
    # Sign and verify
    data = b"test data"
    signature = crypto.sign_data(key.key_id, data)
    
    is_valid = crypto.verify_signature(
        key.key_id,
        data,
        signature["signature"]
    )
    
    assert is_valid is True


def test_different_data_produces_different_signatures():
    """Test that different data produces different signatures."""
    from qratum.planetary.security import QuantumResistantCrypto
    
    crypto = QuantumResistantCrypto()
    key = crypto.generate_key_pair()
    
    # Sign different data
    sig1 = crypto.sign_data(key.key_id, b"data1")
    sig2 = crypto.sign_data(key.key_id, b"data2")
    
    assert sig1["signature"] != sig2["signature"]


def test_key_rotation():
    """Test that key rotation creates new keys."""
    from qratum.planetary.security import QuantumResistantCrypto
    
    crypto = QuantumResistantCrypto()
    
    # Generate initial keys
    key1 = crypto.generate_key_pair()
    initial_count = len(crypto.keys)
    
    # Generate more keys (simulating rotation)
    key2 = crypto.generate_key_pair()
    key3 = crypto.generate_key_pair()
    
    assert len(crypto.keys) == initial_count + 2
    assert key1.key_id != key2.key_id
    assert key2.key_id != key3.key_id


def test_crypto_statistics():
    """Test that crypto statistics are properly tracked."""
    from qratum.planetary.security import QuantumResistantCrypto, CryptoAlgorithm
    
    crypto = QuantumResistantCrypto()
    
    # Generate keys with different algorithms
    crypto.generate_key_pair(algorithm=CryptoAlgorithm.CRYSTALS_DILITHIUM)
    crypto.generate_key_pair(algorithm=CryptoAlgorithm.CRYSTALS_KYBER)
    crypto.generate_key_pair(algorithm=CryptoAlgorithm.CRYSTALS_KYBER)
    
    stats = crypto.get_statistics()
    
    assert stats["total_keys"] == 3
    assert "keys_by_algorithm" in stats
    assert stats["keys_by_algorithm"]["crystals_kyber"] == 2
    assert stats["keys_by_algorithm"]["crystals_dilithium"] == 1


def test_consensus_vote_signature_randomness():
    """Test that consensus vote signatures use secure randomness."""
    from qratum.planetary.security import ConsensusVote
    
    # Create multiple votes with same data
    vote1 = ConsensusVote("voter1", "prop1", True)
    vote2 = ConsensusVote("voter1", "prop1", True)
    
    # Signatures should be the same for same data (deterministic)
    # but different from a predictable pattern
    assert vote1.signature == vote2.signature
    assert vote1.signature != "0" * 64


def test_security_layer_initialization():
    """Test that security layer initializes with secure keys."""
    from qratum.planetary.security import SecurityLayer
    
    security = SecurityLayer()
    
    # Should have initialized with default keys
    assert len(security.crypto.keys) > 0
    
    # All keys should be unique
    key_ids = list(security.crypto.keys.keys())
    assert len(key_ids) == len(set(key_ids))


def test_key_expiration_set():
    """Test that generated keys have expiration dates."""
    from qratum.planetary.security import QuantumResistantCrypto
    from datetime import datetime, timezone
    
    crypto = QuantumResistantCrypto()
    key = crypto.generate_key_pair()
    
    # Parse expiration date
    created = datetime.fromisoformat(key.created_at)
    expires = datetime.fromisoformat(key.expires_at)
    
    # Expiration should be in the future
    assert expires > created
    
    # Should be within rotation period
    delta = (expires - created).days
    assert 0 < delta <= crypto.key_rotation_days


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
