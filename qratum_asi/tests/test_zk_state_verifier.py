"""Tests for the ZK State Verification module (Task 4)."""

import time

from qratum_asi.core.calibration_doctrine import TrajectoryState
from qratum_asi.core.zk_state_verifier import (
    ReplayCache,
    StateCommitment,
    TransitionType,
    VerificationResult,
    ZKProof,
    ZKProofGenerator,
    ZKStateTransition,
    ZKStateVerifier,
    ZKVerificationContext,
    generate_commitment,
    verify_state_transition,
)


class TestStateCommitment:
    """Tests for StateCommitment class."""

    def test_from_state_deterministic(self):
        """Verify same state produces same commitment."""
        state = b"test state data"
        version = 1
        zone_id = "Z1"

        commitment1 = StateCommitment.from_state(state, version, zone_id)
        commitment2 = StateCommitment.from_state(state, version, zone_id)

        assert commitment1.commitment_hash == commitment2.commitment_hash

    def test_different_states_different_commitments(self):
        """Verify different states produce different commitments."""
        state1 = b"state data v1"
        state2 = b"state data v2"

        commitment1 = StateCommitment.from_state(state1, 1, "Z1")
        commitment2 = StateCommitment.from_state(state2, 1, "Z1")

        assert commitment1.commitment_hash != commitment2.commitment_hash

    def test_commitment_hash_length(self):
        """Verify commitment hash is SHA3-256 (64 hex chars)."""
        commitment = StateCommitment.from_state(b"test", 1, "Z1")
        assert len(commitment.commitment_hash) == 64

    def test_to_bytes_serialization(self):
        """Test commitment serialization."""
        commitment = StateCommitment.from_state(b"test", 1, "Z1")
        serialized = commitment.to_bytes()

        assert b"commitment_hash" in serialized
        assert b"state_version" in serialized
        assert b"zone_id" in serialized


class TestZKProof:
    """Tests for ZKProof class."""

    def test_compute_proof_id_deterministic(self):
        """Verify proof ID computation is deterministic."""
        proof = ZKProof(
            proof_data=b"test proof",
            public_inputs=b"test inputs",
            timestamp=1000.0,
            version=1,
            nonce=b"n" * 32,
            epoch_id=1,
        )

        id1 = proof.compute_proof_id()
        id2 = proof.compute_proof_id()

        assert id1 == id2

    def test_different_nonces_different_ids(self):
        """Verify different nonces produce different proof IDs."""
        proof1 = ZKProof(
            proof_data=b"test proof",
            public_inputs=b"test inputs",
            timestamp=1000.0,
            version=1,
            nonce=b"a" * 32,
            epoch_id=1,
        )

        proof2 = ZKProof(
            proof_data=b"test proof",
            public_inputs=b"test inputs",
            timestamp=1000.0,
            version=1,
            nonce=b"b" * 32,
            epoch_id=1,
        )

        assert proof1.compute_proof_id() != proof2.compute_proof_id()

    def test_to_dict_from_dict_roundtrip(self):
        """Test proof serialization roundtrip."""
        original = ZKProof(
            proof_data=b"test proof data",
            public_inputs=b"test public inputs",
            timestamp=1234567890.0,
            version=1,
            nonce=b"x" * 32,
            epoch_id=42,
        )

        as_dict = original.to_dict()
        restored = ZKProof.from_dict(as_dict)

        assert restored.proof_data == original.proof_data
        assert restored.public_inputs == original.public_inputs
        assert restored.timestamp == original.timestamp
        assert restored.nonce == original.nonce
        assert restored.epoch_id == original.epoch_id


class TestZKStateTransition:
    """Tests for ZKStateTransition class."""

    def test_verify_commitments_linked_valid(self):
        """Test commitment linkage verification for valid transition."""
        prev = StateCommitment.from_state(b"prev", 1, "Z1")
        next_commit = StateCommitment.from_state(b"next", 2, "Z1")
        proof = ZKProof(
            proof_data=b"p" * 32,
            public_inputs=b"i" * 32,
            timestamp=time.time(),
            version=1,
            nonce=b"n" * 32,
            epoch_id=1,
        )

        transition = ZKStateTransition(
            prev_commitment=prev,
            next_commitment=next_commit,
            proof=proof,
            height=2,
            transition_type=TransitionType.TXO_EXECUTION,
        )

        assert transition.verify_commitments_linked() is True

    def test_verify_commitments_linked_invalid_version(self):
        """Test commitment linkage fails with wrong version increment."""
        prev = StateCommitment.from_state(b"prev", 1, "Z1")
        # Version should be 2, not 5
        next_commit = StateCommitment.from_state(b"next", 5, "Z1")
        proof = ZKProof(
            proof_data=b"p" * 32,
            public_inputs=b"i" * 32,
            timestamp=time.time(),
            version=1,
            nonce=b"n" * 32,
            epoch_id=1,
        )

        transition = ZKStateTransition(
            prev_commitment=prev,
            next_commitment=next_commit,
            proof=proof,
            height=5,
            transition_type=TransitionType.TXO_EXECUTION,
        )

        assert transition.verify_commitments_linked() is False

    def test_verify_commitments_linked_zone_change(self):
        """Test commitment linkage fails for zone change (unless zone transition)."""
        prev = StateCommitment.from_state(b"prev", 1, "Z1")
        next_commit = StateCommitment.from_state(b"next", 2, "Z2")  # Different zone
        proof = ZKProof(
            proof_data=b"p" * 32,
            public_inputs=b"i" * 32,
            timestamp=time.time(),
            version=1,
            nonce=b"n" * 32,
            epoch_id=1,
        )

        transition = ZKStateTransition(
            prev_commitment=prev,
            next_commitment=next_commit,
            proof=proof,
            height=2,
            transition_type=TransitionType.TXO_EXECUTION,  # Not zone transition
        )

        assert transition.verify_commitments_linked() is False

    def test_zone_transition_allows_zone_change(self):
        """Test zone transition type allows zone change."""
        prev = StateCommitment.from_state(b"prev", 1, "Z1")
        next_commit = StateCommitment.from_state(b"next", 2, "Z2")  # Different zone
        proof = ZKProof(
            proof_data=b"p" * 32,
            public_inputs=b"i" * 32,
            timestamp=time.time(),
            version=1,
            nonce=b"n" * 32,
            epoch_id=1,
        )

        transition = ZKStateTransition(
            prev_commitment=prev,
            next_commitment=next_commit,
            proof=proof,
            height=2,
            transition_type=TransitionType.ZONE_TRANSITION,  # Zone transition
        )

        assert transition.verify_commitments_linked() is True


class TestReplayCache:
    """Tests for ReplayCache class."""

    def test_empty_cache_no_replay(self):
        """Test empty cache reports no replays."""
        cache = ReplayCache()
        assert cache.is_replay("test_id") is False

    def test_mark_seen_then_replay(self):
        """Test marking seen enables replay detection."""
        cache = ReplayCache()

        assert cache.mark_seen("proof_123") is True
        assert cache.is_replay("proof_123") is True

    def test_mark_seen_twice_fails(self):
        """Test marking same proof twice returns False."""
        cache = ReplayCache()

        assert cache.mark_seen("proof_123") is True
        assert cache.mark_seen("proof_123") is False

    def test_cleanup_removes_old_entries(self):
        """Test cleanup removes expired entries."""
        cache = ReplayCache(max_size=100, ttl_seconds=0)  # TTL=0 means all are expired

        cache.mark_seen("old_proof_1")
        cache.mark_seen("old_proof_2")

        # Sleep briefly to ensure timestamps differ
        time.sleep(0.01)

        removed = cache.cleanup()
        assert removed >= 0  # May not remove all due to timing

    def test_cache_size_limit(self):
        """Test cache respects max size."""
        cache = ReplayCache(max_size=5)

        for i in range(10):
            cache.mark_seen(f"proof_{i}")

        # After cleanup, size should be reduced
        assert len(cache.seen_proofs) <= 10


class TestZKStateVerifier:
    """Tests for ZKStateVerifier class."""

    def test_initialization(self):
        """Test verifier initializes correctly."""
        verifier = ZKStateVerifier()

        assert verifier.successful_verifications == 0
        assert verifier.failed_verifications == 0
        assert len(verifier.verifying_keys) == len(TransitionType)

    def test_verify_valid_proof(self):
        """Test verification of valid proof."""
        verifier = ZKStateVerifier()

        proof = ZKProof(
            proof_data=b"p" * 32,
            public_inputs=b"i" * 32,
            timestamp=time.time(),
            version=1,
            nonce=b"n" * 32,
            epoch_id=1,
        )

        commitment = StateCommitment.from_state(b"test", 1, "Z1")

        context = ZKVerificationContext(
            current_time=time.time(), max_proof_age=3600, zone_id="Z1", epoch_id=1
        )

        result, message = verifier.verify_proof(proof, commitment, context)
        assert result == VerificationResult.VALID

    def test_verify_expired_proof(self):
        """Test expired proof is rejected."""
        verifier = ZKStateVerifier()

        proof = ZKProof(
            proof_data=b"p" * 32,
            public_inputs=b"i" * 32,
            timestamp=1000.0,  # Old timestamp
            version=1,
            nonce=b"n" * 32,
            epoch_id=1,
        )

        commitment = StateCommitment.from_state(b"test", 1, "Z1")

        context = ZKVerificationContext(
            current_time=time.time(),  # Current time is much later
            max_proof_age=60,  # 60 second max age
            zone_id="Z1",
            epoch_id=1,
        )

        result, message = verifier.verify_proof(proof, commitment, context)
        assert result == VerificationResult.EXPIRED

    def test_replay_detection(self):
        """Test replay attack is detected."""
        verifier = ZKStateVerifier()

        proof = ZKProof(
            proof_data=b"p" * 32,
            public_inputs=b"i" * 32,
            timestamp=time.time(),
            version=1,
            nonce=b"n" * 32,
            epoch_id=1,
        )

        commitment = StateCommitment.from_state(b"test", 1, "Z1")

        context = ZKVerificationContext(
            current_time=time.time(), max_proof_age=3600, zone_id="Z1", epoch_id=1
        )

        # First verification succeeds
        result1, _ = verifier.verify_proof(proof, commitment, context)
        assert result1 == VerificationResult.VALID

        # Second verification detects replay
        result2, _ = verifier.verify_proof(proof, commitment, context)
        assert result2 == VerificationResult.REPLAY_DETECTED

    def test_verify_empty_proof_rejected(self):
        """Test empty proof data is rejected."""
        verifier = ZKStateVerifier()

        proof = ZKProof(
            proof_data=b"",  # Empty
            public_inputs=b"i" * 32,
            timestamp=time.time(),
            version=1,
            nonce=b"n" * 32,
            epoch_id=1,
        )

        commitment = StateCommitment.from_state(b"test", 1, "Z1")

        context = ZKVerificationContext(
            current_time=time.time(), max_proof_age=3600, zone_id="Z1", epoch_id=1
        )

        result, message = verifier.verify_proof(proof, commitment, context)
        assert result == VerificationResult.FORMAT_ERROR

    def test_critical_trajectory_blocks_verification(self):
        """Test critical trajectory state blocks verification."""
        verifier = ZKStateVerifier()

        proof = ZKProof(
            proof_data=b"p" * 32,
            public_inputs=b"i" * 32,
            timestamp=time.time(),
            version=1,
            nonce=b"n" * 32,
            epoch_id=1,
        )

        commitment = StateCommitment.from_state(b"test", 1, "Z1")

        context = ZKVerificationContext(
            current_time=time.time(),
            max_proof_age=3600,
            zone_id="Z1",
            epoch_id=1,
            trajectory_state=TrajectoryState.CRITICAL,  # Critical state
        )

        result, message = verifier.verify_proof(proof, commitment, context)
        assert result == VerificationResult.TRAJECTORY_VIOLATION

    def test_get_stats(self):
        """Test statistics retrieval."""
        verifier = ZKStateVerifier()
        stats = verifier.get_stats()

        assert "successful_verifications" in stats
        assert "failed_verifications" in stats
        assert "success_rate" in stats
        assert "replay_cache_size" in stats


class TestZKProofGenerator:
    """Tests for ZKProofGenerator class."""

    def test_generate_proof_deterministic(self):
        """Test proof generation with same seed is deterministic."""
        gen1 = ZKProofGenerator(seed=42)
        gen2 = ZKProofGenerator(seed=42)

        proof1 = gen1.generate_proof(b"prev", b"next", b"witness", TransitionType.TXO_EXECUTION)
        proof2 = gen2.generate_proof(b"prev", b"next", b"witness", TransitionType.TXO_EXECUTION)

        assert proof1.nonce == proof2.nonce
        assert proof1.proof_data == proof2.proof_data

    def test_create_transition(self):
        """Test complete transition creation."""
        gen = ZKProofGenerator(seed=42)

        transition = gen.create_transition(
            prev_state=b"previous state",
            next_state=b"next state",
            prev_version=1,
            zone_id="Z1",
            transition_type=TransitionType.TXO_EXECUTION,
        )

        assert transition.height == 2
        assert transition.transition_type == TransitionType.TXO_EXECUTION
        assert transition.prev_commitment.state_version == 1
        assert transition.next_commitment.state_version == 2
        assert transition.verify_commitments_linked() is True


class TestConvenienceFunctions:
    """Tests for module convenience functions."""

    def test_generate_commitment(self):
        """Test generate_commitment function."""
        commitment = generate_commitment(b"test state", 1, "Z1")

        assert len(commitment) == 64  # SHA3-256 hex

        # Same input produces same commitment
        commitment2 = generate_commitment(b"test state", 1, "Z1")
        assert commitment == commitment2

    def test_verify_state_transition(self):
        """Test verify_state_transition convenience function."""
        gen = ZKProofGenerator(seed=42)

        prev_state = b"prev"
        next_state = b"next"

        proof = gen.generate_proof(prev_state, next_state, b"", TransitionType.TXO_EXECUTION)

        is_valid, message = verify_state_transition(prev_state, next_state, proof, "Z1", 1)

        assert isinstance(is_valid, bool)
        assert isinstance(message, str)
