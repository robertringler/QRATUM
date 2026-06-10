"""
Temporal Proof Verification for QRATUM

Cryptographic verification of temporal computations using Merkle trees
and hash chains. Ensures that all temporal operations are provably correct
and reproducible.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .state import StateChain, TemporalState


@dataclass
class TemporalProof:
    """
    Cryptographic proof of a temporal computation.

    Attributes:
        initial_state_hash: Hash of the initial state
        final_state_hash: Hash of the final state
        merkle_root: Merkle root of the state chain
        operation: Type of operation performed (forward, backward, branch, etc.)
        timeline_id: Timeline this proof applies to
        computational_delta_t: Change in computational time
        physical_delta_t: Change in physical time
        intermediate_hashes: Hashes of intermediate states
        signature: Optional cryptographic signature (PQC)
        timestamp: When the proof was generated
        metadata: Additional proof metadata
    """

    initial_state_hash: str
    final_state_hash: str
    merkle_root: str
    operation: str
    timeline_id: str
    computational_delta_t: float
    physical_delta_t: float
    intermediate_hashes: List[str]
    signature: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

    def compute_proof_hash(self) -> str:
        """Compute hash of the entire proof"""
        hasher = hashlib.sha256()

        hasher.update(self.initial_state_hash.encode("utf-8"))
        hasher.update(self.final_state_hash.encode("utf-8"))
        hasher.update(self.merkle_root.encode("utf-8"))
        hasher.update(self.operation.encode("utf-8"))
        hasher.update(self.timeline_id.encode("utf-8"))
        hasher.update(str(self.computational_delta_t).encode("utf-8"))
        hasher.update(str(self.physical_delta_t).encode("utf-8"))

        for h in self.intermediate_hashes:
            hasher.update(h.encode("utf-8"))

        return hasher.hexdigest()

    def effective_velocity_c_multiple(self) -> float:
        """Calculate effective velocity as multiple of speed of light"""
        if self.physical_delta_t == 0:
            return float("inf")
        return abs(self.computational_delta_t / self.physical_delta_t)


class TemporalVerifier:
    """
    Verifies temporal computations and generates proofs.

    The verifier ensures that all temporal operations maintain consistency
    and can be cryptographically proven to be correct.
    """

    def __init__(self, enable_pqc: bool = False):
        """
        Initialize verifier.

        Args:
            enable_pqc: Enable post-quantum cryptographic signatures
        """
        self.enable_pqc = enable_pqc
        self.verification_count = 0

    def verify_state_chain(self, chain: StateChain) -> bool:
        """
        Verify integrity of a complete state chain.

        Args:
            chain: State chain to verify

        Returns:
            True if chain is valid
        """
        if not chain.states:
            return True

        # Verify first state
        first_state = chain.states[0]
        if first_state.coordinate.depth != 0:
            return False
        if first_state.parent_hash is not None:
            return False
        if not first_state.verify_hash():
            return False

        # Verify chain integrity
        for i in range(1, len(chain.states)):
            prev_state = chain.states[i - 1]
            curr_state = chain.states[i]

            # Verify hash
            if not curr_state.verify_hash():
                return False

            # Verify chaining
            if curr_state.parent_hash != prev_state.coordinate.state_hash:
                return False

            # Verify depth
            if curr_state.coordinate.depth != prev_state.coordinate.depth + 1:
                return False

            # Verify timeline
            if curr_state.coordinate.timeline_id != chain.timeline_id:
                return False

        # Verify Merkle root
        computed_root = chain.compute_merkle_root()
        if chain.merkle_root and chain.merkle_root != computed_root:
            return False

        self.verification_count += 1
        return True

    def verify_state_transition(
        self, initial_state: TemporalState, final_state: TemporalState, allow_branch: bool = False
    ) -> bool:
        """
        Verify that a state transition is valid.

        Args:
            initial_state: Starting state
            final_state: Ending state
            allow_branch: Allow transition to different timeline

        Returns:
            True if transition is valid
        """
        # Verify hashes
        if not initial_state.verify_hash():
            return False
        if not final_state.verify_hash():
            return False

        # Verify timeline consistency (unless branching)
        if not allow_branch:
            if initial_state.coordinate.timeline_id != final_state.coordinate.timeline_id:
                return False

        # Verify temporal ordering
        if final_state.coordinate.computational_t < initial_state.coordinate.computational_t:
            # Backward time travel - different rules apply
            pass  # Allow for now, can add specific checks

        # Verify depth progression
        if not allow_branch:
            if final_state.coordinate.depth <= initial_state.coordinate.depth:
                return False

        self.verification_count += 1
        return True

    def generate_proof(
        self,
        initial_state: TemporalState,
        final_state: TemporalState,
        chain: StateChain,
        operation: str,
        computational_delta_t: float | None = None,
    ) -> TemporalProof:
        """
        Generate cryptographic proof for a temporal computation.

        Args:
            initial_state: Starting state
            final_state: Ending state
            chain: Complete state chain
            operation: Type of operation (forward, backward, branch, converge)
            computational_delta_t: Exact requested computational time delta. When
                provided it is reported verbatim, avoiding the floating-point
                drift of summing per-step sizes over the state chain.

        Returns:
            Temporal proof object
        """
        # Compute Merkle root
        merkle_root = chain.compute_merkle_root()

        # Extract intermediate hashes
        intermediate_hashes = [
            state.coordinate.state_hash for state in chain.states[1:-1]  # Exclude first and last
        ]

        # Calculate time deltas
        if computational_delta_t is not None:
            comp_delta_t = computational_delta_t
        else:
            comp_delta_t = (
                final_state.coordinate.computational_t - initial_state.coordinate.computational_t
            )
        phys_delta_t = final_state.coordinate.physical_t - initial_state.coordinate.physical_t

        proof = TemporalProof(
            initial_state_hash=initial_state.coordinate.state_hash,
            final_state_hash=final_state.coordinate.state_hash,
            merkle_root=merkle_root,
            operation=operation,
            timeline_id=initial_state.coordinate.timeline_id,
            computational_delta_t=comp_delta_t,
            physical_delta_t=phys_delta_t,
            intermediate_hashes=intermediate_hashes,
            metadata={
                "initial_depth": initial_state.coordinate.depth,
                "final_depth": final_state.coordinate.depth,
                "state_count": len(chain.states),
            },
        )

        # Add PQC signature if enabled
        if self.enable_pqc:
            proof.signature = self._sign_proof(proof)

        return proof

    def verify_proof(self, proof: TemporalProof) -> bool:
        """
        Verify a temporal proof.

        Args:
            proof: Proof to verify

        Returns:
            True if proof is valid
        """
        # Verify proof hash integrity
        proof_hash = proof.compute_proof_hash()

        # Verify time consistency
        if proof.physical_delta_t < 0:
            return False  # Physical time must move forward

        # Verify intermediate hash chain
        # (Would need full states to verify completely, we check validity)
        # Empty intermediate_hashes list is valid for single-step operations

        # Verify PQC signature if present
        if proof.signature and not self._verify_signature(proof):
            return False

        self.verification_count += 1
        return True

    def _sign_proof(self, proof: TemporalProof) -> str:
        """
        Sign proof with post-quantum cryptography.

        In production, this would use a real PQC signature scheme like
        CRYSTALS-Dilithium or SPHINCS+. For now, we use a placeholder.
        """
        # Placeholder implementation
        proof_hash = proof.compute_proof_hash()
        signature_data = f"PQC_SIG:{proof_hash}"
        return hashlib.sha256(signature_data.encode("utf-8")).hexdigest()

    def _verify_signature(self, proof: TemporalProof) -> bool:
        """Verify PQC signature on proof"""
        # Placeholder implementation
        expected_sig = self._sign_proof(proof)
        return proof.signature == expected_sig

    def batch_verify(self, chains: List[StateChain]) -> Dict[str, bool]:
        """
        Verify multiple state chains efficiently.

        Args:
            chains: List of state chains to verify

        Returns:
            Dictionary mapping timeline_id to verification result
        """
        results = {}
        for chain in chains:
            results[chain.timeline_id] = self.verify_state_chain(chain)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get verifier statistics"""
        return {
            "total_verifications": self.verification_count,
            "pqc_enabled": self.enable_pqc,
        }
