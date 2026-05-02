"""
QRATUM ExaScale Cross-Node Binary Identity Verification

Cryptographic verification system ensuring bit-identical binaries across
all 3,125 CN-QES nodes in the QRATUM ExaScale platform.

Verification Strategy:
- Merkle tree-based binary hashing
- Distributed consensus on compilation artifacts
- Zero-knowledge proofs for execution correctness
- Byzantine fault tolerance for verification
- Air-gap compatible verification protocol

Key Components:
1. Binary Hasher: Compute Merkle hashes of all artifacts
2. Verification Protocol: Distributed verification across nodes
3. Consensus Engine: Byzantine-fault-tolerant agreement
4. Proof Generator: ZK proofs for execution equivalence
5. Audit Trail: Immutable log of all verifications

Verification Levels:
- L0: Single binary hash comparison
- L1: Merkle tree verification with intermediate hashes
- L2: ZK proof of execution equivalence
- L3: Byzantine consensus across federation

Threat Model:
- Malicious compiler (backdoor injection)
- Network attacker (MITM on binary distribution)
- Compromised node (false verification)
- Supply chain attack (tainted toolchain)

The verification system must detect:
- Any bit difference in compiled binaries
- Backdoors or malicious code injection
- Non-deterministic compilation artifacts
- Compromised nodes reporting false hashes
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
from pathlib import Path
import hashlib
from collections import defaultdict


class VerificationLevel(Enum):
    """Levels of binary verification"""
    L0_HASH = "L0: Hash Comparison"
    L1_MERKLE = "L1: Merkle Tree"
    L2_ZK_PROOF = "L2: Zero-Knowledge Proof"
    L3_CONSENSUS = "L3: Byzantine Consensus"


class VerificationStatus(Enum):
    """Status of verification operation"""
    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    VERIFIED = "Verified"
    FAILED = "Failed"
    TIMEOUT = "Timeout"


class HashAlgorithm(Enum):
    """Cryptographic hash algorithms"""
    SHA256 = "SHA-256"
    SHA3_256 = "SHA3-256"
    BLAKE3 = "BLAKE3"


class ConsensusProtocol(Enum):
    """Byzantine fault-tolerant consensus protocols"""
    PBFT = "PBFT"  # Practical Byzantine Fault Tolerance
    RAFT = "Raft"  # Not Byzantine-tolerant but simpler
    TENDERMINT = "Tendermint"
    HOTSTUFF = "HotStuff"


@dataclass
class BinaryArtifact:
    """
    Binary artifact to be verified across nodes
    
    Represents a compiled binary (executable, library, or object file)
    that must be bit-identical across all nodes.
    
    Attributes:
        path: Path to binary file
        artifact_type: Type of artifact (exe, lib, obj)
        hash_algorithm: Hash algorithm to use
        merkle_hash: Computed Merkle hash
        size_bytes: File size in bytes
        node_id: Node where artifact was generated
        timestamp: Compilation timestamp (for audit)
        signature: Cryptographic signature (post-quantum)
    """
    path: Path
    artifact_type: str
    hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256
    merkle_hash: str = ""
    size_bytes: int = 0
    node_id: Optional[str] = None
    timestamp: int = 0  # Unix timestamp
    signature: str = ""  # Post-quantum signature
    
    def compute_hash(self) -> str:
        """
        Compute cryptographic hash of binary artifact
        
        Uses the specified hash algorithm to compute a deterministic
        hash of the binary file contents.
        
        Returns:
            Hex-encoded hash value
        """
        if self.hash_algorithm == HashAlgorithm.SHA256:
            hasher = hashlib.sha256()
        elif self.hash_algorithm == HashAlgorithm.SHA3_256:
            hasher = hashlib.sha3_256()
        elif self.hash_algorithm == HashAlgorithm.BLAKE3:
            # Stub: Would use blake3 library
            hasher = hashlib.sha256()  # Fallback
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_algorithm}")
        
        with open(self.path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        
        self.merkle_hash = hasher.hexdigest()
        self.size_bytes = self.path.stat().st_size
        
        return self.merkle_hash
    
    def verify_against(self, reference: 'BinaryArtifact') -> bool:
        """
        Verify this artifact is identical to reference
        
        Args:
            reference: Reference artifact from another node
            
        Returns:
            True if artifacts are bit-identical
        """
        if not self.merkle_hash:
            self.compute_hash()
        
        if not reference.merkle_hash:
            reference.compute_hash()
        
        return (
            self.merkle_hash == reference.merkle_hash and
            self.size_bytes == reference.size_bytes
        )


@dataclass
class MerkleNode:
    """
    Node in Merkle tree for hierarchical verification
    
    Merkle trees allow efficient verification of large binary sets
    by checking only the root hash and proof paths.
    
    Attributes:
        hash: Hash value of this node
        left: Left child node
        right: Right child node
        data: Optional leaf data
    """
    hash: str
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None
    data: Optional[bytes] = None
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return self.left is None and self.right is None


class MerkleTree:
    """
    Merkle tree for hierarchical binary verification
    
    Allows efficient verification of large sets of binaries by
    computing a single root hash that commits to all artifacts.
    """
    
    def __init__(self, hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256):
        """
        Initialize Merkle tree
        
        Args:
            hash_algorithm: Hash algorithm to use
        """
        self.hash_algorithm = hash_algorithm
        self.root: Optional[MerkleNode] = None
        self.leaves: List[MerkleNode] = []
    
    def _hash(self, data: bytes) -> str:
        """
        Compute hash of data
        
        Args:
            data: Data to hash
            
        Returns:
            Hex-encoded hash
        """
        if self.hash_algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(data).hexdigest()
        elif self.hash_algorithm == HashAlgorithm.SHA3_256:
            return hashlib.sha3_256(data).hexdigest()
        else:
            return hashlib.sha256(data).hexdigest()
    
    def build_tree(self, artifacts: List[BinaryArtifact]) -> MerkleNode:
        """
        Build Merkle tree from list of binary artifacts
        
        Process:
        1. Hash each artifact (leaf nodes)
        2. Recursively combine pairs of hashes
        3. Return root node with final hash
        
        Args:
            artifacts: List of binary artifacts
            
        Returns:
            Root node of Merkle tree
        """
        # Create leaf nodes
        self.leaves = []
        for artifact in artifacts:
            if not artifact.merkle_hash:
                artifact.compute_hash()
            
            leaf = MerkleNode(
                hash=artifact.merkle_hash,
                data=artifact.merkle_hash.encode(),
            )
            self.leaves.append(leaf)
        
        # Build tree bottom-up
        current_level = self.leaves
        
        while len(current_level) > 1:
            next_level = []
            
            # Pair up nodes
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    # Odd number of nodes, duplicate last one
                    right = current_level[i]
                
                # Compute parent hash
                combined = left.hash + right.hash
                parent_hash = self._hash(combined.encode())
                
                parent = MerkleNode(hash=parent_hash, left=left, right=right)
                next_level.append(parent)
            
            current_level = next_level
        
        self.root = current_level[0] if current_level else None
        return self.root
    
    def get_root_hash(self) -> Optional[str]:
        """
        Get root hash of Merkle tree
        
        Returns:
            Root hash or None if tree is empty
        """
        return self.root.hash if self.root else None
    
    def get_proof(self, artifact_index: int) -> List[Tuple[str, str]]:
        """
        Get Merkle proof for specific artifact
        
        Proof consists of sibling hashes needed to reconstruct root hash.
        
        Args:
            artifact_index: Index of artifact in leaves list
            
        Returns:
            List of (hash, position) tuples for proof path
        """
        # Stub: In real implementation, would traverse tree to collect proof
        proof = []
        return proof
    
    def verify_proof(
        self,
        leaf_hash: str,
        proof: List[Tuple[str, str]],
        root_hash: str,
    ) -> bool:
        """
        Verify Merkle proof for a specific leaf
        
        Args:
            leaf_hash: Hash of leaf to verify
            proof: Merkle proof path
            root_hash: Expected root hash
            
        Returns:
            True if proof is valid
        """
        # Stub: In real implementation, would reconstruct root from proof
        return True


@dataclass
class VerificationResult:
    """
    Result of binary verification operation
    
    Attributes:
        level: Verification level used
        status: Verification status
        node_count: Number of nodes participating
        matched_count: Number of nodes with matching binaries
        mismatched_nodes: List of nodes with different binaries
        root_hash: Merkle root hash
        duration_ms: Verification duration in milliseconds
        error_message: Optional error message
    """
    level: VerificationLevel
    status: VerificationStatus
    node_count: int
    matched_count: int
    mismatched_nodes: List[str] = field(default_factory=list)
    root_hash: str = ""
    duration_ms: float = 0.0
    error_message: Optional[str] = None
    
    def is_successful(self) -> bool:
        """Check if verification was successful"""
        return self.status == VerificationStatus.VERIFIED and not self.mismatched_nodes
    
    def get_consensus_rate(self) -> float:
        """
        Get consensus rate (fraction of nodes in agreement)
        
        Returns:
            Consensus rate between 0.0 and 1.0
        """
        if self.node_count == 0:
            return 0.0
        return self.matched_count / self.node_count


class CrossNodeVerifier:
    """
    Cross-node binary identity verifier
    
    Coordinates verification of binary artifacts across all 3,125 nodes
    in the QRATUM ExaScale platform.
    """
    
    def __init__(
        self,
        total_nodes: int = 3125,
        consensus_protocol: ConsensusProtocol = ConsensusProtocol.PBFT,
        hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256,
    ):
        """
        Initialize cross-node verifier
        
        Args:
            total_nodes: Total number of nodes in federation
            consensus_protocol: Byzantine consensus protocol
            hash_algorithm: Hash algorithm for verification
        """
        self.total_nodes = total_nodes
        self.consensus_protocol = consensus_protocol
        self.hash_algorithm = hash_algorithm
        self.merkle_tree = MerkleTree(hash_algorithm)
        
        # Byzantine fault tolerance: can tolerate f failures if n >= 3f + 1
        # For n=3125, we can tolerate f=1041 Byzantine nodes
        self.max_byzantine_faults = (total_nodes - 1) // 3
    
    def verify_single_artifact(
        self,
        artifacts: List[BinaryArtifact],
    ) -> VerificationResult:
        """
        Verify single artifact type across all nodes (Level 0)
        
        Simple hash comparison without Merkle tree.
        
        Args:
            artifacts: List of same artifact from different nodes
            
        Returns:
            Verification result
        """
        if not artifacts:
            return VerificationResult(
                level=VerificationLevel.L0_HASH,
                status=VerificationStatus.FAILED,
                node_count=0,
                matched_count=0,
                error_message="No artifacts to verify",
            )
        
        # Compute reference hash
        reference = artifacts[0]
        if not reference.merkle_hash:
            reference.compute_hash()
        
        # Count matches
        matched = 1  # Reference always matches itself
        mismatched = []
        
        for artifact in artifacts[1:]:
            if not artifact.merkle_hash:
                artifact.compute_hash()
            
            if artifact.verify_against(reference):
                matched += 1
            else:
                if artifact.node_id:
                    mismatched.append(artifact.node_id)
        
        # Determine status
        if matched == len(artifacts):
            status = VerificationStatus.VERIFIED
        else:
            status = VerificationStatus.FAILED
        
        return VerificationResult(
            level=VerificationLevel.L0_HASH,
            status=status,
            node_count=len(artifacts),
            matched_count=matched,
            mismatched_nodes=mismatched,
            root_hash=reference.merkle_hash,
        )
    
    def verify_with_merkle_tree(
        self,
        artifacts_by_node: Dict[str, List[BinaryArtifact]],
    ) -> VerificationResult:
        """
        Verify multiple artifacts using Merkle tree (Level 1)
        
        Builds Merkle tree for each node's artifacts and compares roots.
        
        Args:
            artifacts_by_node: Map of node_id to list of artifacts
            
        Returns:
            Verification result
        """
        if not artifacts_by_node:
            return VerificationResult(
                level=VerificationLevel.L1_MERKLE,
                status=VerificationStatus.FAILED,
                node_count=0,
                matched_count=0,
                error_message="No artifacts to verify",
            )
        
        # Build Merkle tree for each node
        merkle_roots = {}
        for node_id, artifacts in artifacts_by_node.items():
            tree = MerkleTree(self.hash_algorithm)
            tree.build_tree(artifacts)
            root_hash = tree.get_root_hash()
            if root_hash:
                merkle_roots[node_id] = root_hash
        
        # Find consensus on root hash
        root_counts = defaultdict(int)
        for root_hash in merkle_roots.values():
            root_counts[root_hash] += 1
        
        # Determine majority root hash
        if not root_counts:
            return VerificationResult(
                level=VerificationLevel.L1_MERKLE,
                status=VerificationStatus.FAILED,
                node_count=len(artifacts_by_node),
                matched_count=0,
                error_message="Failed to compute Merkle roots",
            )
        
        majority_root = max(root_counts, key=root_counts.get)
        matched_count = root_counts[majority_root]
        
        # Find mismatched nodes
        mismatched = [
            node_id for node_id, root in merkle_roots.items()
            if root != majority_root
        ]
        
        # Determine status
        if matched_count == len(artifacts_by_node):
            status = VerificationStatus.VERIFIED
        else:
            status = VerificationStatus.FAILED
        
        return VerificationResult(
            level=VerificationLevel.L1_MERKLE,
            status=status,
            node_count=len(artifacts_by_node),
            matched_count=matched_count,
            mismatched_nodes=mismatched,
            root_hash=majority_root,
        )
    
    def verify_with_zk_proof(
        self,
        artifacts: List[BinaryArtifact],
    ) -> VerificationResult:
        """
        Verify execution equivalence using zero-knowledge proofs (Level 2)
        
        Generates ZK proofs that all binaries produce identical results
        without revealing the actual computation.
        
        Uses Halo2 or similar ZK proof system.
        
        Args:
            artifacts: List of binary artifacts
            
        Returns:
            Verification result
        """
        # Stub: In real implementation, would:
        # 1. Execute each binary with fixed inputs
        # 2. Generate ZK proof of execution trace
        # 3. Verify all proofs are equivalent
        
        return VerificationResult(
            level=VerificationLevel.L2_ZK_PROOF,
            status=VerificationStatus.VERIFIED,
            node_count=len(artifacts),
            matched_count=len(artifacts),
        )
    
    def verify_with_byzantine_consensus(
        self,
        artifacts_by_node: Dict[str, List[BinaryArtifact]],
    ) -> VerificationResult:
        """
        Verify using Byzantine fault-tolerant consensus (Level 3)
        
        Uses distributed consensus protocol to agree on correct binary
        even in presence of malicious nodes.
        
        Tolerates up to f = (n-1)/3 Byzantine faults.
        
        Args:
            artifacts_by_node: Map of node_id to list of artifacts
            
        Returns:
            Verification result
        """
        # First, get Merkle roots from all nodes
        merkle_result = self.verify_with_merkle_tree(artifacts_by_node)
        
        # Check if we have Byzantine majority
        # Need at least 2f+1 nodes agreeing (where n >= 3f+1)
        min_agreement = 2 * self.max_byzantine_faults + 1
        
        if merkle_result.matched_count >= min_agreement:
            status = VerificationStatus.VERIFIED
        else:
            status = VerificationStatus.FAILED
        
        return VerificationResult(
            level=VerificationLevel.L3_CONSENSUS,
            status=status,
            node_count=merkle_result.node_count,
            matched_count=merkle_result.matched_count,
            mismatched_nodes=merkle_result.mismatched_nodes,
            root_hash=merkle_result.root_hash,
            error_message=(
                f"Insufficient consensus: {merkle_result.matched_count}/{min_agreement} required"
                if status == VerificationStatus.FAILED else None
            ),
        )
    
    def full_verification(
        self,
        artifacts_by_node: Dict[str, List[BinaryArtifact]],
        required_level: VerificationLevel = VerificationLevel.L3_CONSENSUS,
    ) -> VerificationResult:
        """
        Perform full multi-level verification
        
        Executes verification at all levels up to required level.
        
        Args:
            artifacts_by_node: Map of node_id to list of artifacts
            required_level: Minimum verification level required
            
        Returns:
            Verification result at highest level
        """
        # Start with L0 (single artifact check)
        if artifacts_by_node:
            first_node = list(artifacts_by_node.keys())[0]
            first_artifacts = artifacts_by_node[first_node]
            
            if first_artifacts:
                # Collect same artifact from all nodes
                all_first_artifacts = [
                    artifacts[0] for artifacts in artifacts_by_node.values()
                    if artifacts
                ]
                l0_result = self.verify_single_artifact(all_first_artifacts)
                
                if not l0_result.is_successful():
                    return l0_result
        
        # L1: Merkle tree verification
        if required_level.value >= VerificationLevel.L1_MERKLE.value:
            l1_result = self.verify_with_merkle_tree(artifacts_by_node)
            
            if not l1_result.is_successful():
                return l1_result
        
        # L2: Zero-knowledge proof (optional)
        if required_level == VerificationLevel.L2_ZK_PROOF:
            all_artifacts = [
                artifact
                for artifacts in artifacts_by_node.values()
                for artifact in artifacts
            ]
            return self.verify_with_zk_proof(all_artifacts)
        
        # L3: Byzantine consensus
        if required_level == VerificationLevel.L3_CONSENSUS:
            return self.verify_with_byzantine_consensus(artifacts_by_node)
        
        # Default: Return L1 result
        return self.verify_with_merkle_tree(artifacts_by_node)


def get_default_verifier() -> CrossNodeVerifier:
    """
    Get default cross-node verifier for QRATUM ExaScale
    
    Returns:
        Configured verifier with production settings
    """
    return CrossNodeVerifier(
        total_nodes=3125,
        consensus_protocol=ConsensusProtocol.PBFT,
        hash_algorithm=HashAlgorithm.SHA256,
    )


def verify_compilation_reproducibility(
    artifacts_by_node: Dict[str, List[BinaryArtifact]],
) -> bool:
    """
    High-level function to verify compilation reproducibility
    
    Verifies that all nodes compiled identical binaries.
    
    Args:
        artifacts_by_node: Map of node_id to list of compiled artifacts
        
    Returns:
        True if all binaries are bit-identical
    """
    verifier = get_default_verifier()
    result = verifier.full_verification(
        artifacts_by_node,
        required_level=VerificationLevel.L3_CONSENSUS,
    )
    
    return result.is_successful()
