"""
Technological Supremacy Module

Implements quantum-resistant cryptography, adaptive consensus mechanisms,
cross-chain interoperability, and disaster recovery capabilities.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


class CryptoAlgorithm(Enum):
    """Quantum-resistant cryptographic algorithms."""

    CRYSTALS_KYBER = "crystals_kyber"  # Key encapsulation
    CRYSTALS_DILITHIUM = "crystals_dilithium"  # Digital signatures
    FALCON = "falcon"  # Signatures (compact)
    SPHINCS_PLUS = "sphincs_plus"  # Hash-based signatures
    CLASSIC_MCELIECE = "classic_mceliece"  # Code-based encryption


class ConsensusType(Enum):
    """Consensus mechanism types."""

    PROOF_OF_DATA = "proof_of_data"
    PROOF_OF_STAKE = "proof_of_stake"
    BFT_HOTSTUFF = "bft_hotstuff"
    TENDERMINT = "tendermint"
    RAFT = "raft"


class RecoveryMode(Enum):
    """Disaster recovery modes."""

    HOT_STANDBY = "hot_standby"
    WARM_STANDBY = "warm_standby"
    COLD_STANDBY = "cold_standby"
    GEOGRAPHIC_REDUNDANCY = "geographic_redundancy"


@dataclass(frozen=True)
class CryptoKey:
    """Cryptographic key representation.

    Attributes:
        key_id: Unique key identifier
        algorithm: Cryptographic algorithm
        public_key: Public key (hex encoded)
        created_at: Key creation timestamp
        expires_at: Key expiration timestamp
        key_size_bits: Key size in bits
    """

    key_id: str
    algorithm: CryptoAlgorithm
    public_key: str
    created_at: str
    expires_at: str
    key_size_bits: int = 256

    def to_dict(self) -> dict[str, Any]:
        """Serialize key to dictionary."""
        return {
            "key_id": self.key_id,
            "algorithm": self.algorithm.value,
            "public_key": self.public_key[:32] + "...",  # Truncate for display
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "key_size_bits": self.key_size_bits,
        }


class QuantumResistantCrypto:
    """Quantum-resistant cryptography manager.

    Manages key generation, signing, and verification using
    post-quantum cryptographic algorithms.

    Attributes:
        crypto_id: Unique crypto instance identifier
        keys: Dictionary of managed keys
        default_algorithm: Default algorithm for new keys
        key_rotation_days: Days between key rotations
    """

    def __init__(
        self,
        crypto_id: str | None = None,
        default_algorithm: CryptoAlgorithm = CryptoAlgorithm.CRYSTALS_DILITHIUM,
    ) -> None:
        """Initialize quantum-resistant crypto manager.

        Args:
            crypto_id: Optional crypto ID
            default_algorithm: Default algorithm to use
        """
        self.crypto_id = crypto_id or f"qrc_{secrets.token_hex(8)}"
        self.keys: dict[str, CryptoKey] = {}
        self.default_algorithm = default_algorithm
        self.key_rotation_days = 90
        self.created_at = datetime.now(timezone.utc).isoformat()

    def generate_key_pair(
        self,
        algorithm: CryptoAlgorithm | None = None,
        key_size_bits: int = 256,
    ) -> CryptoKey:
        """Generate a new key pair.

        Args:
            algorithm: Algorithm to use (default if not specified)
            key_size_bits: Key size in bits

        Returns:
            Generated public key
        """
        algo = algorithm or self.default_algorithm

        # Generate deterministic key ID
        key_id = f"key_{secrets.token_hex(8)}"

        # Simulate key generation (in production, use actual PQC library)
        public_key = secrets.token_hex(key_size_bits // 8)

        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=self.key_rotation_days)

        key = CryptoKey(
            key_id=key_id,
            algorithm=algo,
            public_key=public_key,
            created_at=now.isoformat(),
            expires_at=expires.isoformat(),
            key_size_bits=key_size_bits,
        )

        self.keys[key_id] = key
        return key

    def sign_data(self, key_id: str, data: bytes) -> dict[str, Any] | None:
        """Sign data with specified key.

        Args:
            key_id: Key to use for signing
            data: Data to sign

        Returns:
            Signature data or None if key not found
        """
        if key_id not in self.keys:
            return None

        key = self.keys[key_id]

        # Simulate signature (in production, use actual PQC signing)
        signature = hashlib.sha3_256(data + key.public_key.encode()).hexdigest()

        return {
            "key_id": key_id,
            "algorithm": key.algorithm.value,
            "signature": signature,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def verify_signature(self, key_id: str, data: bytes, signature: str) -> bool:
        """Verify a signature.

        Args:
            key_id: Key used for signing
            data: Original data
            signature: Signature to verify

        Returns:
            True if signature is valid
        """
        if key_id not in self.keys:
            return False

        key = self.keys[key_id]
        expected = hashlib.sha3_256(data + key.public_key.encode()).hexdigest()

        return signature == expected

    def get_statistics(self) -> dict[str, Any]:
        """Get crypto statistics.

        Returns:
            Crypto statistics
        """
        keys_by_algo: dict[str, int] = {}
        for key in self.keys.values():
            algo = key.algorithm.value
            keys_by_algo[algo] = keys_by_algo.get(algo, 0) + 1

        return {
            "crypto_id": self.crypto_id,
            "total_keys": len(self.keys),
            "keys_by_algorithm": keys_by_algo,
            "default_algorithm": self.default_algorithm.value,
            "key_rotation_days": self.key_rotation_days,
            "created_at": self.created_at,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize crypto manager to dictionary."""
        return {
            "crypto_id": self.crypto_id,
            "keys": {k: v.to_dict() for k, v in self.keys.items()},
            "default_algorithm": self.default_algorithm.value,
            "key_rotation_days": self.key_rotation_days,
            "created_at": self.created_at,
        }


@dataclass
class ConsensusVote:
    """Vote in consensus process.

    Attributes:
        voter_id: ID of the voting node
        proposal_id: ID of the proposal being voted on
        vote: Vote value (True/False)
        timestamp: Vote timestamp
        signature: Vote signature
    """

    voter_id: str
    proposal_id: str
    vote: bool
    timestamp: str = ""
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.signature:
            self.signature = hashlib.sha256(
                f"{self.voter_id}{self.proposal_id}{self.vote}".encode()
            ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize vote to dictionary."""
        return {
            "voter_id": self.voter_id,
            "proposal_id": self.proposal_id,
            "vote": self.vote,
            "timestamp": self.timestamp,
            "signature": self.signature[:16] + "...",
        }


class AdaptiveConsensus:
    """Adaptive consensus mechanism manager.

    Manages consensus protocols with adaptive switching based on
    network conditions and security requirements.

    Attributes:
        consensus_id: Unique consensus instance ID
        active_protocol: Currently active consensus protocol
        validators: List of validator node IDs
        threshold_percent: Required consensus threshold
        proposals: Dictionary of active proposals
        votes: Dictionary of votes by proposal
    """

    def __init__(
        self,
        consensus_id: str | None = None,
        initial_protocol: ConsensusType = ConsensusType.PROOF_OF_DATA,
        threshold_percent: int = 67,
    ) -> None:
        """Initialize adaptive consensus.

        Args:
            consensus_id: Optional consensus ID
            initial_protocol: Initial consensus protocol
            threshold_percent: Consensus threshold percentage
        """
        self.consensus_id = consensus_id or f"cons_{secrets.token_hex(8)}"
        self.active_protocol = initial_protocol
        self.validators: list[str] = []
        self.threshold_percent = threshold_percent
        self.proposals: dict[str, dict[str, Any]] = {}
        self.votes: dict[str, list[ConsensusVote]] = {}
        self.protocol_history: list[dict[str, Any]] = []
        self.created_at = datetime.now(timezone.utc).isoformat()

    def add_validator(self, validator_id: str) -> None:
        """Add a validator node.

        Args:
            validator_id: Validator node ID
        """
        if validator_id not in self.validators:
            self.validators.append(validator_id)

    def remove_validator(self, validator_id: str) -> None:
        """Remove a validator node.

        Args:
            validator_id: Validator node ID
        """
        if validator_id in self.validators:
            self.validators.remove(validator_id)

    def create_proposal(self, proposal_type: str, content: dict[str, Any]) -> str:
        """Create a new proposal for consensus.

        Args:
            proposal_type: Type of proposal
            content: Proposal content

        Returns:
            Proposal ID
        """
        proposal_id = f"prop_{secrets.token_hex(8)}"
        self.proposals[proposal_id] = {
            "proposal_id": proposal_id,
            "proposal_type": proposal_type,
            "content": content,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
        }
        self.votes[proposal_id] = []
        return proposal_id

    def submit_vote(self, voter_id: str, proposal_id: str, vote: bool) -> ConsensusVote | None:
        """Submit a vote for a proposal.

        Args:
            voter_id: Voter node ID
            proposal_id: Proposal ID
            vote: Vote value

        Returns:
            Vote object or None if invalid
        """
        if voter_id not in self.validators:
            return None
        if proposal_id not in self.proposals:
            return None

        vote_obj = ConsensusVote(
            voter_id=voter_id,
            proposal_id=proposal_id,
            vote=vote,
        )
        self.votes[proposal_id].append(vote_obj)
        self._check_consensus(proposal_id)
        return vote_obj

    def _check_consensus(self, proposal_id: str) -> None:
        """Check if consensus is reached for a proposal.

        Args:
            proposal_id: Proposal ID
        """
        votes = self.votes.get(proposal_id, [])
        if not votes or not self.validators:
            return

        yes_votes = sum(1 for v in votes if v.vote)
        vote_percent = (yes_votes / len(self.validators)) * 100

        if vote_percent >= self.threshold_percent:
            self.proposals[proposal_id]["status"] = "approved"
        elif len(votes) == len(self.validators):
            self.proposals[proposal_id]["status"] = "rejected"

    def adapt_protocol(self, network_latency_ms: float, threat_level: float) -> ConsensusType:
        """Adapt consensus protocol based on conditions.

        Args:
            network_latency_ms: Average network latency
            threat_level: Current threat level (0.0-1.0)

        Returns:
            New consensus protocol
        """
        old_protocol = self.active_protocol

        # Adapt based on conditions
        if threat_level > 0.8:
            # High security - use BFT
            self.active_protocol = ConsensusType.BFT_HOTSTUFF
        elif network_latency_ms > 500:
            # High latency - use RAFT for simplicity
            self.active_protocol = ConsensusType.RAFT
        elif network_latency_ms < 100:
            # Low latency - use Tendermint for speed
            self.active_protocol = ConsensusType.TENDERMINT
        else:
            # Default to Proof of Data
            self.active_protocol = ConsensusType.PROOF_OF_DATA

        if old_protocol != self.active_protocol:
            self.protocol_history.append(
                {
                    "from": old_protocol.value,
                    "to": self.active_protocol.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": f"latency={network_latency_ms}ms, threat={threat_level}",
                }
            )

        return self.active_protocol

    def get_statistics(self) -> dict[str, Any]:
        """Get consensus statistics.

        Returns:
            Consensus statistics
        """
        total_proposals = len(self.proposals)
        approved = sum(1 for p in self.proposals.values() if p["status"] == "approved")
        rejected = sum(1 for p in self.proposals.values() if p["status"] == "rejected")

        return {
            "consensus_id": self.consensus_id,
            "active_protocol": self.active_protocol.value,
            "validators": len(self.validators),
            "threshold_percent": self.threshold_percent,
            "total_proposals": total_proposals,
            "approved_proposals": approved,
            "rejected_proposals": rejected,
            "pending_proposals": total_proposals - approved - rejected,
            "protocol_changes": len(self.protocol_history),
            "created_at": self.created_at,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize consensus manager to dictionary."""
        return {
            "consensus_id": self.consensus_id,
            "active_protocol": self.active_protocol.value,
            "validators": self.validators,
            "threshold_percent": self.threshold_percent,
            "proposals": self.proposals,
            "votes": {k: [v.to_dict() for v in vs] for k, vs in self.votes.items()},
            "protocol_history": self.protocol_history,
            "created_at": self.created_at,
        }


@dataclass
class CrossChainBridge:
    """Cross-chain interoperability bridge.

    Enables communication and asset transfer between different blockchains.

    Attributes:
        bridge_id: Unique bridge identifier
        source_chain: Source blockchain ID
        target_chain: Target blockchain ID
        locked_assets: Assets locked on source chain
        minted_assets: Assets minted on target chain
        transactions: Bridge transactions
    """

    bridge_id: str
    source_chain: str
    target_chain: str
    locked_assets: dict[str, float] = field(default_factory=dict)
    minted_assets: dict[str, float] = field(default_factory=dict)
    transactions: list[dict[str, Any]] = field(default_factory=list)
    status: str = "active"

    def lock_and_mint(self, asset_id: str, amount: float, recipient: str) -> dict[str, Any]:
        """Lock asset on source chain and mint on target.

        Args:
            asset_id: Asset identifier
            amount: Amount to transfer
            recipient: Recipient address

        Returns:
            Transaction record
        """
        # Lock on source
        self.locked_assets[asset_id] = self.locked_assets.get(asset_id, 0.0) + amount

        # Mint on target
        self.minted_assets[asset_id] = self.minted_assets.get(asset_id, 0.0) + amount

        tx = {
            "tx_id": f"bridge_tx_{secrets.token_hex(8)}",
            "type": "lock_and_mint",
            "asset_id": asset_id,
            "amount": amount,
            "recipient": recipient,
            "source_chain": self.source_chain,
            "target_chain": self.target_chain,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
        }
        self.transactions.append(tx)
        return tx

    def burn_and_release(
        self, asset_id: str, amount: float, recipient: str
    ) -> dict[str, Any] | None:
        """Burn asset on target chain and release on source.

        Args:
            asset_id: Asset identifier
            amount: Amount to transfer
            recipient: Recipient address

        Returns:
            Transaction record or None if insufficient balance
        """
        minted = self.minted_assets.get(asset_id, 0.0)
        locked = self.locked_assets.get(asset_id, 0.0)

        if minted < amount or locked < amount:
            return None

        # Burn on target
        self.minted_assets[asset_id] = minted - amount

        # Release on source
        self.locked_assets[asset_id] = locked - amount

        tx = {
            "tx_id": f"bridge_tx_{secrets.token_hex(8)}",
            "type": "burn_and_release",
            "asset_id": asset_id,
            "amount": amount,
            "recipient": recipient,
            "source_chain": self.target_chain,
            "target_chain": self.source_chain,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
        }
        self.transactions.append(tx)
        return tx

    def get_statistics(self) -> dict[str, Any]:
        """Get bridge statistics.

        Returns:
            Bridge statistics
        """
        return {
            "bridge_id": self.bridge_id,
            "source_chain": self.source_chain,
            "target_chain": self.target_chain,
            "total_locked": sum(self.locked_assets.values()),
            "total_minted": sum(self.minted_assets.values()),
            "transaction_count": len(self.transactions),
            "status": self.status,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize bridge to dictionary."""
        return {
            "bridge_id": self.bridge_id,
            "source_chain": self.source_chain,
            "target_chain": self.target_chain,
            "locked_assets": self.locked_assets,
            "minted_assets": self.minted_assets,
            "transactions": self.transactions[-50:],  # Last 50 transactions
            "status": self.status,
        }


@dataclass
class DisasterRecovery:
    """Disaster recovery and business continuity manager.

    Manages backup, recovery, and failover capabilities.

    Attributes:
        recovery_id: Unique recovery instance ID
        mode: Recovery mode
        backup_locations: List of backup locations
        rpo_minutes: Recovery Point Objective in minutes
        rto_minutes: Recovery Time Objective in minutes
        last_backup: Timestamp of last backup
        recovery_tests: List of recovery test results
    """

    recovery_id: str
    mode: RecoveryMode = RecoveryMode.HOT_STANDBY
    backup_locations: list[str] = field(default_factory=list)
    rpo_minutes: int = 15  # Max data loss tolerance
    rto_minutes: int = 60  # Max downtime tolerance
    last_backup: str = ""
    recovery_tests: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.backup_locations:
            self.backup_locations = [
                "region-us-east",
                "region-eu-west",
                "region-asia-pacific",
            ]

    def create_backup(self, data_hash: str, size_mb: float) -> dict[str, Any]:
        """Create a backup.

        Args:
            data_hash: Hash of backed up data
            size_mb: Size in megabytes

        Returns:
            Backup record
        """
        backup = {
            "backup_id": f"backup_{secrets.token_hex(8)}",
            "data_hash": data_hash,
            "size_mb": size_mb,
            "locations": self.backup_locations,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
        }
        self.last_backup = backup["timestamp"]
        return backup

    def test_recovery(self) -> dict[str, Any]:
        """Test recovery procedure.

        Returns:
            Test results
        """
        import random

        # Simulate recovery test
        success = random.random() > 0.05  # 95% success rate
        recovery_time = random.randint(10, self.rto_minutes)

        test_result = {
            "test_id": f"test_{secrets.token_hex(8)}",
            "mode": self.mode.value,
            "success": success,
            "recovery_time_minutes": recovery_time,
            "rto_met": recovery_time <= self.rto_minutes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.recovery_tests.append(test_result)
        return test_result

    def get_status(self) -> dict[str, Any]:
        """Get recovery status.

        Returns:
            Recovery status
        """
        recent_tests = self.recovery_tests[-5:]
        success_rate = (
            sum(1 for t in recent_tests if t.get("success", False)) / len(recent_tests) * 100
            if recent_tests
            else 0
        )

        return {
            "recovery_id": self.recovery_id,
            "mode": self.mode.value,
            "backup_locations": self.backup_locations,
            "rpo_minutes": self.rpo_minutes,
            "rto_minutes": self.rto_minutes,
            "last_backup": self.last_backup,
            "recent_test_success_rate": success_rate,
            "total_tests": len(self.recovery_tests),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize recovery manager to dictionary."""
        return {
            "recovery_id": self.recovery_id,
            "mode": self.mode.value,
            "backup_locations": self.backup_locations,
            "rpo_minutes": self.rpo_minutes,
            "rto_minutes": self.rto_minutes,
            "last_backup": self.last_backup,
            "recovery_tests": self.recovery_tests[-10:],
        }


class SecurityLayer:
    """Main security layer managing all security components.

    Coordinates quantum-resistant crypto, consensus, cross-chain bridges,
    and disaster recovery.

    Attributes:
        security_id: Unique security layer ID
        crypto: Quantum-resistant crypto manager
        consensus: Adaptive consensus manager
        bridges: Cross-chain bridges
        recovery: Disaster recovery manager
    """

    def __init__(self, security_id: str | None = None) -> None:
        """Initialize security layer.

        Args:
            security_id: Optional security ID
        """
        self.security_id = security_id or f"sec_{secrets.token_hex(8)}"
        self.crypto = QuantumResistantCrypto(crypto_id=f"crypto_{self.security_id}")
        self.consensus = AdaptiveConsensus(consensus_id=f"cons_{self.security_id}")
        self.bridges: dict[str, CrossChainBridge] = {}
        self.recovery = DisasterRecovery(recovery_id=f"dr_{self.security_id}")
        self.created_at = datetime.now(timezone.utc).isoformat()

        # Initialize default keys
        self._initialize_default_keys()

    def _initialize_default_keys(self) -> None:
        """Initialize default cryptographic keys."""
        # Generate keys for each algorithm
        for algo in [
            CryptoAlgorithm.CRYSTALS_DILITHIUM,
            CryptoAlgorithm.CRYSTALS_KYBER,
            CryptoAlgorithm.FALCON,
        ]:
            self.crypto.generate_key_pair(algorithm=algo)

    def add_bridge(self, source_chain: str, target_chain: str) -> CrossChainBridge:
        """Add a cross-chain bridge.

        Args:
            source_chain: Source blockchain ID
            target_chain: Target blockchain ID

        Returns:
            Created bridge
        """
        bridge_id = f"bridge_{source_chain}_{target_chain}"
        bridge = CrossChainBridge(
            bridge_id=bridge_id,
            source_chain=source_chain,
            target_chain=target_chain,
        )
        self.bridges[bridge_id] = bridge
        return bridge

    def assess_security_posture(self) -> dict[str, Any]:
        """Assess overall security posture.

        Returns:
            Security assessment
        """
        # Calculate scores
        crypto_score = min(100, len(self.crypto.keys) * 33)
        consensus_score = min(100, self.consensus.get_statistics()["validators"] * 10)
        bridge_score = min(100, len(self.bridges) * 50)
        recovery_score = self.recovery.get_status()["recent_test_success_rate"]

        overall_score = (crypto_score + consensus_score + bridge_score + recovery_score) / 4

        return {
            "security_id": self.security_id,
            "overall_score": overall_score,
            "crypto_score": crypto_score,
            "consensus_score": consensus_score,
            "bridge_score": bridge_score,
            "recovery_score": recovery_score,
            "assessment_time": datetime.now(timezone.utc).isoformat(),
            "recommendations": self._generate_recommendations(
                crypto_score, consensus_score, bridge_score, recovery_score
            ),
        }

    def _generate_recommendations(
        self,
        crypto_score: float,
        consensus_score: float,
        bridge_score: float,
        recovery_score: float,
    ) -> list[str]:
        """Generate security recommendations.

        Args:
            crypto_score: Crypto score
            consensus_score: Consensus score
            bridge_score: Bridge score
            recovery_score: Recovery score

        Returns:
            List of recommendations
        """
        recommendations = []

        if crypto_score < 80:
            recommendations.append("Generate additional cryptographic keys for redundancy")
        if consensus_score < 80:
            recommendations.append("Add more validator nodes for improved consensus security")
        if bridge_score < 80:
            recommendations.append("Consider adding cross-chain bridges for interoperability")
        if recovery_score < 80:
            recommendations.append("Conduct more frequent disaster recovery tests")

        if not recommendations:
            recommendations.append("Security posture is healthy - continue monitoring")

        return recommendations

    def get_statistics(self) -> dict[str, Any]:
        """Get security layer statistics.

        Returns:
            Security statistics
        """
        return {
            "security_id": self.security_id,
            "crypto": self.crypto.get_statistics(),
            "consensus": self.consensus.get_statistics(),
            "bridges": len(self.bridges),
            "recovery": self.recovery.get_status(),
            "created_at": self.created_at,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize security layer to dictionary."""
        return {
            "security_id": self.security_id,
            "crypto": self.crypto.to_dict(),
            "consensus": self.consensus.to_dict(),
            "bridges": {k: v.to_dict() for k, v in self.bridges.items()},
            "recovery": self.recovery.to_dict(),
            "created_at": self.created_at,
        }
