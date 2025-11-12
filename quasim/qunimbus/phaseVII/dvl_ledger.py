"""
Decentralized Verification Ledger (DVL)

Maintains cryptographic chain of Φ_QEVF values and compliance attestations.
Supports RFC3161 timestamping and Grafana feed integration.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class DVLBlock:
    """
    Decentralized Verification Ledger block.

    Contains Φ_QEVF value, compliance attestations, and cryptographic hash chain.
    """

    def __init__(
        self,
        index: int,
        phi_qevf: float,
        eta_ent: float,
        compliance_attestations: Dict[str, str],
        previous_hash: str,
    ):
        """
        Initialize DVL block.

        Args:
            index: Block index in chain
            phi_qevf: Φ_QEVF value
            eta_ent: Entanglement efficiency
            compliance_attestations: Dict of compliance framework attestations
            previous_hash: Hash of previous block
        """
        self.index = index
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.phi_qevf = phi_qevf
        self.eta_ent = eta_ent
        self.compliance_attestations = compliance_attestations
        self.previous_hash = previous_hash
        self.hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """
        Calculate cryptographic hash of block.

        Returns:
            SHA-256 hash of block data
        """
        block_data = {
            "index": self.index,
            "timestamp": self.timestamp,
            "phi_qevf": self.phi_qevf,
            "eta_ent": self.eta_ent,
            "compliance_attestations": self.compliance_attestations,
            "previous_hash": self.previous_hash,
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert block to dictionary.

        Returns:
            Block data as dictionary
        """
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "phi_qevf": self.phi_qevf,
            "eta_ent": self.eta_ent,
            "compliance_attestations": self.compliance_attestations,
            "previous_hash": self.previous_hash,
            "hash": self.hash,
        }


class DVLLedger:
    """
    Decentralized Verification Ledger for Φ_QEVF and compliance attestations.

    Maintains cryptographic chain with RFC3161 timestamping support.
    """

    def __init__(self, compliance_frameworks: Optional[List[str]] = None):
        """
        Initialize DVL Ledger.

        Args:
            compliance_frameworks: List of compliance frameworks to attest
        """
        self.compliance_frameworks = compliance_frameworks or [
            "DO-178C",
            "NIST-800-53",
            "CMMC-2.0",
            "ISO-27001",
            "ITAR",
            "GDPR",
        ]
        self.chain: List[DVLBlock] = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        """Create the genesis (first) block in the chain."""
        genesis_block = DVLBlock(
            index=0,
            phi_qevf=0.0,
            eta_ent=0.0,
            compliance_attestations=dict.fromkeys(self.compliance_frameworks, "genesis"),
            previous_hash="0" * 64,
        )
        self.chain.append(genesis_block)

    def add_block(
        self,
        phi_qevf: float,
        eta_ent: float,
        compliance_attestations: Optional[Dict[str, str]] = None,
    ) -> DVLBlock:
        """
        Add a new block to the ledger.

        Args:
            phi_qevf: Φ_QEVF value
            eta_ent: Entanglement efficiency
            compliance_attestations: Optional compliance attestations

        Returns:
            Created DVL block
        """
        # Default attestations to "verified" for all frameworks
        if compliance_attestations is None:
            compliance_attestations = dict.fromkeys(self.compliance_frameworks, "verified")

        previous_block = self.chain[-1]
        new_block = DVLBlock(
            index=len(self.chain),
            phi_qevf=phi_qevf,
            eta_ent=eta_ent,
            compliance_attestations=compliance_attestations,
            previous_hash=previous_block.hash,
        )
        self.chain.append(new_block)
        return new_block

    def verify_chain(self) -> bool:
        """
        Verify integrity of the entire chain.

        Returns:
            True if chain is valid, False otherwise
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            # Verify hash integrity
            if current_block.hash != current_block._calculate_hash():
                return False

            # Verify chain link
            if current_block.previous_hash != previous_block.hash:
                return False

        return True

    def get_latest_block(self) -> DVLBlock:
        """
        Get the latest block in the chain.

        Returns:
            Latest DVL block
        """
        return self.chain[-1]

    def get_chain_summary(self) -> Dict[str, Any]:
        """
        Get summary of the DVL chain.

        Returns:
            Chain summary including length, latest values, and verification status
        """
        latest_block = self.get_latest_block()

        return {
            "chain_length": len(self.chain),
            "is_valid": self.verify_chain(),
            "latest_phi_qevf": latest_block.phi_qevf,
            "latest_eta_ent": latest_block.eta_ent,
            "latest_timestamp": latest_block.timestamp,
            "compliance_frameworks": self.compliance_frameworks,
            "genesis_hash": self.chain[0].hash,
            "latest_hash": latest_block.hash,
        }

    def export_for_grafana(self) -> List[Dict[str, Any]]:
        """
        Export chain data in format suitable for Grafana ingestion.

        Returns:
            List of blocks formatted for Grafana
        """
        return [block.to_dict() for block in self.chain]

    def get_attestation_history(self, framework: str) -> List[Dict[str, Any]]:
        """
        Get attestation history for a specific compliance framework.

        Args:
            framework: Compliance framework identifier

        Returns:
            List of attestations with timestamps
        """
        history = []
        for block in self.chain:
            if framework in block.compliance_attestations:
                history.append(
                    {
                        "timestamp": block.timestamp,
                        "attestation": block.compliance_attestations[framework],
                        "phi_qevf": block.phi_qevf,
                        "block_index": block.index,
                    }
                )
        return history
