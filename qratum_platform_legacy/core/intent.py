"""Platform Intent and Contract System.

Immutable intent and contract structures for deterministic execution
authorization across all vertical modules.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional


class VerticalType(Enum):
    """Enumeration of supported vertical modules."""

    JURIS = "juris"  # Legal AI
    VITRA = "vitra"  # Bioinformatics & Drug Discovery
    ECORA = "ecora"  # Climate & Energy Systems
    CAPRA = "capra"  # Financial Risk & Derivatives
    SENTRA = "sentra"  # Aerospace & Defense
    NEURA = "neura"  # Neuroscience & BCI
    FLUXA = "fluxa"  # Supply Chain & Logistics


@dataclass(frozen=True)
class PlatformIntent:
    """Universal intent structure for all vertical modules.

    Attributes:
        vertical: Target vertical module
        operation: Specific operation within the vertical
        parameters: Immutable operation parameters
        compliance_attestations: Required compliance confirmations
        user_id: User identifier for audit trail
        session_id: Session identifier for grouping operations
        timestamp: Intent creation timestamp (UTC)
    """

    vertical: VerticalType
    operation: str
    parameters: Dict[str, Any]
    compliance_attestations: FrozenSet[str] = field(default_factory=frozenset)
    user_id: str = "system"
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate intent after initialization."""
        if not self.operation:
            raise ValueError("Operation cannot be empty")
        if not isinstance(self.parameters, dict):
            raise ValueError("Parameters must be a dictionary")


@dataclass(frozen=True)
class PlatformContract:
    """Immutable execution authorization contract.

    Binds intent to execution context with cryptographic commitment.

    Attributes:
        intent: The original platform intent
        contract_hash: SHA-256 hash of intent serialization
        authorized: Whether execution is authorized
        authorization_timestamp: When authorization was granted
        expiry_timestamp: When authorization expires
        restrictions: Any restrictions on execution
    """

    intent: PlatformIntent
    contract_hash: str
    authorized: bool = False
    authorization_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expiry_timestamp: Optional[datetime] = None
    restrictions: FrozenSet[str] = field(default_factory=frozenset)

    def is_valid(self) -> bool:
        """Check if contract is currently valid.

        Returns:
            True if contract is authorized and not expired
        """
        if not self.authorized:
            return False
        if self.expiry_timestamp is not None:
            if datetime.now(timezone.utc) > self.expiry_timestamp:
                return False
        return True

    def has_attestation(self, attestation: str) -> bool:
        """Check if required attestation is present.

        Args:
            attestation: Attestation to check for

        Returns:
            True if attestation is present in intent
        """
        return attestation in self.intent.compliance_attestations
