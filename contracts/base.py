"""Base Contract System - Immutable, Hash-Addressed Contracts.

This module provides the base contract class for all QRATUM contracts,
ensuring immutability, hash-addressing, and deterministic serialization.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class BaseContract:
    """Immutable base contract with hash-addressing.

    All contracts in QRATUM are immutable, hash-addressed, and deterministic.
    This is enforced through frozen=True dataclass and deterministic serialization.

    Attributes:
        contract_id: Unique contract identifier (SHA-256 hash)
        contract_type: Type of contract (e.g., "IntentContract")
        created_at: ISO 8601 timestamp of contract creation
        version: Contract schema version
        metadata: Additional contract metadata
    """

    contract_id: str
    contract_type: str
    created_at: str
    version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate contract after initialization."""
        if not self.contract_id:
            raise ValueError("contract_id cannot be empty")
        if not self.contract_type:
            raise ValueError("contract_type cannot be empty")
        if not self.created_at:
            raise ValueError("created_at cannot be empty")

        # Validate ISO 8601 timestamp
        try:
            datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid ISO 8601 timestamp: {self.created_at}") from e

    def serialize(self) -> dict[str, Any]:
        """Serialize contract to deterministic dictionary.

        Returns:
            Dictionary representation with sorted keys
        """
        return {
            "contract_id": self.contract_id,
            "contract_type": self.contract_type,
            "created_at": self.created_at,
            "version": self.version,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 0) -> str:
        """Serialize contract to deterministic JSON.

        Args:
            indent: JSON indentation level (0 for compact)

        Returns:
            JSON string with sorted keys
        """
        serialized = self.serialize()
        if indent > 0:
            return json.dumps(serialized, sort_keys=True, indent=indent)
        return json.dumps(serialized, sort_keys=True)

    def verify_hash(self) -> bool:
        """Verify contract_id matches content hash.

        Returns:
            True if hash is valid, False otherwise
        """
        # Compute hash of content (excluding contract_id itself)
        content = self.serialize()
        content_without_id = {k: v for k, v in content.items() if k != "contract_id"}
        json_str = json.dumps(content_without_id, sort_keys=True)
        computed_hash = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
        return computed_hash == self.contract_id


def compute_contract_hash(content: dict[str, Any]) -> str:
    """Compute SHA-256 hash of contract content.

    Args:
        content: Contract content dictionary (without contract_id)

    Returns:
        Hexadecimal SHA-256 hash string
    """
    json_str = json.dumps(content, sort_keys=True)
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def generate_contract_id(contract_type: str, content: dict[str, Any]) -> str:
    """Generate deterministic contract ID from content.

    Args:
        contract_type: Type of contract
        content: Contract-specific content

    Returns:
        SHA-256 hash as contract ID
    """
    # Include contract type in hash
    hash_content = {
        "contract_type": contract_type,
        **content,
    }
    return compute_contract_hash(hash_content)


def get_current_timestamp() -> str:
    """Get current timestamp in ISO 8601 format.

    Returns:
        ISO 8601 timestamp string with 'Z' suffix
    """
    from datetime import timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
