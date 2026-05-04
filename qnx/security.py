from __future__ import annotations

import hashlib
import hmac
from typing import Any

from .types import SecurityLevel


def validate_security_context(level: SecurityLevel) -> None:
    """Placeholder for security hooks.

    The function currently performs no enforcement but provides a place to add
    validation logic when integrating with restricted environments.
    """


def compute_integrity_hash(data: Any) -> str:
    """Compute a deterministic integrity hash for the provided data."""

    serialized = str(data).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def verify_integrity(data: Any, expected_hash: str) -> bool:
    """Verify that the computed hash matches the expected value."""

    actual = compute_integrity_hash(data)
    return hmac.compare_digest(actual, expected_hash)
