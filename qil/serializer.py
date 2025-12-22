"""QIL Deterministic Serialization.

This module provides deterministic serialization for QIL Intent objects,
ensuring consistent hashing and reproducibility.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from qil.ast import Intent


def serialize_intent(intent: Intent) -> dict[str, Any]:
    """Serialize intent to deterministic dictionary.

    Args:
        intent: Intent to serialize

    Returns:
        Dictionary representation with sorted keys
    """
    return intent.serialize()


def to_json(intent: Intent, indent: int = 0) -> str:
    """Serialize intent to deterministic JSON string.

    Args:
        intent: Intent to serialize
        indent: JSON indentation level (0 for compact)

    Returns:
        JSON string with sorted keys
    """
    serialized = serialize_intent(intent)
    if indent > 0:
        return json.dumps(serialized, sort_keys=True, indent=indent)
    return json.dumps(serialized, sort_keys=True)


def compute_hash(intent: Intent) -> str:
    """Compute SHA-256 hash of intent.

    Args:
        intent: Intent to hash

    Returns:
        Hexadecimal SHA-256 hash string
    """
    json_str = to_json(intent)
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def intent_to_canonical_form(intent: Intent) -> str:
    """Convert intent to canonical QIL form.

    Regenerates QIL source code from parsed intent in canonical format.

    Args:
        intent: Intent to convert

    Returns:
        Canonical QIL source code
    """
    lines = [f"INTENT {intent.name} {{"]

    # Objective (required)
    lines.append(f"    OBJECTIVE {intent.objective.name}")

    # Hardware (if specified)
    if intent.hardware:
        hw_parts = ["    HARDWARE"]
        if intent.hardware.only_clusters:
            hw_parts.append("ONLY " + " AND ".join(intent.hardware.only_clusters))
        if intent.hardware.not_clusters:
            hw_parts.append("NOT " + " AND ".join(intent.hardware.not_clusters))
        lines.append(" ".join(hw_parts))

    # Constraints
    for constraint in intent.constraints:
        value_str = f'"{constraint.value}"' if isinstance(constraint.value, str) else str(
            constraint.value
        )
        lines.append(f"    CONSTRAINT {constraint.name} {constraint.operator} {value_str}")

    # Capabilities
    for capability in intent.capabilities:
        lines.append(f"    CAPABILITY {capability.name}")

    # Time specifications
    for time_spec in intent.time_specs:
        # Format value to avoid unnecessary .0 for integers
        value_str = str(int(time_spec.value)) if time_spec.value == int(time_spec.value) else str(time_spec.value)
        lines.append(f"    TIME {time_spec.key}: {value_str}{time_spec.unit}")

    # Authorities
    for authority in intent.authorities:
        lines.append(f"    AUTHORITY {authority.key}: {authority.value}")

    # Trust
    if intent.trust:
        lines.append(f"    TRUST level: {intent.trust.level}")

    lines.append("}")
    return "\n".join(lines)
