"""Cryptographic utilities for QRATUM platform."""

import hashlib
import json
from typing import Any, Dict


def compute_deterministic_hash(data: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of data.

    Args:
        data: Dictionary to hash

    Returns:
        Hex-encoded SHA-256 hash
    """
    # Use sorted keys and compact separators for deterministic JSON
    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode()).hexdigest()


def compute_deterministic_seed(data: Dict[str, Any]) -> int:
    """Compute deterministic integer seed from data for random number generation.

    Args:
        data: Dictionary to hash

    Returns:
        Integer seed derived from hash
    """
    hash_hex = compute_deterministic_hash(data)
    # Convert first 8 hex chars to int (32 bits)
    return int(hash_hex[:8], 16)


def compute_deterministic_float(data: str) -> float:
    """Compute deterministic float value from string using SHA-256.

    Args:
        data: String to hash

    Returns:
        Float value between 0 and 10
    """
    hash_hex = hashlib.sha256(data.encode()).hexdigest()
    # Convert first 8 hex chars to int and scale to 0-10 range
    return (int(hash_hex[:8], 16) % 1000) / 100.0


def verify_hash_chain(
    events: list, get_hash_func: callable, get_prev_func: callable
) -> bool:
    """Verify integrity of a hash chain.

    Args:
        events: List of events in chain
        get_hash_func: Function to get hash from event
        get_prev_func: Function to get previous hash from event

    Returns:
        True if chain is valid, False otherwise
    """
    if not events:
        return True

    prev_hash = None
    for event in events:
        if get_prev_func(event) != prev_hash:
            return False
        prev_hash = get_hash_func(event)

    return True
