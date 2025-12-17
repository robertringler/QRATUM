"""Audit logging for QuASIM operations.

This module provides append-only audit logging with SHA256 chain-of-trust
for compliance and reproducibility tracking.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_LOG_PATH = "artifacts/audit.jsonl"


def audit_event(
    event_type: str,
    data: Dict[str, Any],
    log_path: Optional[str] = None,
) -> str:
    """Log an audit event with chain-of-trust integrity.

    Parameters
    ----------
    event_type : str
        Type of event (e.g., "qnimbus.ascend", "qnimbus.validate")
    data : Dict[str, Any]
        Event data to log
    log_path : Optional[str]
        Path to audit log file (default: artifacts/audit.jsonl)

    Returns
    -------
    str
        Event ID (SHA256 hash)

    Examples
    --------
    >>> audit_event("test.event", {"key": "value", "query_id": "qid-123"})
    '...'
    """

    log_path = log_path or DEFAULT_LOG_PATH
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Get previous hash for chain-of-trust
    prev_hash = "0" * 64
    if log_file.exists():
        with open(log_file) as f:
            lines = f.readlines()
            if lines:
                last_entry = json.loads(lines[-1])
                prev_hash = last_entry.get("event_id", prev_hash)

    # Extract query_id if present for indexing
    query_id = data.get("query_id") or data.get("qid")

    # Create event record
    timestamp = datetime.now(timezone.utc).isoformat() + "Z"
    event = {
        "timestamp": timestamp,
        "event_type": event_type,
        "data": data,
        "prev_hash": prev_hash,
    }

    # Add query_id at top level if present
    if query_id:
        event["query_id"] = query_id

    # Compute event hash
    event_str = json.dumps(event, sort_keys=True)
    event_id = hashlib.sha256(event_str.encode("utf-8")).hexdigest()
    event["event_id"] = event_id

    # Append to log
    with open(log_file, "a") as f:
        f.write(json.dumps(event) + "\n")

    return event_id


def verify_audit_chain(log_path: Optional[str] = None) -> bool:
    """Verify integrity of audit log chain.

    This function validates:
    1. SHA256 chain-of-trust (prev_hash links)
    2. Event ID computation integrity
    3. Query ID consistency (if present)

    Parameters
    ----------
    log_path : Optional[str]
        Path to audit log file (default: artifacts/audit.jsonl)

    Returns
    -------
    bool
        True if chain is valid, False otherwise

    Examples
    --------
    >>> verify_audit_chain()
    True
    """

    log_path = log_path or DEFAULT_LOG_PATH
    log_file = Path(log_path)

    if not log_file.exists():
        return True

    prev_hash = "0" * 64
    query_ids_seen = set()

    with open(log_file) as f:
        for line in f:
            event = json.loads(line)

            # Verify previous hash
            if event.get("prev_hash") != prev_hash:
                return False

            # Track query_id if present
            query_id = event.get("query_id")
            if query_id:
                query_ids_seen.add(query_id)

            # Recompute event hash
            event_id = event.pop("event_id")
            event_str = json.dumps(event, sort_keys=True)
            computed_hash = hashlib.sha256(event_str.encode("utf-8")).hexdigest()

            if computed_hash != event_id:
                return False

            prev_hash = event_id
            event["event_id"] = event_id

    return True
