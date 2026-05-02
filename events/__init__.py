"""Events - Event & Causality Fabric for QRATUM.

This module provides the global append-only event log with causal chains
for deterministic execution tracking and audit.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from events.log import Event, EventLog, get_global_event_log, log_event

__all__ = [
    "Event",
    "EventLog",
    "get_global_event_log",
    "log_event",
]

__version__ = "1.0.0"
