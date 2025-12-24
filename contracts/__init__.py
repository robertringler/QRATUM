"""Contracts - Immutable Contract System for QRATUM.

This module provides immutable, hash-addressed contracts for intent authorization,
capability binding, temporal management, and event sequencing.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from contracts.base import (BaseContract, compute_contract_hash,
                            generate_contract_id, get_current_timestamp)
from contracts.capability import (CapabilityContract, ClusterTopology,
                                  create_capability_contract)
from contracts.event import EventContract, create_event_contract
from contracts.intent import IntentContract, create_intent_contract
from contracts.temporal import TemporalContract, create_temporal_contract

__all__ = [
    # Base
    "BaseContract",
    "compute_contract_hash",
    "generate_contract_id",
    "get_current_timestamp",
    # Intent
    "IntentContract",
    "create_intent_contract",
    # Capability
    "CapabilityContract",
    "ClusterTopology",
    "create_capability_contract",
    # Temporal
    "TemporalContract",
    "create_temporal_contract",
    # Event
    "EventContract",
    "create_event_contract",
]

__version__ = "1.0.0"
