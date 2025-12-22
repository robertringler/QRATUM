"""Event Contract - Expected Event Sequence and Causality.

This module implements the EventContract for defining expected event sequences
and causal relationships in execution.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from contracts.base import (
    BaseContract,
    generate_contract_id,
    get_current_timestamp,
)


@dataclass(frozen=True)
class EventContract(BaseContract):
    """Immutable contract defining expected event sequence.

    Attributes:
        intent_contract_id: Reference to IntentContract
        expected_events: List of expected event types in order
        causal_chains: List of causal relationships between events
        event_constraints: Constraints on event properties
        event_proof: Proof of event authorization
    """

    intent_contract_id: str = ""
    expected_events: list[str] = field(default_factory=list)
    causal_chains: list[dict[str, Any]] = field(default_factory=list)
    event_constraints: dict[str, Any] = field(default_factory=dict)
    event_proof: str = ""

    def __post_init__(self) -> None:
        """Validate event contract after initialization."""
        super().__post_init__()
        if not self.intent_contract_id:
            raise ValueError("intent_contract_id cannot be empty")
        if not self.expected_events:
            raise ValueError("expected_events cannot be empty")
        if not self.event_proof:
            raise ValueError("event_proof cannot be empty")

    def serialize(self) -> dict[str, Any]:
        """Serialize event contract to dictionary."""
        base = super().serialize()
        base.update(
            {
                "intent_contract_id": self.intent_contract_id,
                "expected_events": self.expected_events,
                "causal_chains": self.causal_chains,
                "event_constraints": self.event_constraints,
                "event_proof": self.event_proof,
            }
        )
        return base

    def has_event(self, event_type: str) -> bool:
        """Check if event type is in expected sequence.

        Args:
            event_type: Event type to check

        Returns:
            True if event is expected, False otherwise
        """
        return event_type in self.expected_events

    def get_event_index(self, event_type: str) -> int:
        """Get index of event type in expected sequence.

        Args:
            event_type: Event type to find

        Returns:
            Index of event, or -1 if not found
        """
        try:
            return self.expected_events.index(event_type)
        except ValueError:
            return -1

    def is_ordered_correctly(self, event_sequence: list[str]) -> bool:
        """Check if event sequence matches expected order.

        Args:
            event_sequence: Observed event sequence

        Returns:
            True if order is correct, False otherwise
        """
        # Check if all observed events are expected
        for event in event_sequence:
            if event not in self.expected_events:
                return False

        # Check if observed events maintain expected order
        last_index = -1
        for event in event_sequence:
            index = self.get_event_index(event)
            if index <= last_index:
                return False
            last_index = index

        return True

    def get_next_expected_event(self, observed_events: list[str]) -> str | None:
        """Get next expected event given observed sequence.

        Args:
            observed_events: List of observed event types

        Returns:
            Next expected event type, or None if sequence complete
        """
        if not observed_events:
            return self.expected_events[0] if self.expected_events else None

        # Find last observed event index
        last_observed = observed_events[-1]
        last_index = self.get_event_index(last_observed)

        if last_index == -1 or last_index >= len(self.expected_events) - 1:
            return None

        return self.expected_events[last_index + 1]


def create_event_contract(
    intent_contract_id: str,
    expected_events: list[str],
    causal_chains: list[dict[str, Any]] | None = None,
    event_constraints: dict[str, Any] | None = None,
    event_proof: str = "event_authorized",
) -> EventContract:
    """Create an EventContract for event sequence definition.

    Args:
        intent_contract_id: Reference to IntentContract
        expected_events: List of expected event types in order
        causal_chains: Optional list of causal relationships
        event_constraints: Optional constraints on event properties
        event_proof: Proof of event authorization

    Returns:
        Immutable EventContract
    """
    content = {
        "intent_contract_id": intent_contract_id,
        "expected_events": expected_events,
        "causal_chains": causal_chains or [],
        "event_constraints": event_constraints or {},
        "event_proof": event_proof,
        "created_at": get_current_timestamp(),
        "version": "1.0.0",
    }

    contract_id = generate_contract_id("EventContract", content)

    return EventContract(
        contract_id=contract_id,
        contract_type="EventContract",
        created_at=content["created_at"],
        version=content["version"],
        intent_contract_id=content["intent_contract_id"],
        expected_events=content["expected_events"],
        causal_chains=content["causal_chains"],
        event_constraints=content["event_constraints"],
        event_proof=content["event_proof"],
    )
