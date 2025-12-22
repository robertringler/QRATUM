"""Temporal Contract - Time Budgets, Deadlines, and Rollback Authorization.

This module implements the TemporalContract for explicit time resource management.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from contracts.base import (
    BaseContract,
    generate_contract_id,
    get_current_timestamp,
)


@dataclass(frozen=True)
class TemporalContract(BaseContract):
    """Immutable contract for time resource management.

    Attributes:
        intent_contract_id: Reference to IntentContract
        deadline_seconds: Deadline in seconds from contract creation
        budget_seconds: Time budget in seconds
        window_start: Optional window start timestamp
        window_end: Optional window end timestamp
        rollback_authorized: Whether rollback is authorized on timeout
        temporal_proof: Proof of temporal authorization
    """

    intent_contract_id: str = ""
    deadline_seconds: float = 0.0
    budget_seconds: float = 0.0
    window_start: str = ""
    window_end: str = ""
    rollback_authorized: bool = False
    temporal_proof: str = ""

    def __post_init__(self) -> None:
        """Validate temporal contract after initialization."""
        super().__post_init__()
        if not self.intent_contract_id:
            raise ValueError("intent_contract_id cannot be empty")
        if self.deadline_seconds <= 0 and self.budget_seconds <= 0:
            raise ValueError("Either deadline_seconds or budget_seconds must be positive")
        if not self.temporal_proof:
            raise ValueError("temporal_proof cannot be empty")

        # Validate window timestamps if provided
        if self.window_start:
            try:
                datetime.fromisoformat(self.window_start.replace("Z", "+00:00"))
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Invalid window_start timestamp: {self.window_start}") from e

        if self.window_end:
            try:
                datetime.fromisoformat(self.window_end.replace("Z", "+00:00"))
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Invalid window_end timestamp: {self.window_end}") from e

    def serialize(self) -> dict[str, Any]:
        """Serialize temporal contract to dictionary."""
        base = super().serialize()
        base.update(
            {
                "intent_contract_id": self.intent_contract_id,
                "deadline_seconds": self.deadline_seconds,
                "budget_seconds": self.budget_seconds,
                "window_start": self.window_start,
                "window_end": self.window_end,
                "rollback_authorized": self.rollback_authorized,
                "temporal_proof": self.temporal_proof,
            }
        )
        return base

    def get_absolute_deadline(self) -> datetime:
        """Calculate absolute deadline timestamp.

        Returns:
            Absolute deadline as datetime
        """
        created = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        return created + timedelta(seconds=self.deadline_seconds)

    def is_expired(self, current_time: datetime | None = None) -> bool:
        """Check if contract deadline has expired.

        Args:
            current_time: Optional current time (defaults to now)

        Returns:
            True if expired, False otherwise
        """
        if self.deadline_seconds <= 0:
            return False

        if current_time is None:
            from datetime import timezone
            current_time = datetime.now(timezone.utc)

        # Ensure current_time is timezone-aware
        if current_time.tzinfo is None:
            from datetime import timezone
            current_time = current_time.replace(tzinfo=timezone.utc)

        deadline = self.get_absolute_deadline()
        return current_time > deadline

    def time_remaining_seconds(self, current_time: datetime | None = None) -> float:
        """Calculate remaining time until deadline.

        Args:
            current_time: Optional current time (defaults to now)

        Returns:
            Remaining seconds (negative if expired)
        """
        if self.deadline_seconds <= 0:
            return float("inf")

        if current_time is None:
            from datetime import timezone
            current_time = datetime.now(timezone.utc)

        # Ensure current_time is timezone-aware
        if current_time.tzinfo is None:
            from datetime import timezone
            current_time = current_time.replace(tzinfo=timezone.utc)

        deadline = self.get_absolute_deadline()
        delta = deadline - current_time
        return delta.total_seconds()

    def is_within_window(self, current_time: datetime | None = None) -> bool:
        """Check if current time is within execution window.

        Args:
            current_time: Optional current time (defaults to now)

        Returns:
            True if within window, False otherwise
        """
        if not self.window_start and not self.window_end:
            return True  # No window constraint

        if current_time is None:
            from datetime import timezone
            current_time = datetime.now(timezone.utc)

        # Ensure current_time is timezone-aware
        if current_time.tzinfo is None:
            from datetime import timezone
            current_time = current_time.replace(tzinfo=timezone.utc)

        if self.window_start:
            start = datetime.fromisoformat(self.window_start.replace("Z", "+00:00"))
            if current_time < start:
                return False

        if self.window_end:
            end = datetime.fromisoformat(self.window_end.replace("Z", "+00:00"))
            if current_time > end:
                return False

        return True


def create_temporal_contract(
    intent_contract_id: str,
    deadline_seconds: float = 0.0,
    budget_seconds: float = 0.0,
    window_start: str = "",
    window_end: str = "",
    rollback_authorized: bool = False,
    temporal_proof: str = "temporal_authorized",
) -> TemporalContract:
    """Create a TemporalContract for time resource management.

    Args:
        intent_contract_id: Reference to IntentContract
        deadline_seconds: Deadline in seconds from creation
        budget_seconds: Time budget in seconds
        window_start: Optional window start timestamp (ISO 8601)
        window_end: Optional window end timestamp (ISO 8601)
        rollback_authorized: Whether rollback is authorized
        temporal_proof: Proof of temporal authorization

    Returns:
        Immutable TemporalContract
    """
    content = {
        "intent_contract_id": intent_contract_id,
        "deadline_seconds": deadline_seconds,
        "budget_seconds": budget_seconds,
        "window_start": window_start,
        "window_end": window_end,
        "rollback_authorized": rollback_authorized,
        "temporal_proof": temporal_proof,
        "created_at": get_current_timestamp(),
        "version": "1.0.0",
    }

    contract_id = generate_contract_id("TemporalContract", content)

    return TemporalContract(
        contract_id=contract_id,
        contract_type="TemporalContract",
        created_at=content["created_at"],
        version=content["version"],
        intent_contract_id=content["intent_contract_id"],
        deadline_seconds=content["deadline_seconds"],
        budget_seconds=content["budget_seconds"],
        window_start=content["window_start"],
        window_end=content["window_end"],
        rollback_authorized=content["rollback_authorized"],
        temporal_proof=content["temporal_proof"],
    )
