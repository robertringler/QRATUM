"""Aethernet Validator - Validator Lifecycle Management for Sovereign Consensus.

This module implements validator lifecycle management including:
- Registration and staking
- Rotation and slashing
- Health monitoring
- Quorum participation

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from contracts.base import compute_contract_hash, get_current_timestamp


class ValidatorStatus(Enum):
    """Validator lifecycle status."""

    PENDING = "pending"  # Registration submitted
    ACTIVE = "active"  # Fully active validator
    JAILED = "jailed"  # Temporarily suspended
    SLASHED = "slashed"  # Stake slashed due to violation
    UNBONDING = "unbonding"  # Stake being withdrawn
    INACTIVE = "inactive"  # Voluntarily deactivated


class SlashingReason(Enum):
    """Reasons for validator slashing."""

    DOUBLE_SIGNING = "double_signing"  # Signed conflicting blocks
    DOWNTIME = "downtime"  # Extended unavailability
    INVALID_BLOCK = "invalid_block"  # Proposed invalid block
    INVARIANT_VIOLATION = "invariant_violation"  # Fatal invariant violated
    GOVERNANCE_VIOLATION = "governance_violation"  # Violated governance rules


@dataclass(frozen=True)
class ValidatorCredentials:
    """Immutable validator credentials.

    Attributes:
        validator_id: Unique validator identifier
        public_key: Validator's public key (hex)
        node_address: Network address
        created_at: Registration timestamp
    """

    validator_id: str
    public_key: str
    node_address: str
    created_at: str

    def compute_hash(self) -> str:
        """Compute credential hash."""
        content = {
            "validator_id": self.validator_id,
            "public_key": self.public_key,
            "node_address": self.node_address,
            "created_at": self.created_at,
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()

    def serialize(self) -> dict[str, Any]:
        """Serialize credentials."""
        return {
            "validator_id": self.validator_id,
            "public_key": self.public_key,
            "node_address": self.node_address,
            "created_at": self.created_at,
            "credential_hash": self.compute_hash(),
        }


@dataclass
class ValidatorStake:
    """Validator stake information.

    Attributes:
        validator_id: Validator identifier
        total_stake: Total staked amount
        self_stake: Self-bonded stake
        delegated_stake: Delegated stake from others
        unbonding_stake: Stake being unbonded
        unbonding_completion: When unbonding completes
    """

    validator_id: str
    total_stake: int = 0
    self_stake: int = 0
    delegated_stake: int = 0
    unbonding_stake: int = 0
    unbonding_completion: str | None = None

    def effective_stake(self) -> int:
        """Get effective stake (excludes unbonding)."""
        return self.total_stake - self.unbonding_stake

    def add_stake(self, amount: int, is_self: bool = True) -> None:
        """Add stake to validator."""
        self.total_stake += amount
        if is_self:
            self.self_stake += amount
        else:
            self.delegated_stake += amount

    def begin_unbonding(self, amount: int, completion_time: str) -> None:
        """Begin unbonding stake."""
        if amount > self.total_stake - self.unbonding_stake:
            raise ValueError("Cannot unbond more than available stake")
        self.unbonding_stake += amount
        self.unbonding_completion = completion_time

    def complete_unbonding(self) -> int:
        """Complete unbonding and return unbonded amount."""
        amount = self.unbonding_stake
        self.total_stake -= amount
        self.unbonding_stake = 0
        self.unbonding_completion = None
        return amount

    def slash(self, percentage: float) -> int:
        """Slash stake by percentage.

        Args:
            percentage: Percentage to slash (0.0 to 1.0)

        Returns:
            Amount slashed
        """
        original_stake = self.total_stake
        slash_amount = int(original_stake * percentage)
        self.total_stake -= slash_amount
        # Proportionally reduce self and delegated based on remaining ratio
        if original_stake > 0 and self.total_stake > 0:
            remaining_ratio = self.total_stake / original_stake
            self.self_stake = int(self.self_stake * remaining_ratio)
            self.delegated_stake = int(self.delegated_stake * remaining_ratio)
        elif self.total_stake == 0:
            self.self_stake = 0
            self.delegated_stake = 0
        return slash_amount


@dataclass
class Validator:
    """A network validator.

    Attributes:
        credentials: Validator credentials
        status: Current status
        stake: Stake information
        commission_rate: Commission rate (0.0 to 1.0)
        uptime_score: Uptime score (0.0 to 1.0)
        blocks_proposed: Number of blocks proposed
        blocks_signed: Number of blocks signed
        slashing_events: List of slashing events
        zone_classification: Security zone (Z0-Z3)
    """

    credentials: ValidatorCredentials
    status: ValidatorStatus = ValidatorStatus.PENDING
    stake: ValidatorStake = field(default_factory=lambda: ValidatorStake(""))
    commission_rate: float = 0.10  # 10% default
    uptime_score: float = 1.0
    blocks_proposed: int = 0
    blocks_signed: int = 0
    slashing_events: list[dict[str, Any]] = field(default_factory=list)
    zone_classification: str = "Z1"

    def __post_init__(self):
        """Initialize stake with validator ID."""
        if not self.stake.validator_id:
            self.stake.validator_id = self.credentials.validator_id

    def is_active(self) -> bool:
        """Check if validator is active."""
        return self.status == ValidatorStatus.ACTIVE

    def voting_power(self) -> int:
        """Get voting power based on effective stake."""
        if not self.is_active():
            return 0
        return self.stake.effective_stake()

    def serialize(self) -> dict[str, Any]:
        """Serialize validator."""
        return {
            "credentials": self.credentials.serialize(),
            "status": self.status.value,
            "stake": {
                "total": self.stake.total_stake,
                "self": self.stake.self_stake,
                "delegated": self.stake.delegated_stake,
                "effective": self.stake.effective_stake(),
            },
            "commission_rate": self.commission_rate,
            "uptime_score": self.uptime_score,
            "blocks_proposed": self.blocks_proposed,
            "blocks_signed": self.blocks_signed,
            "slashing_events_count": len(self.slashing_events),
            "zone": self.zone_classification,
        }


@dataclass(frozen=True)
class SlashingEvent:
    """Record of a slashing event.

    Attributes:
        event_id: Unique event identifier
        validator_id: Validator being slashed
        reason: Reason for slashing
        evidence_hash: Hash of evidence
        slash_percentage: Percentage slashed
        slashed_amount: Amount slashed
        epoch: Epoch when slashing occurred
        timestamp: When slashing occurred
    """

    event_id: str
    validator_id: str
    reason: SlashingReason
    evidence_hash: str
    slash_percentage: float
    slashed_amount: int
    epoch: int
    timestamp: str

    def serialize(self) -> dict[str, Any]:
        """Serialize slashing event."""
        return {
            "event_id": self.event_id,
            "validator_id": self.validator_id,
            "reason": self.reason.value,
            "evidence_hash": self.evidence_hash,
            "slash_percentage": self.slash_percentage,
            "slashed_amount": self.slashed_amount,
            "epoch": self.epoch,
            "timestamp": self.timestamp,
        }


class ValidatorRegistry:
    """Registry of validators with lifecycle management.

    Provides:
    - Registration with minimum stake
    - Staking and delegation
    - Rotation and slashing
    - Uptime monitoring
    """

    # Configuration
    MIN_STAKE = 10000  # Minimum stake to register
    MIN_SELF_STAKE_RATIO = 0.1  # Minimum 10% self-stake
    UNBONDING_PERIOD_EPOCHS = 21  # 21 epochs to unbond
    DOWNTIME_JAIL_THRESHOLD = 0.9  # Below 90% uptime triggers jail
    SLASHING_PERCENTAGES = {
        SlashingReason.DOUBLE_SIGNING: 0.05,  # 5%
        SlashingReason.DOWNTIME: 0.01,  # 1%
        SlashingReason.INVALID_BLOCK: 0.10,  # 10%
        SlashingReason.INVARIANT_VIOLATION: 0.50,  # 50%
        SlashingReason.GOVERNANCE_VIOLATION: 0.20,  # 20%
    }

    def __init__(self):
        """Initialize registry."""
        self.validators: dict[str, Validator] = {}
        self.slashing_history: list[SlashingEvent] = []
        self.current_epoch: int = 0
        self._rotation_schedule: list[str] = []
        self._audit_log: list[dict[str, Any]] = []

    def register_validator(
        self,
        public_key: str,
        node_address: str,
        initial_stake: int,
        commission_rate: float = 0.10,
        zone: str = "Z1",
    ) -> Validator:
        """Register a new validator.

        Args:
            public_key: Validator's public key
            node_address: Network address
            initial_stake: Initial stake amount
            commission_rate: Commission rate
            zone: Security zone

        Returns:
            Registered Validator

        Raises:
            ValueError: If registration fails validation
        """
        # Validate minimum stake
        if initial_stake < self.MIN_STAKE:
            raise ValueError(f"Initial stake {initial_stake} below minimum {self.MIN_STAKE}")

        # Validate commission rate
        if not 0.0 <= commission_rate <= 1.0:
            raise ValueError("Commission rate must be between 0 and 1")

        # Generate validator ID
        timestamp = get_current_timestamp()
        validator_id = f"val_{compute_contract_hash({'pk': public_key, 'ts': timestamp})[:16]}"

        # Check for duplicate
        if validator_id in self.validators:
            raise ValueError(f"Validator {validator_id} already exists")

        # Create credentials
        credentials = ValidatorCredentials(
            validator_id=validator_id,
            public_key=public_key,
            node_address=node_address,
            created_at=timestamp,
        )

        # Create stake
        stake = ValidatorStake(validator_id=validator_id)
        stake.add_stake(initial_stake, is_self=True)

        # Create validator
        validator = Validator(
            credentials=credentials,
            status=ValidatorStatus.PENDING,
            stake=stake,
            commission_rate=commission_rate,
            zone_classification=zone,
        )

        self.validators[validator_id] = validator

        # Log registration
        self._log_event(
            "validator_registered",
            {
                "validator_id": validator_id,
                "initial_stake": initial_stake,
            },
        )

        return validator

    def activate_validator(self, validator_id: str) -> bool:
        """Activate a pending validator.

        Args:
            validator_id: Validator to activate

        Returns:
            True if activated
        """
        validator = self.validators.get(validator_id)
        if not validator:
            return False

        if validator.status != ValidatorStatus.PENDING:
            return False

        # Validate self-stake ratio
        if validator.stake.total_stake > 0:
            ratio = validator.stake.self_stake / validator.stake.total_stake
            if ratio < self.MIN_SELF_STAKE_RATIO:
                return False

        validator.status = ValidatorStatus.ACTIVE
        self._log_event("validator_activated", {"validator_id": validator_id})

        return True

    def delegate_stake(
        self,
        validator_id: str,
        amount: int,
        delegator_id: str,
    ) -> bool:
        """Delegate stake to a validator.

        Args:
            validator_id: Validator to delegate to
            amount: Amount to delegate
            delegator_id: Delegator identifier

        Returns:
            True if delegation successful
        """
        validator = self.validators.get(validator_id)
        if not validator:
            return False

        if validator.status not in (ValidatorStatus.ACTIVE, ValidatorStatus.PENDING):
            return False

        validator.stake.add_stake(amount, is_self=False)

        self._log_event(
            "stake_delegated",
            {
                "validator_id": validator_id,
                "delegator_id": delegator_id,
                "amount": amount,
            },
        )

        return True

    def begin_unbonding(
        self,
        validator_id: str,
        amount: int,
    ) -> bool:
        """Begin unbonding stake from validator.

        Args:
            validator_id: Validator to unbond from
            amount: Amount to unbond

        Returns:
            True if unbonding started
        """
        validator = self.validators.get(validator_id)
        if not validator:
            return False

        completion_epoch = self.current_epoch + self.UNBONDING_PERIOD_EPOCHS
        completion_time = f"epoch_{completion_epoch}"

        try:
            validator.stake.begin_unbonding(amount, completion_time)
            validator.status = ValidatorStatus.UNBONDING

            self._log_event(
                "unbonding_started",
                {
                    "validator_id": validator_id,
                    "amount": amount,
                    "completion_epoch": completion_epoch,
                },
            )

            return True
        except ValueError:
            return False

    def slash_validator(
        self,
        validator_id: str,
        reason: SlashingReason,
        evidence_hash: str,
    ) -> SlashingEvent | None:
        """Slash a validator's stake.

        Args:
            validator_id: Validator to slash
            reason: Reason for slashing
            evidence_hash: Hash of evidence

        Returns:
            SlashingEvent if successful
        """
        validator = self.validators.get(validator_id)
        if not validator:
            return None

        # Get slash percentage for reason
        percentage = self.SLASHING_PERCENTAGES.get(reason, 0.01)

        # Perform slash
        slashed_amount = validator.stake.slash(percentage)

        # Create slashing event
        event_id = f"slash_{compute_contract_hash({'vid': validator_id, 'ts': get_current_timestamp()})[:12]}"
        event = SlashingEvent(
            event_id=event_id,
            validator_id=validator_id,
            reason=reason,
            evidence_hash=evidence_hash,
            slash_percentage=percentage,
            slashed_amount=slashed_amount,
            epoch=self.current_epoch,
            timestamp=get_current_timestamp(),
        )

        # Record event
        validator.slashing_events.append(event.serialize())
        self.slashing_history.append(event)

        # Jail validator for serious violations
        if reason in (SlashingReason.DOUBLE_SIGNING, SlashingReason.INVARIANT_VIOLATION):
            validator.status = ValidatorStatus.JAILED

        # Update status for slashing
        if reason == SlashingReason.INVARIANT_VIOLATION:
            validator.status = ValidatorStatus.SLASHED

        self._log_event(
            "validator_slashed",
            {
                "validator_id": validator_id,
                "reason": reason.value,
                "amount": slashed_amount,
            },
        )

        return event

    def jail_validator(self, validator_id: str, reason: str) -> bool:
        """Jail a validator (temporary suspension).

        Args:
            validator_id: Validator to jail
            reason: Reason for jailing

        Returns:
            True if jailed
        """
        validator = self.validators.get(validator_id)
        if not validator:
            return False

        if validator.status == ValidatorStatus.SLASHED:
            return False  # Cannot jail already slashed

        validator.status = ValidatorStatus.JAILED

        self._log_event(
            "validator_jailed",
            {
                "validator_id": validator_id,
                "reason": reason,
            },
        )

        return True

    def unjail_validator(self, validator_id: str) -> bool:
        """Unjail a validator.

        Args:
            validator_id: Validator to unjail

        Returns:
            True if unjailed
        """
        validator = self.validators.get(validator_id)
        if not validator:
            return False

        if validator.status != ValidatorStatus.JAILED:
            return False

        validator.status = ValidatorStatus.ACTIVE

        self._log_event("validator_unjailed", {"validator_id": validator_id})

        return True

    def update_uptime(self, validator_id: str, uptime_score: float) -> None:
        """Update validator's uptime score.

        Args:
            validator_id: Validator to update
            uptime_score: New uptime score (0.0 to 1.0)
        """
        validator = self.validators.get(validator_id)
        if not validator:
            return

        validator.uptime_score = max(0.0, min(1.0, uptime_score))

        # Auto-jail for low uptime
        if (
            validator.uptime_score < self.DOWNTIME_JAIL_THRESHOLD
            and validator.status == ValidatorStatus.ACTIVE
        ):
            self.jail_validator(validator_id, "low_uptime")
            self.slash_validator(
                validator_id,
                SlashingReason.DOWNTIME,
                compute_contract_hash({"reason": "downtime", "uptime": uptime_score}),
            )

    def record_block_proposed(self, validator_id: str) -> None:
        """Record that validator proposed a block."""
        validator = self.validators.get(validator_id)
        if validator:
            validator.blocks_proposed += 1

    def record_block_signed(self, validator_id: str) -> None:
        """Record that validator signed a block."""
        validator = self.validators.get(validator_id)
        if validator:
            validator.blocks_signed += 1

    def get_active_validators(self) -> list[Validator]:
        """Get list of active validators."""
        return [v for v in self.validators.values() if v.status == ValidatorStatus.ACTIVE]

    def get_total_voting_power(self) -> int:
        """Get total voting power of active validators."""
        return sum(v.voting_power() for v in self.get_active_validators())

    def compute_rotation_schedule(self) -> list[str]:
        """Compute deterministic rotation schedule.

        Returns:
            Ordered list of validator IDs for block proposal rotation
        """
        active = self.get_active_validators()

        # Sort deterministically by voting power then ID
        sorted_validators = sorted(
            active,
            key=lambda v: (-v.voting_power(), v.credentials.validator_id),
        )

        self._rotation_schedule = [v.credentials.validator_id for v in sorted_validators]
        return self._rotation_schedule

    def get_proposer_for_slot(self, slot: int) -> str | None:
        """Get proposer for a given slot.

        Args:
            slot: Block slot number

        Returns:
            Validator ID of proposer
        """
        if not self._rotation_schedule:
            self.compute_rotation_schedule()

        if not self._rotation_schedule:
            return None

        return self._rotation_schedule[slot % len(self._rotation_schedule)]

    def advance_epoch(self) -> None:
        """Advance to next epoch."""
        self.current_epoch += 1

        # Complete any finished unbonding
        for validator in self.validators.values():
            if validator.stake.unbonding_completion:
                completion_epoch = int(validator.stake.unbonding_completion.split("_")[1])
                if self.current_epoch >= completion_epoch:
                    validator.stake.complete_unbonding()
                    if validator.status == ValidatorStatus.UNBONDING:
                        if validator.stake.total_stake >= self.MIN_STAKE:
                            validator.status = ValidatorStatus.ACTIVE
                        else:
                            validator.status = ValidatorStatus.INACTIVE

        # Recompute rotation schedule
        self.compute_rotation_schedule()

        self._log_event("epoch_advanced", {"new_epoch": self.current_epoch})

    def _log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log an audit event."""
        self._audit_log.append(
            {
                "timestamp": get_current_timestamp(),
                "epoch": self.current_epoch,
                "event_type": event_type,
                "data": data,
            }
        )

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Get audit log."""
        return self._audit_log.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        status_counts = {}
        for status in ValidatorStatus:
            status_counts[status.value] = sum(
                1 for v in self.validators.values() if v.status == status
            )

        return {
            "total_validators": len(self.validators),
            "active_validators": len(self.get_active_validators()),
            "total_voting_power": self.get_total_voting_power(),
            "current_epoch": self.current_epoch,
            "slashing_events": len(self.slashing_history),
            "validators_by_status": status_counts,
        }
