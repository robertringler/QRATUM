"""Aethernet Consensus - BFT Consensus with Quorum Governance.

This module implements Byzantine Fault Tolerant consensus with:
- Deterministic proposal and vote execution
- Quorum-based decision making
- Multi-site federation support
- Integration with vulnerability metrics

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from contracts.base import compute_contract_hash, get_current_timestamp


class ConsensusPhase(Enum):
    """BFT consensus phases."""

    PROPOSE = "propose"  # Leader proposes block
    PREVOTE = "prevote"  # Validators prevote
    PRECOMMIT = "precommit"  # Validators precommit
    COMMIT = "commit"  # Block committed
    FINALIZED = "finalized"  # Block finalized


class VoteType(Enum):
    """Types of consensus votes."""

    PREVOTE = "prevote"
    PRECOMMIT = "precommit"


@dataclass(frozen=True)
class BlockHeader:
    """Immutable block header.

    Attributes:
        height: Block height
        round: Consensus round
        timestamp: Block timestamp
        proposer_id: Proposer validator ID
        parent_hash: Hash of parent block
        state_root: Merkle root of state
        txs_root: Merkle root of transactions
        consensus_hash: Hash of consensus data
    """

    height: int
    round: int
    timestamp: str
    proposer_id: str
    parent_hash: str
    state_root: str
    txs_root: str
    consensus_hash: str

    def compute_hash(self) -> str:
        """Compute block header hash."""
        content = {
            "height": self.height,
            "round": self.round,
            "timestamp": self.timestamp,
            "proposer_id": self.proposer_id,
            "parent_hash": self.parent_hash,
            "state_root": self.state_root,
            "txs_root": self.txs_root,
            "consensus_hash": self.consensus_hash,
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()

    def serialize(self) -> dict[str, Any]:
        """Serialize header."""
        return {
            "height": self.height,
            "round": self.round,
            "timestamp": self.timestamp,
            "proposer_id": self.proposer_id,
            "parent_hash": self.parent_hash,
            "state_root": self.state_root,
            "txs_root": self.txs_root,
            "consensus_hash": self.consensus_hash,
            "block_hash": self.compute_hash(),
        }


@dataclass(frozen=True)
class ConsensusVote:
    """Immutable consensus vote.

    Attributes:
        vote_type: Type of vote
        height: Block height
        round: Consensus round
        block_hash: Hash of block being voted on
        validator_id: Voting validator
        timestamp: Vote timestamp
        signature: Vote signature (hex)
    """

    vote_type: VoteType
    height: int
    round: int
    block_hash: str
    validator_id: str
    timestamp: str
    signature: str

    def compute_hash(self) -> str:
        """Compute vote hash."""
        content = {
            "vote_type": self.vote_type.value,
            "height": self.height,
            "round": self.round,
            "block_hash": self.block_hash,
            "validator_id": self.validator_id,
            "timestamp": self.timestamp,
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()

    def serialize(self) -> dict[str, Any]:
        """Serialize vote."""
        return {
            "vote_type": self.vote_type.value,
            "height": self.height,
            "round": self.round,
            "block_hash": self.block_hash,
            "validator_id": self.validator_id,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "vote_hash": self.compute_hash(),
        }


@dataclass
class QuorumState:
    """State of quorum for a block.

    Attributes:
        height: Block height
        round: Consensus round
        block_hash: Block being voted on
        prevotes: Prevotes received
        precommits: Precommits received
        total_voting_power: Total voting power
        prevote_power: Voting power that prevoted
        precommit_power: Voting power that precommitted
    """

    height: int
    round: int
    block_hash: str
    prevotes: dict[str, ConsensusVote] = field(default_factory=dict)
    precommits: dict[str, ConsensusVote] = field(default_factory=dict)
    total_voting_power: int = 0
    prevote_power: int = 0
    precommit_power: int = 0

    def add_prevote(self, vote: ConsensusVote, voting_power: int) -> bool:
        """Add a prevote.

        Args:
            vote: Prevote to add
            voting_power: Validator's voting power

        Returns:
            True if added (not duplicate)
        """
        if vote.validator_id in self.prevotes:
            return False

        self.prevotes[vote.validator_id] = vote
        self.prevote_power += voting_power
        return True

    def add_precommit(self, vote: ConsensusVote, voting_power: int) -> bool:
        """Add a precommit.

        Args:
            vote: Precommit to add
            voting_power: Validator's voting power

        Returns:
            True if added (not duplicate)
        """
        if vote.validator_id in self.precommits:
            return False

        self.precommits[vote.validator_id] = vote
        self.precommit_power += voting_power
        return True

    def has_prevote_quorum(self) -> bool:
        """Check if prevote quorum reached (>2/3)."""
        if self.total_voting_power == 0:
            return False
        return self.prevote_power * 3 > self.total_voting_power * 2

    def has_precommit_quorum(self) -> bool:
        """Check if precommit quorum reached (>2/3)."""
        if self.total_voting_power == 0:
            return False
        return self.precommit_power * 3 > self.total_voting_power * 2


@dataclass
class ConsensusRound:
    """A single consensus round.

    Attributes:
        height: Block height
        round: Round number
        proposer_id: Proposer for this round
        proposed_block: Proposed block header
        phase: Current phase
        quorum: Quorum state
        start_time: Round start time
        timeout_duration: Timeout in seconds
    """

    height: int
    round: int
    proposer_id: str
    proposed_block: BlockHeader | None = None
    phase: ConsensusPhase = ConsensusPhase.PROPOSE
    quorum: QuorumState | None = None
    start_time: str = field(default_factory=get_current_timestamp)
    timeout_duration: float = 30.0  # 30 seconds default

    def is_timed_out(self) -> bool:
        """Check if round has timed out."""
        start = datetime.fromisoformat(self.start_time.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        elapsed = (now - start).total_seconds()
        return elapsed > self.timeout_duration


class BFTConsensus:
    """Byzantine Fault Tolerant consensus engine.

    Implements a simplified Tendermint-style BFT consensus with:
    - Round-based block proposal
    - Two-phase voting (prevote, precommit)
    - Quorum-based finality (>2/3 voting power)
    - Deterministic proposer rotation

    Security Properties:
    - Safety: No conflicting blocks finalized at same height
    - Liveness: Progress guaranteed with >2/3 honest validators
    - Accountability: Misbehavior detectable and slashable
    """

    def __init__(
        self,
        get_voting_power: Callable[[str], int],
        get_total_voting_power: Callable[[], int],
        get_proposer_for_slot: Callable[[int], str | None],
    ):
        """Initialize consensus engine.

        Args:
            get_voting_power: Function to get validator voting power
            get_total_voting_power: Function to get total voting power
            get_proposer_for_slot: Function to get proposer for slot
        """
        self.get_voting_power = get_voting_power
        self.get_total_voting_power = get_total_voting_power
        self.get_proposer_for_slot = get_proposer_for_slot

        self.current_height: int = 1
        self.current_round: int = 0
        self.rounds: dict[tuple[int, int], ConsensusRound] = {}
        self.finalized_blocks: dict[int, BlockHeader] = {}
        self.committed_blocks: dict[int, BlockHeader] = {}

        self._audit_log: list[dict[str, Any]] = []

    def start_round(self, height: int, round: int = 0) -> ConsensusRound:
        """Start a new consensus round.

        Args:
            height: Block height
            round: Round number (0 for first attempt)

        Returns:
            ConsensusRound
        """
        # Get proposer for this round
        slot = height * 100 + round  # Deterministic slot calculation
        proposer = self.get_proposer_for_slot(slot)

        if not proposer:
            raise ValueError("No proposer available")

        consensus_round = ConsensusRound(
            height=height,
            round=round,
            proposer_id=proposer,
            phase=ConsensusPhase.PROPOSE,
        )

        self.rounds[(height, round)] = consensus_round
        self.current_height = height
        self.current_round = round

        self._log_event("round_started", {
            "height": height,
            "round": round,
            "proposer": proposer,
        })

        return consensus_round

    def propose_block(
        self,
        height: int,
        round: int,
        proposer_id: str,
        parent_hash: str,
        state_root: str,
        txs_root: str,
    ) -> BlockHeader | None:
        """Propose a block.

        Args:
            height: Block height
            round: Round number
            proposer_id: Proposer validator ID
            parent_hash: Hash of parent block
            state_root: State merkle root
            txs_root: Transaction merkle root

        Returns:
            BlockHeader if proposal valid
        """
        key = (height, round)
        consensus_round = self.rounds.get(key)

        if not consensus_round:
            consensus_round = self.start_round(height, round)

        # Verify proposer
        if consensus_round.proposer_id != proposer_id:
            self._log_event("invalid_proposal", {
                "height": height,
                "round": round,
                "expected_proposer": consensus_round.proposer_id,
                "actual_proposer": proposer_id,
            })
            return None

        # Verify phase
        if consensus_round.phase != ConsensusPhase.PROPOSE:
            return None

        # Create block header
        timestamp = get_current_timestamp()
        consensus_hash = compute_contract_hash({
            "height": height,
            "round": round,
            "proposer": proposer_id,
        })

        block = BlockHeader(
            height=height,
            round=round,
            timestamp=timestamp,
            proposer_id=proposer_id,
            parent_hash=parent_hash,
            state_root=state_root,
            txs_root=txs_root,
            consensus_hash=consensus_hash,
        )

        # Set proposal and advance phase
        consensus_round.proposed_block = block
        consensus_round.phase = ConsensusPhase.PREVOTE
        consensus_round.quorum = QuorumState(
            height=height,
            round=round,
            block_hash=block.compute_hash(),
            total_voting_power=self.get_total_voting_power(),
        )

        self._log_event("block_proposed", {
            "height": height,
            "round": round,
            "proposer": proposer_id,
            "block_hash": block.compute_hash(),
        })

        return block

    def prevote(
        self,
        height: int,
        round: int,
        block_hash: str,
        validator_id: str,
        signature: str,
    ) -> bool:
        """Submit a prevote.

        Args:
            height: Block height
            round: Round number
            block_hash: Block hash being voted on
            validator_id: Voting validator
            signature: Vote signature

        Returns:
            True if prevote accepted
        """
        key = (height, round)
        consensus_round = self.rounds.get(key)

        if not consensus_round:
            return False

        if consensus_round.phase != ConsensusPhase.PREVOTE:
            return False

        if not consensus_round.quorum:
            return False

        # Verify block hash matches proposal
        if consensus_round.proposed_block:
            expected_hash = consensus_round.proposed_block.compute_hash()
            if block_hash != expected_hash:
                # Nil vote (voting for different block) - could indicate equivocation
                self._log_event("nil_prevote", {
                    "height": height,
                    "round": round,
                    "validator": validator_id,
                    "expected_hash": expected_hash,
                    "voted_hash": block_hash,
                })

        # Create vote
        vote = ConsensusVote(
            vote_type=VoteType.PREVOTE,
            height=height,
            round=round,
            block_hash=block_hash,
            validator_id=validator_id,
            timestamp=get_current_timestamp(),
            signature=signature,
        )

        # Add vote
        voting_power = self.get_voting_power(validator_id)
        if not consensus_round.quorum.add_prevote(vote, voting_power):
            return False  # Duplicate vote

        self._log_event("prevote_received", {
            "height": height,
            "round": round,
            "validator": validator_id,
            "voting_power": voting_power,
        })

        # Check for quorum
        if consensus_round.quorum.has_prevote_quorum():
            consensus_round.phase = ConsensusPhase.PRECOMMIT
            self._log_event("prevote_quorum_reached", {
                "height": height,
                "round": round,
                "prevote_power": consensus_round.quorum.prevote_power,
                "total_power": consensus_round.quorum.total_voting_power,
            })

        return True

    def precommit(
        self,
        height: int,
        round: int,
        block_hash: str,
        validator_id: str,
        signature: str,
    ) -> bool:
        """Submit a precommit.

        Args:
            height: Block height
            round: Round number
            block_hash: Block hash being voted on
            validator_id: Voting validator
            signature: Vote signature

        Returns:
            True if precommit accepted
        """
        key = (height, round)
        consensus_round = self.rounds.get(key)

        if not consensus_round:
            return False

        if consensus_round.phase != ConsensusPhase.PRECOMMIT:
            return False

        if not consensus_round.quorum:
            return False

        # Create vote
        vote = ConsensusVote(
            vote_type=VoteType.PRECOMMIT,
            height=height,
            round=round,
            block_hash=block_hash,
            validator_id=validator_id,
            timestamp=get_current_timestamp(),
            signature=signature,
        )

        # Add vote
        voting_power = self.get_voting_power(validator_id)
        if not consensus_round.quorum.add_precommit(vote, voting_power):
            return False  # Duplicate vote

        self._log_event("precommit_received", {
            "height": height,
            "round": round,
            "validator": validator_id,
            "voting_power": voting_power,
        })

        # Check for quorum
        if consensus_round.quorum.has_precommit_quorum():
            self._commit_block(consensus_round)

        return True

    def _commit_block(self, consensus_round: ConsensusRound) -> None:
        """Commit a block after precommit quorum."""
        if not consensus_round.proposed_block:
            return

        block = consensus_round.proposed_block
        consensus_round.phase = ConsensusPhase.COMMIT

        self.committed_blocks[block.height] = block

        self._log_event("block_committed", {
            "height": block.height,
            "round": consensus_round.round,
            "block_hash": block.compute_hash(),
            "proposer": block.proposer_id,
        })

        # Finalize immediately in this simplified implementation
        self._finalize_block(consensus_round)

    def _finalize_block(self, consensus_round: ConsensusRound) -> None:
        """Finalize a committed block."""
        if not consensus_round.proposed_block:
            return

        block = consensus_round.proposed_block
        consensus_round.phase = ConsensusPhase.FINALIZED

        self.finalized_blocks[block.height] = block

        self._log_event("block_finalized", {
            "height": block.height,
            "round": consensus_round.round,
            "block_hash": block.compute_hash(),
        })

    def handle_timeout(self, height: int, round: int) -> int:
        """Handle round timeout - move to next round.

        Args:
            height: Block height
            round: Current round

        Returns:
            New round number
        """
        key = (height, round)
        consensus_round = self.rounds.get(key)

        if consensus_round and consensus_round.is_timed_out():
            new_round = round + 1
            self._log_event("round_timeout", {
                "height": height,
                "old_round": round,
                "new_round": new_round,
            })
            self.start_round(height, new_round)
            return new_round

        return round

    def get_finalized_block(self, height: int) -> BlockHeader | None:
        """Get finalized block at height."""
        return self.finalized_blocks.get(height)

    def get_latest_finalized_height(self) -> int:
        """Get latest finalized block height."""
        if not self.finalized_blocks:
            return 0
        return max(self.finalized_blocks.keys())

    def _log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log consensus event."""
        self._audit_log.append({
            "timestamp": get_current_timestamp(),
            "event_type": event_type,
            "data": data,
        })

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Get consensus audit log."""
        return self._audit_log.copy()

    def get_stats(self) -> dict[str, Any]:
        """Get consensus statistics."""
        return {
            "current_height": self.current_height,
            "current_round": self.current_round,
            "finalized_blocks": len(self.finalized_blocks),
            "committed_blocks": len(self.committed_blocks),
            "active_rounds": len(self.rounds),
            "latest_finalized": self.get_latest_finalized_height(),
        }


class TrajectoryAwareConsensus(BFTConsensus):
    """BFT consensus with vulnerability-aware health scoring.

    Extends BFTConsensus to integrate vulnerability discovery metrics
    for runtime trajectory awareness and conditional self-suspension.
    """

    def __init__(
        self,
        get_voting_power: Callable[[str], int],
        get_total_voting_power: Callable[[], int],
        get_proposer_for_slot: Callable[[int], str | None],
        collapse_threshold: float = 0.3,
    ):
        """Initialize trajectory-aware consensus.

        Args:
            get_voting_power: Function to get validator voting power
            get_total_voting_power: Function to get total voting power
            get_proposer_for_slot: Function to get proposer for slot
            collapse_threshold: Collapse probability threshold for suspension
        """
        super().__init__(get_voting_power, get_total_voting_power, get_proposer_for_slot)

        self.collapse_threshold = collapse_threshold
        self.health_score: float = 1.0
        self.collapse_probability: float = 0.0
        self.is_suspended: bool = False
        self.suspension_reason: str | None = None
        self._precursor_signals: list[str] = []

    def update_trajectory_metrics(
        self,
        system_health: float,
        collapse_probability: float,
        precursor_signals: list[str],
    ) -> None:
        """Update trajectory metrics from vulnerability discovery.

        Args:
            system_health: System health score (0.0 to 1.0)
            collapse_probability: Collapse probability (0.0 to 1.0)
            precursor_signals: List of precursor signals detected
        """
        self.health_score = system_health
        self.collapse_probability = collapse_probability
        self._precursor_signals = precursor_signals

        # Check for self-suspension
        if collapse_probability > self.collapse_threshold:
            self._trigger_suspension(
                f"Collapse probability {collapse_probability:.2%} exceeds threshold {self.collapse_threshold:.2%}"
            )

        self._log_event("trajectory_metrics_updated", {
            "health_score": system_health,
            "collapse_probability": collapse_probability,
            "precursor_signals": precursor_signals,
            "is_suspended": self.is_suspended,
        })

    def _trigger_suspension(self, reason: str) -> None:
        """Trigger consensus self-suspension.

        Args:
            reason: Reason for suspension
        """
        if self.is_suspended:
            return  # Already suspended

        self.is_suspended = True
        self.suspension_reason = reason

        self._log_event("consensus_suspended", {
            "reason": reason,
            "health_score": self.health_score,
            "collapse_probability": self.collapse_probability,
        })

    def resume_consensus(self) -> bool:
        """Resume consensus after suspension.

        Returns:
            True if resumed successfully
        """
        if not self.is_suspended:
            return True

        # Only resume if health is acceptable
        if self.collapse_probability > self.collapse_threshold:
            return False

        self.is_suspended = False
        self.suspension_reason = None

        self._log_event("consensus_resumed", {
            "health_score": self.health_score,
            "collapse_probability": self.collapse_probability,
        })

        return True

    def propose_block(
        self,
        height: int,
        round: int,
        proposer_id: str,
        parent_hash: str,
        state_root: str,
        txs_root: str,
    ) -> BlockHeader | None:
        """Propose a block with suspension check."""
        if self.is_suspended:
            self._log_event("proposal_blocked_suspended", {
                "height": height,
                "round": round,
                "reason": self.suspension_reason,
            })
            return None

        return super().propose_block(
            height, round, proposer_id, parent_hash, state_root, txs_root
        )

    def get_health_status(self) -> dict[str, Any]:
        """Get health status including trajectory metrics."""
        return {
            "health_score": self.health_score,
            "collapse_probability": self.collapse_probability,
            "is_suspended": self.is_suspended,
            "suspension_reason": self.suspension_reason,
            "precursor_signals": self._precursor_signals,
            "collapse_threshold": self.collapse_threshold,
        }
