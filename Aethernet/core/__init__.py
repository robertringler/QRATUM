"""Aethernet Core - Sovereign Network Infrastructure for QRATUM.

This package provides institutional-grade network infrastructure including:
- Validator lifecycle management (registration, staking, rotation, slashing)
- BFT consensus with quorum governance
- Multi-site federation with air-gapped Z3 support
- Trajectory-aware consensus with vulnerability integration

Version: 1.0.0
Status: Production
"""

from Aethernet.core.consensus import (
    BFTConsensus,
    BlockHeader,
    ConsensusPhase,
    ConsensusRound,
    ConsensusVote,
    QuorumState,
    TrajectoryAwareConsensus,
    VoteType,
)
from Aethernet.core.federation import (
    AirGappedReplicator,
    ArchiveBundle,
    FederationCoordinator,
    FederationRegistry,
    FederationSite,
    ReplayVerification,
    ReplicationMode,
    SiteCredentials,
    SiteType,
    SyncStatus,
)
from Aethernet.core.validator import (
    SlashingEvent,
    SlashingReason,
    Validator,
    ValidatorCredentials,
    ValidatorRegistry,
    ValidatorStake,
    ValidatorStatus,
)

__all__ = [
    # Validator
    "Validator",
    "ValidatorCredentials",
    "ValidatorRegistry",
    "ValidatorStake",
    "ValidatorStatus",
    "SlashingEvent",
    "SlashingReason",
    # Consensus
    "BFTConsensus",
    "BlockHeader",
    "ConsensusPhase",
    "ConsensusRound",
    "ConsensusVote",
    "QuorumState",
    "TrajectoryAwareConsensus",
    "VoteType",
    # Federation
    "FederationCoordinator",
    "FederationRegistry",
    "FederationSite",
    "SiteCredentials",
    "SiteType",
    "SyncStatus",
    "ReplicationMode",
    "ArchiveBundle",
    "AirGappedReplicator",
    "ReplayVerification",
]

__version__ = "1.0.0"
