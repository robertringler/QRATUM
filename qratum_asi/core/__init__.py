"""Core types and infrastructure for QRATUM-ASI."""

from qratum_asi.core.types import (
    ASISafetyLevel,
    AuthorizationType,
    ReasoningStrategy,
    ImprovementType,
    GoalCategory,
)
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.events import ASIEvent, ASIEventType
from qratum_asi.core.chain import ASIMerkleChain
from qratum_asi.core.authorization import AuthorizationSystem, AuthorizationRequest

# Calibration Doctrine (12 Axioms)
from qratum_asi.core.calibration_doctrine import (
    CalibrationAxiom,
    CalibrationCategory,
    CalibrationDoctrineEnforcer,
    JurisdictionalProperty,
    TrajectoryMetrics,
    TrajectoryState,
    JurisdictionalClaim,
    CALIBRATION_DOCTRINE,
    get_doctrine_enforcer,
)

# ZK State Verification (Task 4)
from qratum_asi.core.zk_state_verifier import (
    ZKProof,
    ZKStateTransition,
    ZKStateVerifier,
    ZKProofGenerator,
    ZKVerificationContext,
    StateCommitment,
    TransitionType,
    VerificationResult,
    ReplayCache,
    verify_state_transition,
    generate_commitment,
)

__all__ = [
    # Types
    "ASISafetyLevel",
    "AuthorizationType",
    "ReasoningStrategy",
    "ImprovementType",
    "GoalCategory",
    # Core
    "ASIContract",
    "ASIEvent",
    "ASIEventType",
    "ASIMerkleChain",
    "AuthorizationSystem",
    "AuthorizationRequest",
    # Calibration Doctrine
    "CalibrationAxiom",
    "CalibrationCategory",
    "CalibrationDoctrineEnforcer",
    "JurisdictionalProperty",
    "TrajectoryMetrics",
    "TrajectoryState",
    "JurisdictionalClaim",
    "CALIBRATION_DOCTRINE",
    "get_doctrine_enforcer",
    # ZK State Verification
    "ZKProof",
    "ZKStateTransition",
    "ZKStateVerifier",
    "ZKProofGenerator",
    "ZKVerificationContext",
    "StateCommitment",
    "TransitionType",
    "VerificationResult",
    "ReplayCache",
    "verify_state_transition",
    "generate_commitment",
]
