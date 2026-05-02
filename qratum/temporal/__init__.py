"""
QRATUM Temporal Computing Framework

Enables computational time travel through QRATUM's speed-of-light-breaking paradigm.
Transforms time from a constraint into a navigable computational dimension.

Physical Time Travel: Bounded by c, requires infinite energy, causality paradoxes
Computational Time Travel: Bounded by FLOPS, requires determinism + verification

QRATUM achieves "effective FTL" by computing temporal states faster than light could
traverse the physical distance between events. At 2.5 ExaFLOPS, QRATUM completes
8.3 billion calculations per meter light travels.

Example Usage:
    >>> from qratum.temporal import TemporalEngine, FTLComputation, TimelineManager
    >>> from qratum.temporal.constants import SECONDS_PER_YEAR
    >>> 
    >>> # Initialize temporal engine
    >>> engine = TemporalEngine(
    ...     peak_flops=2.5e18,
    ...     deterministic=True,
    ...     merkle_verified=True,
    ... )
    >>> 
    >>> # Travel forward: predict climate 100 years from now
    >>> future_climate, proof = engine.forward(
    ...     initial_state=current_climate,
    ...     delta_t=100 * SECONDS_PER_YEAR,
    ...     evolution_fn=climate_model,
    ... )
    >>> 
    >>> # Explore parallel timelines
    >>> manager = TimelineManager()
    >>> branches = engine.branch(
    ...     branch_point=decision_moment,
    ...     branches=[
    ...         ("choice_a", apply_choice_a),
    ...         ("choice_b", apply_choice_b),
    ...     ],
    ...     evolution_fn=consequence_model,
    ...     delta_t=50 * SECONDS_PER_YEAR,
    ... )
    >>> 
    >>> # Find optimal future across all possibilities
    >>> ftl = FTLComputation(peak_flops=2.5e18)
    >>> best_outcome, best_branch, velocity = ftl.parallel_reality_search(
    ...     branch_point=current_state,
    ...     branch_count=10000,
    ...     evolution_fn=world_model,
    ...     mutation_fn=random_intervention,
    ...     target_fn=lambda s: s.human_flourishing_index,
    ...     duration=100 * SECONDS_PER_YEAR,
    ... )
"""

__version__ = "1.0.0"

# Core temporal engine
from .engine import TemporalEngine, TemporalEngineConfig

# State management
from .state import (
    TemporalState,
    StateChain,
    TemporalCoordinate,
)

# FTL computation
from .ftl import (
    FTLComputation,
    PossibilityOracle,
    EffectiveVelocity,
    PossibilitySpace,
)

# Paradox resolution
from .paradox import (
    ParadoxResolver,
    CausalityValidator,
    ParadoxDetection,
    ParadoxType,
)

# Timeline management
from .timeline import (
    Timeline,
    TimelineManager,
    TimelineBranch,
)

# Temporal reversal
from .reversal import (
    CausalReversal,
    EntropyReversal,
    CheckpointReversal,
    ReversalResult,
    ReversalStrategy,
)

# Verification
from .verification import (
    TemporalVerifier,
    TemporalProof,
)

# Integration
from .integration import (
    TemporalEngineDistributed,
    ExascaleConfig,
    QDRIntegration,
    AetherFabricIntegration,
    PQCIntegration,
    MerkleHardwareAcceleration,
)

# Constants
from .constants import (
    SPEED_OF_LIGHT,
    PLANCK_TIME,
    SECONDS_PER_YEAR,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
    QRATUM_PEAK_FLOPS,
    OPERATIONS_PER_LIGHT_METER,
    TARGET_COMPRESSION_RATIO,
    ParadoxStrategy,
    EntropyStrategy,
    TemporalResolution,
    FTLMechanism,
)

__all__ = [
    # Core
    "TemporalEngine",
    "TemporalEngineConfig",
    
    # State
    "TemporalState",
    "StateChain",
    "TemporalCoordinate",
    
    # FTL
    "FTLComputation",
    "PossibilityOracle",
    "EffectiveVelocity",
    "PossibilitySpace",
    
    # Paradox
    "ParadoxResolver",
    "CausalityValidator",
    "ParadoxDetection",
    "ParadoxType",
    
    # Timeline
    "Timeline",
    "TimelineManager",
    "TimelineBranch",
    
    # Reversal
    "CausalReversal",
    "EntropyReversal",
    "CheckpointReversal",
    "ReversalResult",
    "ReversalStrategy",
    
    # Verification
    "TemporalVerifier",
    "TemporalProof",
    
    # Integration
    "TemporalEngineDistributed",
    "ExascaleConfig",
    "QDRIntegration",
    "AetherFabricIntegration",
    "PQCIntegration",
    "MerkleHardwareAcceleration",
    
    # Constants
    "SPEED_OF_LIGHT",
    "PLANCK_TIME",
    "SECONDS_PER_YEAR",
    "SECONDS_PER_DAY",
    "SECONDS_PER_HOUR",
    "SECONDS_PER_MINUTE",
    "QRATUM_PEAK_FLOPS",
    "OPERATIONS_PER_LIGHT_METER",
    "TARGET_COMPRESSION_RATIO",
    "ParadoxStrategy",
    "EntropyStrategy",
    "TemporalResolution",
    "FTLMechanism",
]
