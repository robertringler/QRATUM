"""qratum_framework — the unified operational spine for QRATUM.

This package is the *single integration surface* described in the
"QRATUM Fully-Wired, Operational Framework" plan.  It does not replace any
existing subsystem; it projects them onto one coherent control plane:

    * one TraceEntry schema with a Merkle-chained ledger          (trace.py)
    * one Operator pipeline running the canonical 9-step loop     (operator.py)
    * one falsification verdict contract                          (falsifier.py)
    * one config-profile loader                                   (config.py)
    * one health / readiness contract                             (health.py)
    * one CLI (`qratum`)                                          (cli.py)

The Operator wraps the strict CIIR–CRS–RIC executor (qagents.ciir_crs_ric)
and emits Merkle-chained TraceEntries.  Domain backends (qubit, plasma,
genomic, consensus, …) are pluggable via the ``OperatorBackend`` protocol so
that every subsystem traverses the *same* path described in §3 of the plan.

Determinism by default: no global RNG is touched anywhere in this package.
"""

from qratum_framework.config import (
    DEFAULT_PROFILE,
    PROFILES,
    PipelineProfile,
    load_profile,
)
from qratum_framework.falsifier import (
    VERDICT_A,
    VERDICT_A0,
    VERDICT_B,
    Falsifier,
    FalsificationVerdict,
)
from qratum_framework.health import (
    HealthReport,
    HealthStatus,
    ReadinessReport,
    check_health,
    check_readiness,
)
from qratum_framework.operator import (
    Operator,
    OperatorBackend,
    OperatorResult,
    StrictCIIRBackend,
)
from qratum_framework.trace import (
    GENESIS_HASH,
    MerkleLedger,
    UnifiedTraceEntry,
    hash_entry,
)

# SDE (Module I — Streaming Drift Engine).  Imported lazily-friendly: the
# import has zero side-effects beyond defining the public names.
from qratum_framework.sde import (
    AlarmEvaluator,
    AlarmThresholds,
    AlarmTier,
    DriftEvent,
    DriftReading,
    DriftScorer,
    KLDriftScorer,
    MMDDriftScorer,
    RollingWindowBuffer,
    SDEConfig,
    SDESnapshot,
    StreamingDriftEngine,
    Token,
    WindowSnapshot,
)

__all__ = [
    # trace
    "UnifiedTraceEntry",
    "MerkleLedger",
    "GENESIS_HASH",
    "hash_entry",
    # operator
    "Operator",
    "OperatorBackend",
    "OperatorResult",
    "StrictCIIRBackend",
    # falsifier
    "FalsificationVerdict",
    "Falsifier",
    "VERDICT_A",
    "VERDICT_A0",
    "VERDICT_B",
    # config
    "PipelineProfile",
    "PROFILES",
    "DEFAULT_PROFILE",
    "load_profile",
    # health
    "HealthStatus",
    "HealthReport",
    "ReadinessReport",
    "check_health",
    "check_readiness",
    # sde (Module I)
    "Token",
    "WindowSnapshot",
    "RollingWindowBuffer",
    "DriftReading",
    "DriftScorer",
    "KLDriftScorer",
    "MMDDriftScorer",
    "AlarmTier",
    "AlarmThresholds",
    "DriftEvent",
    "AlarmEvaluator",
    "SDESnapshot",
    "StreamingDriftEngine",
    "SDEConfig",
]

__version__ = "0.1.0"
