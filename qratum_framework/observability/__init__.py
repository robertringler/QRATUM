"""qratum_framework.observability — LLM behavioral observability infrastructure.

Three deterministic layers per the QRATUM Unified System spec:

* :mod:`qratum_framework.observability.clusters` — emergent semantic
  cluster discovery from persona-tagged co-occurrence graphs (Layer B).
* :mod:`qratum_framework.observability.drift` — real-time anomaly
  detection over the three measurable signals: relevance, log-lift,
  cluster membership (Layer A).
* :mod:`qratum_framework.observability.eval` — prompt × persona × decoding
  matrix runner with regression detection (Layer C).

Determinism by construction: no global RNG is touched anywhere.  Cluster
snapshots and drift events can be appended to a
:class:`qratum_framework.trace.MerkleLedger` for tamper-evident audit, in
keeping with the rest of the operational spine.
"""

from qratum_framework.observability.clusters import (
    Cluster,
    ClusterEngineState,
    CoOccurrenceAccumulator,
    discover_clusters,
)
from qratum_framework.observability.drift import (
    DriftConfig,
    DriftEngine,
    DriftReport,
    Severity,
)
from qratum_framework.observability.eval import (
    EvalReport,
    EvalRunner,
    Persona,
    PromptSpec,
    RegressionVerdict,
    default_personas,
    default_prompts,
)

__all__ = [
    # clusters (B)
    "Cluster",
    "ClusterEngineState",
    "CoOccurrenceAccumulator",
    "discover_clusters",
    # drift (A)
    "DriftConfig",
    "DriftEngine",
    "DriftReport",
    "Severity",
    # eval (C)
    "EvalReport",
    "EvalRunner",
    "Persona",
    "PromptSpec",
    "RegressionVerdict",
    "default_personas",
    "default_prompts",
]
