"""Layer A — RIC Drift Engine.

Detect off-topic clustered semantic emissions under persona influence
using three measurable signals: relevance, log-lift, cluster membership.

See the layer overview in :mod:`qratum_framework.observability`.
"""

from qratum_framework.observability.drift.engine import DriftConfig, DriftEngine
from qratum_framework.observability.drift.report import DriftReport, drift_to_trace_entry
from qratum_framework.observability.drift.severity import Severity, classify_severity
from qratum_framework.observability.drift.signals import (
    LogprobScorer,
    RelevanceScorer,
    cluster_membership_score,
    deterministic_embedding,
    deterministic_logprob,
    log_lift,
    relevance,
)

__all__ = [
    "DriftConfig",
    "DriftEngine",
    "DriftReport",
    "LogprobScorer",
    "RelevanceScorer",
    "Severity",
    "classify_severity",
    "cluster_membership_score",
    "deterministic_embedding",
    "deterministic_logprob",
    "drift_to_trace_entry",
    "log_lift",
    "relevance",
]
