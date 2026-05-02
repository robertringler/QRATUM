"""report.py — drift output schema and ledger adapter.

The spec mandates the exact output shape::

    {
      "anomaly_detected": true,
      "off_topic_tokens": [...],
      "cluster_id": [...],
      "lift_score": 0.0,
      "severity": "low|medium|high"
    }

This module owns that shape and offers
:func:`drift_to_trace_entry` to translate a report into a
:class:`qratum_framework.trace.UnifiedTraceEntry` so anomaly events
can ride the existing Merkle-chained ledger — the audit substrate
Logging Requirement A in the plan calls for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from qratum_framework.observability.drift.severity import Severity
from qratum_framework.trace import UnifiedTraceEntry, hash_state


@dataclass(frozen=True)
class DriftReport:
    """The drift-engine output as a typed record.

    Carries the spec's five mandated fields plus a small set of
    diagnostic side channels (``per_token``, ``thresholds``,
    ``persona``, ``utterance_hash``) used by the eval harness and
    audit consumers.  ``to_schema`` strips the diagnostics and emits
    only the spec shape; ``to_dict`` retains everything for logging.
    """

    anomaly_detected: bool
    off_topic_tokens: Tuple[str, ...]
    cluster_id: Tuple[str, ...]
    lift_score: float
    severity: Severity
    persona: str = "persona"
    utterance_hash: str = ""
    per_token: Tuple[Mapping[str, Any], ...] = ()
    thresholds: Mapping[str, float] = field(default_factory=dict)
    n_tokens: int = 0

    def to_schema(self) -> Dict[str, Any]:
        """Spec-mandated five-field output."""
        return {
            "anomaly_detected": bool(self.anomaly_detected),
            "off_topic_tokens": list(self.off_topic_tokens),
            "cluster_id": list(self.cluster_id),
            "lift_score": float(self.lift_score),
            "severity": str(self.severity),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full diagnostic dict (superset of :meth:`to_schema`)."""
        out = self.to_schema()
        out.update(
            {
                "persona": self.persona,
                "utterance_hash": self.utterance_hash,
                "n_tokens": int(self.n_tokens),
                "per_token": [dict(pt) for pt in self.per_token],
                "thresholds": dict(self.thresholds),
            }
        )
        return out


def drift_to_trace_entry(
    report: DriftReport,
    *,
    step: int,
    intent_raw: str,
) -> UnifiedTraceEntry:
    """Translate a :class:`DriftReport` into a ``UnifiedTraceEntry``.

    The mapping deliberately reuses the existing trace schema rather
    than introducing a parallel one:

    * ``status`` — ``OK`` when no anomaly, ``DRIFT_<SEVERITY>``
      otherwise.  ``status`` strings are not constrained by the spine
      to a fixed enum, so any caller-defined string is accepted.
    * ``failure_mode`` — the categorical severity bucket.
    * ``phi_verdict`` — ``True`` iff no anomaly fired (drift is the
      negation of "all constraints satisfied" in the LLM-behavioural
      sense).
    * ``violated_constraints`` — the off-topic token set.
    * ``metrics`` — anomaly count, peak lift, density, etc.
    * ``metadata`` — diagnostic fields (per-token signals, cluster
      ids), excluded from canonical hashing per the spine convention.
    """
    is_clean = not report.anomaly_detected
    status = "OK" if is_clean else f"DRIFT_{report.severity.upper()}"
    metrics = {
        "lift_score": float(report.lift_score),
        "n_anomalies": float(len(report.off_topic_tokens)),
        "n_tokens": float(report.n_tokens),
        "density": (
            float(len(report.off_topic_tokens)) / float(report.n_tokens)
            if report.n_tokens
            else 0.0
        ),
    }
    return UnifiedTraceEntry(
        step=int(step),
        intent_raw=str(intent_raw),
        intent="drift_check",
        action="anomaly_score",
        status=status,
        failure_mode=None if is_clean else report.severity,
        pre_state_hash=hash_state({"persona": report.persona}),
        post_state_hash=report.utterance_hash or hash_state(None),
        phi_verdict=is_clean,
        violated_constraints=tuple(report.off_topic_tokens),
        metrics=metrics,
        metadata={
            "cluster_id": list(report.cluster_id),
            "per_token": [dict(pt) for pt in report.per_token],
            "thresholds": dict(report.thresholds),
        },
    )


__all__ = ["DriftReport", "drift_to_trace_entry"]
