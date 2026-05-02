"""metrics.py — per-run and matrix-level aggregations.

Per-run metrics (per spec, Step 4):

* ``anomaly_rate``            = |off_topic| / |tokens|
* ``cluster_activation_index`` = |triggered_clusters| / max(1, |total_clusters|)
* ``lift_cluster_rate``       = mean(lift) over gated tokens, ∈ [0, ∞)

Matrix-level aggregation reduces the per-run metrics over a (prompt,
persona) matrix into a single row used by the regression detector and
the report writer.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

from qratum_framework.observability.clusters.state import ClusterEngineState
from qratum_framework.observability.drift.report import DriftReport

if TYPE_CHECKING:
    from qratum_framework.observability.eval.runners import MatrixRow


@dataclass(frozen=True)
class MatrixMetrics:
    """Per-run metric bundle keyed to a single (prompt, persona) cell."""

    anomaly_rate: float
    cluster_activation_index: float
    lift_cluster_rate: float

    def to_dict(self) -> dict:
        return {
            "anomaly_rate": float(self.anomaly_rate),
            "cluster_activation_index": float(self.cluster_activation_index),
            "lift_cluster_rate": float(self.lift_cluster_rate),
        }


def aggregate_run(
    *,
    report: DriftReport,
    cluster_state: ClusterEngineState,
    tokens: Sequence[str],
) -> MatrixMetrics:
    """Reduce a single run's drift report into per-run metrics."""
    n_tokens = max(1, len(tokens))
    anomaly_rate = float(len(report.off_topic_tokens)) / float(n_tokens)
    n_clusters = max(1, len(cluster_state.clusters))
    cluster_activation_index = float(len(report.cluster_id)) / float(n_clusters)

    gated_lifts = [
        float(pt["lift"])
        for pt in report.per_token
        if pt.get("gated") and pt.get("lift") is not None
    ]
    lift_cluster_rate = float(mean(gated_lifts)) if gated_lifts else 0.0

    return MatrixMetrics(
        anomaly_rate=anomaly_rate,
        cluster_activation_index=cluster_activation_index,
        lift_cluster_rate=lift_cluster_rate,
    )


def matrix_summary(rows: Iterable["MatrixRow"]) -> Mapping[str, float]:
    """Reduce per-cell rows to a few headline numbers for CI logs.

    ``rows`` are :class:`MatrixRow` objects from
    :mod:`qratum_framework.observability.eval.runners`; we only depend
    on ``.metrics`` and ``.is_baseline`` here so this helper is light.
    """
    rows = list(rows)
    if not rows:
        return {
            "n_runs": 0,
            "n_baseline_runs": 0,
            "max_anomaly_rate": 0.0,
            "max_lift_cluster_rate": 0.0,
            "mean_cluster_activation_index": 0.0,
        }
    return {
        "n_runs": float(len(rows)),
        "n_baseline_runs": float(sum(1 for r in rows if r.is_baseline)),
        "max_anomaly_rate": max(r.metrics.anomaly_rate for r in rows),
        "max_lift_cluster_rate": max(r.metrics.lift_cluster_rate for r in rows),
        "mean_cluster_activation_index": (
            sum(r.metrics.cluster_activation_index for r in rows) / len(rows)
        ),
    }


__all__ = ["MatrixMetrics", "aggregate_run", "matrix_summary"]
