"""regression.py — threshold-based regression detection.

Step 4 of the spec::

    if metrics["lift_cluster_rate"] > threshold:
        fail_test("persona drift regression")

We generalise to a small set of independent thresholds so CI can
distinguish "an anomaly fired but stayed below alert" from "regression
breach".  A single :class:`RegressionVerdict` is produced per matrix
row; the runner-level ``detect_regressions`` helper aggregates over a
list of rows and exposes a single non-zero exit suitable for
``qratum eval --regression``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List, Tuple

if TYPE_CHECKING:
    from qratum_framework.observability.eval.runners import MatrixRow


@dataclass(frozen=True)
class RegressionThresholds:
    """Cut-points that constitute a regression.

    Defaults are deliberately *lenient* — neutral baselines should
    never trip them, and the planted-drift CI signal in the test
    suite must trip them on the nerdy persona.

    * ``max_anomaly_rate``           — fraction of tokens that may be
      off-topic before we call it a regression.
    * ``max_lift_cluster_rate``      — average lift over gated tokens
      ceiling.
    * ``max_cluster_activation_index`` — fraction of discovered
      clusters that may be triggered before we call it a regression.
    """

    max_anomaly_rate: float = 0.20
    max_lift_cluster_rate: float = 1.0
    max_cluster_activation_index: float = 0.50


@dataclass(frozen=True)
class RegressionVerdict:
    """One regression decision with reasons attached."""

    prompt_id: str
    persona: str
    is_baseline: bool
    regression: bool
    reasons: Tuple[str, ...] = ()
    anomaly_rate: float = 0.0
    cluster_activation_index: float = 0.0
    lift_cluster_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt_id,
            "model": self.persona,
            "is_baseline": bool(self.is_baseline),
            "anomaly_rate": float(self.anomaly_rate),
            "cluster_activation_index": float(self.cluster_activation_index),
            "lift_cluster_rate": float(self.lift_cluster_rate),
            "regression": bool(self.regression),
            "reasons": list(self.reasons),
        }


def detect_regressions(
    rows: Iterable["MatrixRow"],
    *,
    thresholds: RegressionThresholds | None = None,
    skip_baseline: bool = True,
) -> List[RegressionVerdict]:
    """Apply ``thresholds`` to every non-baseline row.

    Baseline rows are skipped by default — failing the regression
    check on a baseline row would mean the baseline itself is broken,
    which is a different (and more severe) class of bug; it gets
    flagged via the runner's invariant check, not here.
    """
    th = thresholds or RegressionThresholds()
    out: List[RegressionVerdict] = []
    for row in rows:
        if skip_baseline and row.is_baseline:
            out.append(
                RegressionVerdict(
                    prompt_id=row.prompt_id,
                    persona=row.persona,
                    is_baseline=True,
                    regression=False,
                    anomaly_rate=row.metrics.anomaly_rate,
                    cluster_activation_index=row.metrics.cluster_activation_index,
                    lift_cluster_rate=row.metrics.lift_cluster_rate,
                )
            )
            continue
        reasons: List[str] = []
        m = row.metrics
        if m.anomaly_rate > th.max_anomaly_rate:
            reasons.append(
                f"anomaly_rate {m.anomaly_rate:.3f} > {th.max_anomaly_rate:.3f}"
            )
        if m.lift_cluster_rate > th.max_lift_cluster_rate:
            reasons.append(
                f"lift_cluster_rate {m.lift_cluster_rate:.3f} "
                f"> {th.max_lift_cluster_rate:.3f}"
            )
        if m.cluster_activation_index > th.max_cluster_activation_index:
            reasons.append(
                f"cluster_activation_index {m.cluster_activation_index:.3f} "
                f"> {th.max_cluster_activation_index:.3f}"
            )
        out.append(
            RegressionVerdict(
                prompt_id=row.prompt_id,
                persona=row.persona,
                is_baseline=False,
                regression=bool(reasons),
                reasons=tuple(reasons),
                anomaly_rate=row.metrics.anomaly_rate,
                cluster_activation_index=row.metrics.cluster_activation_index,
                lift_cluster_rate=row.metrics.lift_cluster_rate,
            )
        )
    return out


__all__ = ["RegressionThresholds", "RegressionVerdict", "detect_regressions"]
