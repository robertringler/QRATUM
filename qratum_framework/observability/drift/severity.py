"""severity.py — three-bucket severity classifier.

The spec's output schema reserves ``severity ∈ {low, medium, high}``
without prescribing the boundary rule.  We define it operationally as
the conjunction of *anomaly density* and *peak lift*:

* **low**    — anomalies present but rare and weakly persona-shifted.
* **medium** — either density or peak lift crosses one threshold.
* **high**   — both cross, i.e. a clustered, strongly amplified emission.

This rule is monotone in both inputs and lets the engine surface a
bounded categorical alongside the raw counts so downstream consumers
(CI, dashboards) can branch on it without re-deriving thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: Public severity literal type.
Severity = Literal["none", "low", "medium", "high"]


@dataclass(frozen=True)
class SeverityThresholds:
    """Configurable cut-points for the bucket rule.

    Defaults are deliberately conservative — they classify a single
    weak anomaly as ``low`` and require both clustered density (≥ 30%
    of tokens) and a peak lift ≥ 2.0 nats to escalate to ``high``.
    """

    medium_density: float = 0.10
    medium_peak_lift: float = 1.0
    high_density: float = 0.30
    high_peak_lift: float = 2.0


def classify_severity(
    *,
    n_anomalies: int,
    n_tokens: int,
    peak_lift: float,
    thresholds: SeverityThresholds | None = None,
) -> Severity:
    """Bucket a single drift report into ``none|low|medium|high``."""
    th = thresholds or SeverityThresholds()
    if n_anomalies <= 0 or n_tokens <= 0:
        return "none"
    density = n_anomalies / n_tokens
    if density >= th.high_density and peak_lift >= th.high_peak_lift:
        return "high"
    if density >= th.medium_density or peak_lift >= th.medium_peak_lift:
        return "medium"
    return "low"


__all__ = ["Severity", "SeverityThresholds", "classify_severity"]
