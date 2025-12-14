"""Outcome metrics and scoring."""
from __future__ import annotations

from dataclasses import dataclass, field

from qscenario.scenario import ScenarioState


@dataclass
class OutcomeSummary:
    success: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, int] = field(default_factory=dict)
    incidents: list[dict[str, object]] = field(default_factory=list)

    def classify(self) -> str:
        if self.success and not self.incidents:
            return "success"
        if not self.success and self.incidents:
            return "failure"
        return "mixed"


def evaluate_outcomes(state: ScenarioState, required_metrics: list[str]) -> OutcomeSummary:
    missing = [name for name in required_metrics if name not in state.metrics]
    success = not missing
    reasons: list[str] = []
    if missing:
        reasons.append(f"missing metrics: {','.join(missing)}")
    if state.incidents:
        reasons.append("incidents detected")
    return OutcomeSummary(success=success, reasons=reasons, metrics=dict(state.metrics), incidents=list(state.incidents))
