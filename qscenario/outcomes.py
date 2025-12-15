"""Outcome metrics and scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from qscenario.scenario import ScenarioState


@dataclass
class OutcomeSummary:
    success: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, int] = field(default_factory=dict)
    incidents: List[Dict[str, object]] = field(default_factory=list)

    def classify(self) -> str:
        if self.success and not self.incidents:
            return "success"
        if not self.success and self.incidents:
            return "failure"
        return "mixed"


def evaluate_outcomes(state: ScenarioState, required_metrics: List[str]) -> OutcomeSummary:
    missing = [name for name in required_metrics if name not in state.metrics]
    success = not missing
    reasons: List[str] = []
    if missing:
        reasons.append(f"missing metrics: {','.join(missing)}")
    if state.incidents:
        reasons.append("incidents detected")
    return OutcomeSummary(
        success=success,
        reasons=reasons,
        metrics=dict(state.metrics),
        incidents=list(state.incidents),
    )
