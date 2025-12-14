"""Node scoring based on scenario outcomes."""

from __future__ import annotations

from typing import Dict

from qscenario.reporting import ScenarioReport


def score_node(report: ScenarioReport) -> Dict[str, object]:
    incidents = len(report.outcome.incidents)
    metric_score = sum(report.outcome.metrics.values()) if report.outcome.metrics else 0
    compliance = 0 if incidents else 1
    safety_penalty = incidents * 5
    score = max(0, metric_score + compliance * 10 - safety_penalty)
    return {
        "node": report.config.name,
        "score": score,
        "incidents": incidents,
        "classification": report.outcome.classify(),
    }
