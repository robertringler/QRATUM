"""Intervention evaluation logic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InterventionOutcome:
    success: bool
    scores: dict[str, float]
    details: list[dict[str, object]]


class InterventionEvaluation:
    """Score interventions using deterministic weights."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {"compliance": 1.0, "impact": 1.0}

    def evaluate(self, applied: list[dict[str, object]]) -> InterventionOutcome:
        scores: dict[str, float] = {"compliance": 0.0, "impact": 0.0}
        details: list[dict[str, object]] = []
        for entry in applied:
            scores["compliance"] += 1.0 if entry.get("allowed", True) else 0.0
            scores["impact"] += entry.get("impact", 0.0)
            details.append(entry)
        total = sum(scores[k] * self.weights.get(k, 1.0) for k in sorted(scores))
        success = total >= 0
        return InterventionOutcome(success=success, scores=scores, details=details)
