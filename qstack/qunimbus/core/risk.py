"""Risk scoring."""
from __future__ import annotations


def risk_score(exposures: dict[str, float], sensitivities: dict[str, float]) -> float:
    score = 0.0
    for key, exposure in exposures.items():
        score += exposure * sensitivities.get(key, 1.0)
    return score
