"""Risk scoring."""

from __future__ import annotations

from typing import Dict


def risk_score(exposures: Dict[str, float], sensitivities: Dict[str, float]) -> float:
    score = 0.0
    for key, exposure in exposures.items():
        score += exposure * sensitivities.get(key, 1.0)
    return score
