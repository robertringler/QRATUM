"""Finance domain hooks."""

from __future__ import annotations

from ..core.risk import risk_score


def portfolio_risk(exposures: dict[str, float]) -> float:
    sensitivities = dict.fromkeys(exposures, 1.0)
    return risk_score(exposures, sensitivities)
