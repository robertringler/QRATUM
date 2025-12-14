"""Finance domain hooks."""
from __future__ import annotations


from ..core.risk import risk_score


def portfolio_risk(exposures: dict[str, float]) -> float:
    sensitivities = {k: 1.0 for k in exposures}
    return risk_score(exposures, sensitivities)
