from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from ..integration.qnx_adapter import valuation_operator


@dataclass(frozen=True)
class ValuationInput:
    metrics: Dict[str, float]


class GovernanceResult(dict):
    pass


class QuNimbusEngine:
    """Deterministic valuation engine with governance and risk hooks."""

    def __init__(self, weights: Dict[str, float], governance_rules: Dict[str, float] | None = None):
        self._weights = weights
        self._governance_rules = governance_rules or {}

    def evaluate(self, payload: ValuationInput) -> float:
        total = 0.0
        for key, weight in sorted(self._weights.items(), key=lambda kv: kv[0]):
            metric = payload.metrics.get(key, 0.0)
            total += weight * metric
        return total

    def governance_score(self, payload: ValuationInput) -> GovernanceResult:
        score = self.evaluate(payload)
        penalties: Dict[str, float] = {}
        for rule, threshold in sorted(self._governance_rules.items(), key=lambda kv: kv[0]):
            metric = payload.metrics.get(rule, 0.0)
            penalties[rule] = 0.0 if metric <= threshold else (metric - threshold)
        return GovernanceResult(
            {"base": score, "penalties": penalties, "final": score - sum(penalties.values())}
        )

    def valuation_operator(self) -> Callable[[Dict[str, float]], Dict[str, float]]:
        return valuation_operator(self)


def macro_weights() -> Dict[str, float]:
    return {"gdp": 0.5, "inflation": -0.2, "employment": 0.3}


def macro_engine() -> QuNimbusEngine:
    return QuNimbusEngine(weights=macro_weights(), governance_rules={"inflation": 0.05})
