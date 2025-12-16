"""Deterministic governance virtual machine for policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class GovernanceRule:
    name: str
    condition_key: str
    threshold: float
    weight: float = 1.0

    def evaluate(self, signals: Dict[str, float]) -> float:
        value = signals.get(self.condition_key, 0.0)
        return self.weight if value >= self.threshold else 0.0


@dataclass
class GovernanceVM:
    rules: List[GovernanceRule] = field(default_factory=list)

    def register_rule(self, rule: GovernanceRule) -> None:
        self.rules.append(rule)

    def score(self, signals: Dict[str, float]) -> float:
        return sum(rule.evaluate(signals) for rule in self.rules)

    def decision(self, signals: Dict[str, float], min_score: float = 0.0) -> Dict[str, float]:
        score_value = self.score(signals)
        return {"score": score_value, "approved": score_value >= min_score}
