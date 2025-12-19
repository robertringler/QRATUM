"""Deterministic governance and voting primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class GovernanceRule:
    identifier: str
    threshold: float
    weight: float

    def evaluate(self, metrics: Dict[str, float]) -> float:
        score = metrics.get(self.identifier, 0.0)
        return self.weight if score >= self.threshold else 0.0


def vote_outcome(votes: Dict[str, int]) -> Dict[str, object]:
    tally = sum(votes.values()) or 1
    normalized = {k: v / tally for k, v in votes.items()}
    winner = sorted(normalized.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return {"winner": winner[0], "share": winner[1], "distribution": normalized}


def governance_score(rules: List[GovernanceRule], metrics: Dict[str, float]) -> Dict[str, float]:
    total_weight = sum(rule.weight for rule in rules) or 1.0
    satisfied = sum(rule.evaluate(metrics) for rule in rules)
    return {"score": satisfied / total_weight, "details": sum(metrics.values())}


def deterministic_auction(bids: Dict[str, float]) -> Dict[str, object]:
    ordered = sorted(bids.items(), key=lambda kv: (-kv[1], kv[0]))
    winner, price = ordered[0]
    clearing_price = ordered[1][1] if len(ordered) > 1 else price
    return {"winner": winner, "clearing_price": clearing_price, "ordered": ordered}
