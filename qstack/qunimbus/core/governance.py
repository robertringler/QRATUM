"""Deterministic governance and voting primitives."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GovernanceRule:
    identifier: str
    threshold: float
    weight: float

    def evaluate(self, metrics: dict[str, float]) -> float:
        score = metrics.get(self.identifier, 0.0)
        return self.weight if score >= self.threshold else 0.0


def vote_outcome(votes: dict[str, int]) -> dict[str, object]:
    tally = sum(votes.values()) or 1
    normalized = {k: v / tally for k, v in votes.items()}
    winner = sorted(normalized.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return {"winner": winner[0], "share": winner[1], "distribution": normalized}


def governance_score(rules: list[GovernanceRule], metrics: dict[str, float]) -> dict[str, float]:
    total_weight = sum(rule.weight for rule in rules) or 1.0
    satisfied = sum(rule.evaluate(metrics) for rule in rules)
    return {"score": satisfied / total_weight, "details": sum(metrics.values())}


def deterministic_auction(bids: dict[str, float]) -> dict[str, object]:
    ordered = sorted(bids.items(), key=lambda kv: (-kv[1], kv[0]))
    winner, price = ordered[0]
    clearing_price = ordered[1][1] if len(ordered) > 1 else price
    return {"winner": winner, "clearing_price": clearing_price, "ordered": ordered}
