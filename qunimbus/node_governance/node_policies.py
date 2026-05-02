"""Per-node policy definitions and evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NodePolicyResult:
    compliant: bool
    score: int
    reason: str


def evaluate_node_policies(
    metrics: dict[str, int], expectations: dict[str, int]
) -> NodePolicyResult:
    penalty = 0
    for key, expected in sorted(expectations.items()):
        value = metrics.get(key, 0)
        if value > expected:
            penalty += value - expected
    score = max(0, 100 - penalty)
    compliant = penalty == 0
    reason = "ok" if compliant else f"penalty:{penalty}"
    return NodePolicyResult(compliant=compliant, score=score, reason=reason)


def rank_nodes(nodes: list[tuple[str, NodePolicyResult]]) -> list[tuple[str, int]]:
    return sorted([(name, result.score) for name, result in nodes], key=lambda t: (-t[1], t[0]))
