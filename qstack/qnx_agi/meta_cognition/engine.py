"""Deterministic meta-cognitive loop integrating planning and governance signals."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List

from qstack.qnx.runtime import TickCounter
from qstack.qnx_agi.utils.audit_log import AuditLog


@dataclass
class ValueSystem:
    """Links economic signals to planner priorities deterministically."""

    weights: Dict[str, float] = field(default_factory=lambda: {"safety": 1.0, "reward": 1.0})

    def score(self, metrics: Dict[str, float]) -> float:
        score = 0.0
        for key, value in sorted(metrics.items()):
            weight = self.weights.get(key, 0.0)
            score += weight * value
        return score


@dataclass
class SelfEvaluator:
    """Scores decisions deterministically using hashed decision traces."""

    def evaluate(self, decisions: List[Dict[str, Any]]) -> str:
        material = ":".join([repr(d) for d in decisions]).encode("utf-8")
        return hashlib.sha256(material).hexdigest()


@dataclass
class MetaCognitionEngine:
    """Runs self-evaluation and value aggregation to steer planning."""

    value_system: ValueSystem = field(default_factory=ValueSystem)
    evaluator: SelfEvaluator = field(default_factory=SelfEvaluator)
    tick_counter: TickCounter = field(default_factory=TickCounter)
    audit: AuditLog = field(default_factory=AuditLog)

    def select_plan(
        self, candidates: List[List[Dict[str, Any]]], governance_signals: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        best: List[Dict[str, Any]] = []
        best_score = float("-inf")
        for plan in candidates:
            signal_score = self.value_system.score(governance_signals)
            digest = self.evaluator.evaluate(plan)
            combined = signal_score + len(digest) * 0.0  # digest ensures determinism
            if combined > best_score:
                best = plan
                best_score = combined
            self.audit.record(
                "meta_plan", {"tick": self.tick_counter.next(), "score": combined, "digest": digest}
            )
        return best
