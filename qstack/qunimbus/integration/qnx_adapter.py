"""Operator adapter for QNX runtime."""

from __future__ import annotations

from typing import Dict


def valuation_operator(engine):
    from ..core.engine import ValuationInput

    def op(state, goal: Dict[str, float]):
        return engine.evaluate(ValuationInput(metrics=goal))

    return op


def governance_operator(engine):
    from ..core.engine import ValuationInput

    def op(state, goal: Dict[str, float]):
        return engine.governance_score(ValuationInput(metrics=goal))

    return op
