"""Adapter constructing QNX operators for QuASIM simulations."""

from __future__ import annotations

from typing import Dict

from ..core.engine import SimulationEngine, build_default_engine
from ..domains.aerospace import evaluate_flight
from ..domains.finance import evaluate_portfolio
from ..domains.pharma import evaluate_trial


def build_qnx_operator_library(engine: SimulationEngine | None = None) -> Dict[str, callable]:
    engine = engine or build_default_engine()
    # Ensure essential kernels are present even if a bare engine is supplied.
    engine.register_kernel("telemetry", engine.telemetry_kernel)
    engine.register_kernel("finance", engine.finance_kernel)
    engine.register_kernel("pharma", engine.pharma_kernel)

    def aero(state, goal):
        telemetry = goal.get("telemetry", {})
        return engine.run_kernel("telemetry", telemetry)

    def finance(state, goal):
        snapshot = goal.get("snapshot", {})
        return engine.run_kernel("finance", snapshot)

    def pharma(state, goal):
        species = goal.get("species", {})
        return engine.run_kernel("pharma", species)

    return {
        "aero_eval": aero,
        "finance_eval": finance,
        "pharma_eval": pharma,
        "aero_report": lambda state, goal: evaluate_flight(goal),
        "finance_report": lambda state, goal: evaluate_portfolio(goal),
        "pharma_report": lambda state, goal: evaluate_trial(goal),
    }
