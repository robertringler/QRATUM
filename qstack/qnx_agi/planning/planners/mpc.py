"""MPC planner wrapper."""

from __future__ import annotations

from typing import Any, Callable

from ..base import MPCPlanner, Planner

__all__ = ["build_mpc"]


def build_mpc(
    predict_fn: Callable[[dict[str, Any]], dict[str, Any]],
    cost_fn: Callable[[dict[str, Any], dict[str, Any]], float],
    horizon: int = 3,
    envelope=None,
) -> Planner:
    return MPCPlanner(predict_fn=predict_fn, cost_fn=cost_fn, envelope=envelope, horizon=horizon)
