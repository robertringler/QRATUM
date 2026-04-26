"""Aerospace governance weights."""

from __future__ import annotations

from ..core.incentives import incentive_budget


def allocate_mission_budget(total: float, participants: dict[str, float]) -> dict[str, float]:
    return incentive_budget(total, participants)
