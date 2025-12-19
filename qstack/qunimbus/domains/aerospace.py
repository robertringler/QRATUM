"""Aerospace governance weights."""

from __future__ import annotations

from typing import Dict

from ..core.incentives import incentive_budget


def allocate_mission_budget(total: float, participants: Dict[str, float]) -> Dict[str, float]:
    return incentive_budget(total, participants)
