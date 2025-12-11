"""Incentive budgeting."""
from __future__ import annotations


def incentive_budget(total: float, participants: dict[str, float]) -> dict[str, float]:
    weight_sum = sum(participants.values()) or 1.0
    return {name: total * weight / weight_sum for name, weight in participants.items()}
