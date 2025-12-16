"""Deterministic reward helpers for agents."""

from __future__ import annotations

from typing import Dict, List

from qagents.base import AgentState


def aggregate_rewards(agent_states: List[AgentState]) -> Dict[str, float]:
    return {state.agent_id: sum(state.rewards) for state in agent_states}


def shaped_reward(outcome: Dict[str, float], weights: Dict[str, float]) -> float:
    total = 0.0
    for key, weight in sorted(weights.items()):
        total += outcome.get(key, 0.0) * weight
    return total
