"""Deterministic strategy helpers for agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qagents.base import AgentObservation, AgentState


@dataclass
class StrategyDecision:
    action: dict[str, Any]
    justification: str


class Strategy:
    """Base strategy API."""

    name: str = "base"

    def decide(self, observation: AgentObservation, state: AgentState) -> StrategyDecision:
        raise NotImplementedError


@dataclass
class ThresholdStrategy(Strategy):
    """Deterministic threshold trigger for a single metric."""

    metric: str
    threshold: float
    above_action: dict[str, Any]
    below_action: dict[str, Any]
    name: str = "threshold"

    def decide(self, observation: AgentObservation, state: AgentState) -> StrategyDecision:
        value = observation.view.get(self.metric, 0)
        if value >= self.threshold:
            return StrategyDecision(action=dict(self.above_action), justification="above_threshold")
        return StrategyDecision(action=dict(self.below_action), justification="below_threshold")


@dataclass
class ScriptedStrategy(Strategy):
    """Replay a deterministic script of actions keyed by tick."""

    script: dict[int, dict[str, Any]]
    default_action: dict[str, Any]
    name: str = "scripted"

    def decide(self, observation: AgentObservation, state: AgentState) -> StrategyDecision:
        action = self.script.get(observation.tick, self.default_action)
        return StrategyDecision(action=dict(action), justification="scripted")


class PolicyAdapter:
    """Adapter to wrap StrategyDecision into an AgentPolicy-compatible callable."""

    def __init__(self, strategy: Strategy):
        self.strategy = strategy

    def decide(self, observation: AgentObservation, state: AgentState) -> dict[str, Any]:
        decision = self.strategy.decide(observation, state)
        return {
            "action": decision.action,
            "reason": decision.justification,
            "strategy": self.strategy.name,
        }


class DeterminismChecker:
    """Helper to assert that strategies return stable outputs for identical inputs."""

    def __init__(self, strategy: Strategy):
        self.strategy = strategy

    def is_deterministic(self, observation: AgentObservation, state: AgentState) -> bool:
        first = self.strategy.decide(observation, state)
        second = self.strategy.decide(observation, state)
        return first == second
