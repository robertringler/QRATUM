"""Deterministic agent primitives for Q-Stack multi-agent simulations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class AgentObservation:
    """A deterministic snapshot of what an agent can see."""

    tick: int
    view: dict[str, Any]
    provenance: str = ""


@dataclass
class AgentState:
    """Mutable per-agent state tracked during simulations."""

    agent_id: str
    memory: dict[str, Any] = field(default_factory=dict)
    rewards: list[float] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    observations: list[AgentObservation] = field(default_factory=list)

    def remember(self, key: str, value: Any) -> None:
        self.memory[key] = value

    def add_reward(self, reward: float) -> float:
        self.rewards.append(reward)
        return sum(self.rewards)

    def record_action(self, action: dict[str, Any]) -> None:
        self.actions.append(action)

    def record_observation(self, observation: AgentObservation) -> None:
        self.observations.append(observation)


class AgentPolicy(Protocol):
    """Policy interface producing deterministic actions."""

    def decide(self, observation: AgentObservation, state: AgentState) -> dict[str, Any]:
        ...


@dataclass
class Agent:
    """Agent wrapper that binds a policy to deterministic state updates."""

    agent_id: str
    policy: AgentPolicy
    state: AgentState = field(init=False)

    def __post_init__(self) -> None:
        self.state = AgentState(agent_id=self.agent_id)

    def observe(self, observation: AgentObservation) -> None:
        self.state.record_observation(observation)
        self.state.remember("last_view", observation.view)

    def act(self, observation: AgentObservation) -> dict[str, Any]:
        self.observe(observation)
        action = self.policy.decide(observation, self.state)
        self.state.record_action(action)
        return action

    def reward(self, value: float) -> float:
        return self.state.add_reward(value)


@dataclass
class AgentLog:
    """Captures agent trajectories for reporting and replay."""

    agent_id: str
    actions: list[dict[str, Any]]
    observations: list[AgentObservation]
    rewards: list[float]

    @classmethod
    def from_state(cls, state: AgentState) -> "AgentLog":
        return cls(
            agent_id=state.agent_id,
            actions=list(state.actions),
            observations=list(state.observations),
            rewards=list(state.rewards),
        )


class LambdaPolicy:
    """Utility policy that defers to a pure function for decisions."""

    def __init__(self, fn: Callable[[AgentObservation, AgentState], dict[str, Any]]):
        self.fn = fn

    def decide(self, observation: AgentObservation, state: AgentState) -> dict[str, Any]:
        return self.fn(observation, state)
