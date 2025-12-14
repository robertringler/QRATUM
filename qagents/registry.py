"""Registry for deterministic agent management."""
from __future__ import annotations


from qagents.base import Agent


class AgentRegistry:
    """Simple deterministic registry keyed by agent_id."""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> None:
        if agent.agent_id in self._agents:
            raise ValueError(f"Agent {agent.agent_id} already registered")
        self._agents[agent.agent_id] = agent

    def get(self, agent_id: str) -> Agent:
        return self._agents[agent_id]

    def all(self):
        return [self._agents[k] for k in sorted(self._agents)]
