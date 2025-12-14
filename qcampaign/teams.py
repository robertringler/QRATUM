"""Red/Blue/Gray team abstractions."""
from __future__ import annotations

from dataclasses import dataclass, field

from qagents.base import AgentObservation, Agent


@dataclass
class Team:
    name: str
    agents: list[Agent] = field(default_factory=list)

    def decide(self, tick: int, scenario_view: dict[str, object]) -> list[dict[str, object]]:
        observation = AgentObservation(tick=tick, view=scenario_view, provenance=self.name)
        decisions: list[dict[str, object]] = []
        for agent in sorted(self.agents, key=lambda a: a.agent_id):
            decisions.append(agent.act(observation))
        return decisions


class RedTeam(Team):
    pass


class BlueTeam(Team):
    pass


class GrayTeam(Team):
    pass
