"""Red/Blue/Gray team abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from qagents.base import Agent, AgentObservation


@dataclass
class Team:
    name: str
    agents: List[Agent] = field(default_factory=list)

    def decide(self, tick: int, scenario_view: Dict[str, object]) -> List[Dict[str, object]]:
        observation = AgentObservation(tick=tick, view=scenario_view, provenance=self.name)
        decisions: List[Dict[str, object]] = []
        for agent in sorted(self.agents, key=lambda a: a.agent_id):
            decisions.append(agent.act(observation))
        return decisions


class RedTeam(Team):
    pass


class BlueTeam(Team):
    pass


class GrayTeam(Team):
    pass
