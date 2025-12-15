"""Campaign scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from qcampaign.teams import Team


@dataclass
class CampaignScorecard:
    team_scores: Dict[str, float]
    logs: List[Dict[str, object]]


class CampaignScorer:
    def score(self, teams: List[Team], logs: List[Dict[str, object]]) -> CampaignScorecard:
        team_rewards = {team.name: 0.0 for team in teams}
        for team in teams:
            team_rewards[team.name] = sum(agent.state.add_reward(0.0) for agent in team.agents)
        # Simple fairness: reward defensive (Blue) for fewer incidents
        incidents = sum(1 for log in logs if log.get("actions"))
        base_bonus = max(0.0, 12.0 - incidents)
        for team in teams:
            if isinstance(team, Team):
                team_rewards[team.name] += base_bonus
        return CampaignScorecard(team_scores=team_rewards, logs=logs)
