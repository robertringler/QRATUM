"""Campaign definitions for multi-agent exercises."""
from __future__ import annotations

from dataclasses import dataclass, field

from qcampaign.teams import Team
from qcampaign.scoring import CampaignScorecard, CampaignScorer
from qcampaign.reporting import CampaignReport


@dataclass
class Campaign:
    name: str
    scenario_config: dict[str, object]
    teams: list[Team]
    timeline_ticks: int
    constraints: dict[str, object] = field(default_factory=dict)
    logs: list[dict[str, object]] = field(default_factory=list)

    def record(self, entry: dict[str, object]) -> None:
        self.logs.append(entry)

    def run(self) -> CampaignScorecard:
        scorer = CampaignScorer()
        for tick in range(self.timeline_ticks):
            for team in sorted(self.teams, key=lambda t: t.name):
                actions = team.decide(tick, self.scenario_config)
                self.record({"tick": tick, "team": team.name, "actions": actions})
        scorecard = scorer.score(self.teams, self.logs)
        return scorecard

    def report(self, scorecard: CampaignScorecard) -> CampaignReport:
        return CampaignReport.from_scorecard(self, scorecard)
