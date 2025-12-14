"""Campaign reporting."""
from __future__ import annotations

from dataclasses import dataclass

from qcampaign.scoring import CampaignScorecard


@dataclass
class CampaignReport:
    name: str
    scores: dict[str, float]
    highlights: list[str]

    @classmethod
    def from_scorecard(cls, campaign, scorecard: CampaignScorecard) -> "CampaignReport":
        highlights = [f"{team}: score={score}" for team, score in sorted(scorecard.team_scores.items())]
        return cls(name=campaign.name, scores=scorecard.team_scores, highlights=highlights)

    def summary(self) -> dict[str, object]:
        return {"name": self.name, "scores": dict(self.scores), "highlights": list(self.highlights)}
