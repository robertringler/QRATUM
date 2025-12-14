"""Upgrade process for constitutional versions."""
from __future__ import annotations

from dataclasses import dataclass

from qconstitution.articles import ArticleSet, ConstitutionalArticle
from qconstitution.charter import Charter, ConstitutionalVersion


@dataclass(frozen=True)
class ConstitutionalUpgradeProposal:
    proposal_id: str
    proposer: str
    added_articles: list[ConstitutionalArticle]
    target_version: str
    rationale: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "proposer": self.proposer,
            "added_articles": [art.article_id for art in self.added_articles],
            "target_version": self.target_version,
            "rationale": self.rationale,
        }


@dataclass
class UpgradePath:
    threshold: int = 1

    def evaluate(self, proposal: ConstitutionalUpgradeProposal, votes: list[str]) -> bool:
        return len(votes) >= self.threshold

    def apply(self, charter: Charter, proposal: ConstitutionalUpgradeProposal, votes: list[str]) -> Charter:
        if not self.evaluate(proposal, votes):
            raise ValueError("upgrade did not meet threshold")
        merged_articles = ArticleSet(charter.active.articles.articles + proposal.added_articles)
        new_version = ConstitutionalVersion(
            version_id=proposal.target_version,
            enacted_tick=charter.active.enacted_tick + 1,
            articles=merged_articles,
        )
        return Charter(active=new_version)
