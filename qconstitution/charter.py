"""Constitutional charter definitions."""
from __future__ import annotations

from dataclasses import dataclass

from qconstitution.articles import ArticleSet, ConstitutionalArticle


@dataclass(frozen=True)
class ConstitutionalVersion:
    version_id: str
    enacted_tick: int
    articles: ArticleSet


@dataclass
class Charter:
    active: ConstitutionalVersion

    def article_index(self) -> dict[str, ConstitutionalArticle]:
        return {art.article_id: art for art in self.active.articles.articles}

    def describe(self) -> dict[str, object]:
        return {"version_id": self.active.version_id, "enacted_tick": self.active.enacted_tick, "articles": self.active.articles.as_dict()}


def default_charter() -> Charter:
    articles = ArticleSet(
        [
            ConstitutionalArticle(
                article_id="safety-envelope",
                description="Nodes must declare allowed syscalls and remain within them.",
                constraints={"scope": ["node"], "requires_allowed_syscalls": True},
            ),
            ConstitutionalArticle(
                article_id="audit-ledger",
                description="All violations must be recorded to the ledger.",
                constraints={"scope": ["node", "cluster"], "requires_ledger": True},
            ),
        ]
    )
    return Charter(active=ConstitutionalVersion(version_id="v1", enacted_tick=0, articles=articles))
