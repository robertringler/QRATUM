"""Constitutional principles enforced across Q-Stack operations."""

from __future__ import annotations

from dataclasses import dataclass

ARTICLE_QNX_SAFETY = "ARTICLE_QNX_SAFETY"
ARTICLE_QUASIM_BOUNDS = "ARTICLE_QUASIM_BOUNDS"
ARTICLE_QUNIMBUS_GOVERNANCE = "ARTICLE_QUNIMBUS_GOVERNANCE"


@dataclass(frozen=True)
class ConstitutionalArticle:
    """Declarative article that encodes an alignment principle."""

    article_id: str
    description: str
    applies_to: set[str]

    def is_applicable(self, operation: str) -> bool:
        return operation in self.applies_to or "*" in self.applies_to


@dataclass(frozen=True)
class Constitution:
    """Collection of constitutional articles."""

    articles: list[ConstitutionalArticle]

    def applicable_articles(self, operation: str) -> list[ConstitutionalArticle]:
        return [article for article in self.articles if article.is_applicable(operation)]

    def describe(self) -> list[dict]:
        return [
            {
                "article_id": article.article_id,
                "description": article.description,
                "applies_to": sorted(article.applies_to),
            }
            for article in self.articles
        ]


DEFAULT_CONSTITUTION = Constitution(
    articles=[
        ConstitutionalArticle(
            article_id=ARTICLE_QNX_SAFETY,
            description="QNX operations must maintain positive timesteps and a defined security envelope.",
            applies_to={"qnx.lifecycle", "qnx.simulation"},
        ),
        ConstitutionalArticle(
            article_id=ARTICLE_QUASIM_BOUNDS,
            description="QuASIM simulations must run within bounded, declared workspace and precision settings.",
            applies_to={"quasim.simulation"},
        ),
        ConstitutionalArticle(
            article_id=ARTICLE_QUNIMBUS_GOVERNANCE,
            description="QuNimbus scenarios require node governance to remain enabled for evaluation.",
            applies_to={"qunimbus.synthetic_market"},
        ),
    ]
)
