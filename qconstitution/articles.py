"""Constitutional articles."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConstitutionalArticle:
    article_id: str
    description: str
    constraints: dict[str, object] = field(default_factory=dict)

    def applies_to(self, subject: str) -> bool:
        scope = self.constraints.get("scope", [])
        return not scope or subject in scope


@dataclass
class ArticleSet:
    articles: list[ConstitutionalArticle] = field(default_factory=list)

    def add(self, article: ConstitutionalArticle) -> None:
        self.articles.append(article)

    def get(self, article_id: str) -> ConstitutionalArticle:
        for art in self.articles:
            if art.article_id == article_id:
                return art
        raise KeyError(article_id)

    def as_dict(self) -> dict[str, dict[str, object]]:
        return {art.article_id: {"description": art.description, "constraints": art.constraints} for art in self.articles}
