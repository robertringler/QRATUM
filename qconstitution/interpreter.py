"""Interprets constitutional articles against system state."""

from __future__ import annotations

from dataclasses import dataclass

from qconstitution.charter import Charter


@dataclass
class ConstitutionalInterpreter:
    charter: Charter

    def evaluate_constraints(self, subject: str, context: dict[str, object]) -> list[str]:
        violations: list[str] = []
        for article in self.charter.active.articles.articles:
            if not article.applies_to(subject):
                continue
            constraints = article.constraints
            if constraints.get("requires_allowed_syscalls"):
                allowed = context.get("allowed_syscalls", [])
                if not allowed:
                    violations.append(f"{article.article_id}: allowed syscalls required")
            if constraints.get("requires_ledger") and not context.get("ledger_enabled", False):
                violations.append(f"{article.article_id}: ledger logging must be enabled")
        return violations
