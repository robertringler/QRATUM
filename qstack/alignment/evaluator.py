"""Alignment evaluation engine applying constitutional policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from qstack.alignment.constitution import DEFAULT_CONSTITUTION, Constitution
from qstack.alignment.policies import AlignmentPolicy, default_policies
from qstack.alignment.violations import AlignmentViolation, ViolationSeverity
from qstack.config import QStackConfig


@dataclass
class AlignmentEvaluator:
    """Evaluates alignment policies for Q-Stack operations."""

    config: QStackConfig
    constitution: Constitution = DEFAULT_CONSTITUTION
    policies: list[AlignmentPolicy] = field(default_factory=default_policies)

    def _evaluate(self, operation: str, context: dict, phase: str) -> list[AlignmentViolation]:
        violations: list[AlignmentViolation] = []
        applicable_articles = {
            article.article_id for article in self.constitution.applicable_articles(operation)
        }
        for policy in self.policies:
            policy_violations = policy.evaluate(operation, self.config, context)
            for violation in policy_violations:
                if violation.article_id in applicable_articles:
                    violations.append(violation)
        return violations

    def pre_operation_check(self, operation: str, context: dict) -> list[AlignmentViolation]:
        return self._evaluate(operation, context, phase="pre")

    def post_operation_check(self, operation: str, context: dict) -> list[AlignmentViolation]:
        return self._evaluate(operation, context, phase="post")

    def has_fatal(self, violations: Iterable[AlignmentViolation]) -> bool:
        return any(violation.severity == ViolationSeverity.FATAL for violation in violations)

    def policy_descriptions(self) -> list[dict[str, str]]:
        return [
            {"policy_id": policy.policy_id, "description": policy.description}
            for policy in self.policies
        ]
