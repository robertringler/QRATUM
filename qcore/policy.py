"""Q-Core Policy Evaluation Engine.

This module implements policy evaluation for intents and contracts.
Policies define rules and constraints that govern authorization.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from qil.ast import Intent


@dataclass
class PolicyRule:
    """Represents a single policy rule.

    Attributes:
        name: Rule name
        description: Rule description
        predicate: Function that evaluates the rule
        severity: Severity level (error, warning, info)
    """

    name: str
    description: str
    predicate: Callable[[Intent], bool]
    severity: str = "error"


@dataclass
class PolicyEvaluationResult:
    """Result of policy evaluation.

    Attributes:
        passed: Whether all policies passed
        violations: List of violated rules
        warnings: List of warnings
        info: List of informational messages
    """

    passed: bool
    violations: list[str]
    warnings: list[str]
    info: list[str]


class PolicyEngine:
    """Policy evaluation engine for intents."""

    def __init__(self) -> None:
        """Initialize policy engine with default rules."""
        self.rules: list[PolicyRule] = []
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default policy rules."""
        # Rule: Intent must have an objective
        self.add_rule(
            PolicyRule(
                name="require_objective",
                description="Intent must have an objective",
                predicate=lambda intent: intent.objective is not None,
                severity="error",
            )
        )

        # Rule: Intent must have at least one authority
        self.add_rule(
            PolicyRule(
                name="require_authority",
                description="Intent must specify at least one authority",
                predicate=lambda intent: len(intent.authorities) > 0,
                severity="error",
            )
        )

        # Rule: Untrusted intents must not access sensitive resources
        self.add_rule(
            PolicyRule(
                name="untrusted_restrictions",
                description="Untrusted intents must run in sandbox",
                predicate=lambda intent: (
                    not intent.trust
                    or intent.trust.level != "untrusted"
                    or "sandbox" in [c.name for c in intent.capabilities]
                ),
                severity="error",
            )
        )

        # Rule: Time constraints should be reasonable
        self.add_rule(
            PolicyRule(
                name="reasonable_time_constraints",
                description="Time constraints should be > 0 and < 1 week",
                predicate=lambda intent: all(
                    0 < t.to_seconds() < 7 * 24 * 3600 for t in intent.time_specs
                ),
                severity="warning",
            )
        )

    def add_rule(self, rule: PolicyRule) -> None:
        """Add a policy rule to the engine.

        Args:
            rule: PolicyRule to add
        """
        self.rules.append(rule)

    def remove_rule(self, rule_name: str) -> None:
        """Remove a policy rule by name.

        Args:
            rule_name: Name of rule to remove
        """
        self.rules = [r for r in self.rules if r.name != rule_name]

    def evaluate(self, intent: Intent) -> PolicyEvaluationResult:
        """Evaluate an intent against all policy rules.

        Args:
            intent: Intent to evaluate

        Returns:
            PolicyEvaluationResult with evaluation outcome
        """
        violations: list[str] = []
        warnings: list[str] = []
        info: list[str] = []

        for rule in self.rules:
            try:
                if not rule.predicate(intent):
                    message = f"{rule.name}: {rule.description}"
                    if rule.severity == "error":
                        violations.append(message)
                    elif rule.severity == "warning":
                        warnings.append(message)
                    else:
                        info.append(message)
            except Exception as e:
                # Rule evaluation failed
                violations.append(f"{rule.name}: Rule evaluation failed: {e}")

        passed = len(violations) == 0

        return PolicyEvaluationResult(
            passed=passed,
            violations=violations,
            warnings=warnings,
            info=info,
        )

    def evaluate_constraint(
        self, constraint_name: str, constraint_value: Any, actual_value: Any
    ) -> bool:
        """Evaluate a single constraint.

        Args:
            constraint_name: Name of constraint
            constraint_value: Expected constraint value
            actual_value: Actual value to check

        Returns:
            True if constraint is satisfied, False otherwise
        """
        # Simple equality check for now
        return constraint_value == actual_value


def create_custom_policy_rule(
    name: str,
    description: str,
    predicate: Callable[[Intent], bool],
    severity: str = "error",
) -> PolicyRule:
    """Create a custom policy rule.

    Args:
        name: Rule name
        description: Rule description
        predicate: Function that evaluates the rule
        severity: Severity level (error, warning, info)

    Returns:
        PolicyRule instance
    """
    return PolicyRule(
        name=name,
        description=description,
        predicate=predicate,
        severity=severity,
    )


def evaluate_hardware_policy(
    intent: Intent, available_clusters: list[str]
) -> PolicyEvaluationResult:
    """Evaluate hardware policy for an intent.

    Args:
        intent: Intent to evaluate
        available_clusters: List of available cluster types

    Returns:
        PolicyEvaluationResult
    """
    violations: list[str] = []

    if not intent.hardware:
        # No hardware requirements specified
        return PolicyEvaluationResult(
            passed=True,
            violations=[],
            warnings=[],
            info=["No hardware requirements specified"],
        )

    # Check ONLY clause
    if intent.hardware.only_clusters:
        unavailable = [c for c in intent.hardware.only_clusters if c not in available_clusters]
        if unavailable:
            violations.append(f"Required clusters not available: {', '.join(unavailable)}")

    # Check NOT clause
    if intent.hardware.not_clusters:
        # Ensure at least one available cluster after exclusions
        remaining = [c for c in available_clusters if c not in intent.hardware.not_clusters]
        if not remaining:
            violations.append("All available clusters are excluded by NOT clause")

    passed = len(violations) == 0

    return PolicyEvaluationResult(
        passed=passed,
        violations=violations,
        warnings=[],
        info=[],
    )
