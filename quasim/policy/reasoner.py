"""Policy Reasoner for QuASIM Phase VIII.

Implements rule-based compliance layer for automated configuration
mutation approval/rejection based on DO-178C, NIST 800-53, CMMC 2.0,
and ISO 27001 policies.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from quasim.audit.log import audit_event


class PolicyDecision(Enum):
    """Policy decision outcomes."""

    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"


class PolicyFramework(Enum):
    """Supported compliance frameworks."""

    DO_178C = "DO-178C"
    NIST_800_53 = "NIST-800-53"
    CMMC_2_0 = "CMMC-2.0"
    ISO_27001 = "ISO-27001"


@dataclass
class PolicyRule:
    """Compliance policy rule."""

    rule_id: str
    framework: PolicyFramework
    description: str
    condition: str
    action: PolicyDecision
    severity: str  # "critical", "high", "medium", "low"


@dataclass
class ConfigurationMutation:
    """Proposed configuration mutation."""

    parameter: str
    current_value: Any
    proposed_value: Any
    rationale: str
    requestor: str


@dataclass
class PolicyEvaluation:
    """Policy evaluation result."""

    mutation: ConfigurationMutation
    decision: PolicyDecision
    violated_rules: List[PolicyRule]
    approved_by: List[str]
    conditions: List[str]
    timestamp: str


class PolicyReasoner:
    """Logic-based policy reasoning engine.

    Encodes compliance policies from multiple frameworks and automatically
    evaluates configuration mutations for approval or rejection.
    """

    def __init__(self):
        """Initialize the Policy Reasoner with compliance rules."""

        self.rules: List[PolicyRule] = []
        self._initialize_rules()

    def _initialize_rules(self) -> None:
        """Initialize compliance policy rules."""

        # DO-178C Level A Rules
        self.rules.extend(
            [
                PolicyRule(
                    rule_id="DO178C-001",
                    framework=PolicyFramework.DO_178C,
                    description="Safety-critical parameters require MC/DC testing",
                    condition="safety_critical_change",
                    action=PolicyDecision.CONDITIONAL,
                    severity="critical",
                ),
                PolicyRule(
                    rule_id="DO178C-002",
                    framework=PolicyFramework.DO_178C,
                    description="Changes must maintain deterministic behavior",
                    condition="non_deterministic_change",
                    action=PolicyDecision.REJECTED,
                    severity="critical",
                ),
                PolicyRule(
                    rule_id="DO178C-003",
                    framework=PolicyFramework.DO_178C,
                    description="Traceability matrix must be updated",
                    condition="traceability_required",
                    action=PolicyDecision.CONDITIONAL,
                    severity="high",
                ),
            ]
        )

        # NIST 800-53 Rules
        self.rules.extend(
            [
                PolicyRule(
                    rule_id="NIST-AC-01",
                    framework=PolicyFramework.NIST_800_53,
                    description="Access control changes require authorization",
                    condition="access_control_change",
                    action=PolicyDecision.CONDITIONAL,
                    severity="high",
                ),
                PolicyRule(
                    rule_id="NIST-AU-02",
                    framework=PolicyFramework.NIST_800_53,
                    description="Audit configuration changes must be logged",
                    condition="audit_config_change",
                    action=PolicyDecision.CONDITIONAL,
                    severity="high",
                ),
                PolicyRule(
                    rule_id="NIST-CM-02",
                    framework=PolicyFramework.NIST_800_53,
                    description="Configuration baseline must be maintained",
                    condition="baseline_deviation",
                    action=PolicyDecision.REJECTED,
                    severity="medium",
                ),
            ]
        )

        # CMMC 2.0 Level 2 Rules
        self.rules.extend(
            [
                PolicyRule(
                    rule_id="CMMC-AC-L2-001",
                    framework=PolicyFramework.CMMC_2_0,
                    description="CUI handling changes require review",
                    condition="cui_handling_change",
                    action=PolicyDecision.CONDITIONAL,
                    severity="high",
                ),
                PolicyRule(
                    rule_id="CMMC-SC-L2-001",
                    framework=PolicyFramework.CMMC_2_0,
                    description="Cryptographic changes require validation",
                    condition="crypto_change",
                    action=PolicyDecision.CONDITIONAL,
                    severity="critical",
                ),
            ]
        )

        # ISO 27001 Rules
        self.rules.extend(
            [
                PolicyRule(
                    rule_id="ISO27001-A5.1",
                    framework=PolicyFramework.ISO_27001,
                    description="Information security policy compliance required",
                    condition="security_policy_change",
                    action=PolicyDecision.CONDITIONAL,
                    severity="high",
                ),
                PolicyRule(
                    rule_id="ISO27001-A8.1",
                    framework=PolicyFramework.ISO_27001,
                    description="Asset management changes require documentation",
                    condition="asset_management_change",
                    action=PolicyDecision.CONDITIONAL,
                    severity="medium",
                ),
            ]
        )

    def evaluate_mutation(
        self, mutation: ConfigurationMutation, context: Optional[Dict[str, Any]] = None
    ) -> PolicyEvaluation:
        """Evaluate a configuration mutation against compliance policies.

        Parameters
        ----------
        mutation : ConfigurationMutation
            Proposed configuration change
        context : Optional[Dict[str, Any]]
            Additional context for evaluation

        Returns
        -------
        PolicyEvaluation
            Evaluation result with decision and conditions
        """

        context = context or {}
        violated_rules = []
        conditions = []
        decision = PolicyDecision.APPROVED

        # Evaluate against all rules
        for rule in self.rules:
            if self._check_condition(rule.condition, mutation, context):
                violated_rules.append(rule)

                if rule.action == PolicyDecision.REJECTED:
                    decision = PolicyDecision.REJECTED
                    break
                elif rule.action == PolicyDecision.CONDITIONAL:
                    if decision != PolicyDecision.REJECTED:
                        decision = PolicyDecision.CONDITIONAL
                    conditions.append(self._get_condition_text(rule))

        # Determine approval authorities
        approved_by = []
        if decision == PolicyDecision.APPROVED:
            approved_by = ["automated_policy_reasoner"]
        elif decision == PolicyDecision.CONDITIONAL:
            approved_by = self._required_approvers(violated_rules)

        from datetime import datetime, timezone

        evaluation = PolicyEvaluation(
            mutation=mutation,
            decision=decision,
            violated_rules=violated_rules,
            approved_by=approved_by,
            conditions=conditions,
            timestamp=datetime.now(timezone.utc).isoformat() + "Z",
        )

        # Log evaluation
        audit_event(
            "policy_reasoner.evaluation_complete",
            {
                "parameter": mutation.parameter,
                "decision": decision.value,
                "violated_rules": [r.rule_id for r in violated_rules],
                "conditions_count": len(conditions),
            },
        )

        return evaluation

    def _check_condition(
        self, condition: str, mutation: ConfigurationMutation, context: Dict[str, Any]
    ) -> bool:
        """Check if a rule condition is met.

        Parameters
        ----------
        condition : str
            Condition to check
        mutation : ConfigurationMutation
            Proposed mutation
        context : Dict[str, Any]
            Evaluation context

        Returns
        -------
        bool
            True if condition is met
        """

        # Safety-critical changes
        if condition == "safety_critical_change":
            safety_params = [
                "phi_tolerance",
                "error_threshold",
                "safety_margin",
                "critical_timeout",
            ]
            return mutation.parameter in safety_params

        # Non-deterministic changes
        if condition == "non_deterministic_change":
            return (
                "random" in mutation.parameter.lower()
                or "nondeterministic" in str(mutation.rationale).lower()
            )

        # Traceability required
        if condition == "traceability_required":
            return context.get("requires_traceability", True)

        # Access control changes
        if condition == "access_control_change":
            return "auth" in mutation.parameter.lower() or "access" in mutation.parameter.lower()

        # Audit configuration changes
        if condition == "audit_config_change":
            return "audit" in mutation.parameter.lower() or "log" in mutation.parameter.lower()

        # Baseline deviation
        if condition == "baseline_deviation":
            baseline_params = context.get("baseline_params", {})
            if mutation.parameter in baseline_params:
                baseline_value = baseline_params[mutation.parameter]
                # Check if deviation exceeds threshold
                try:
                    deviation = abs(float(mutation.proposed_value) - float(baseline_value)) / float(
                        baseline_value
                    )
                    return deviation > 0.2  # 20% deviation threshold
                except (ValueError, TypeError, ZeroDivisionError):
                    return False
            return False

        # CUI handling changes
        if condition == "cui_handling_change":
            return "cui" in mutation.parameter.lower() or context.get("affects_cui", False)

        # Cryptographic changes
        if condition == "crypto_change":
            return "crypto" in mutation.parameter.lower() or "encrypt" in mutation.parameter.lower()

        # Security policy changes
        if condition == "security_policy_change":
            return "security" in mutation.parameter.lower() or context.get(
                "affects_security_policy", False
            )

        # Asset management changes
        if condition == "asset_management_change":
            return "resource" in mutation.parameter.lower() or "asset" in mutation.parameter.lower()

        return False

    def _get_condition_text(self, rule: PolicyRule) -> str:
        """Get human-readable condition text for a rule."""

        condition_texts = {
            "safety_critical_change": f"MC/DC testing required ({rule.framework.value} {rule.rule_id})",
            "traceability_required": f"Traceability matrix update required ({rule.framework.value} {rule.rule_id})",
            "access_control_change": f"Authorization from security officer required ({rule.framework.value} {rule.rule_id})",
            "audit_config_change": f"Audit trail documentation required ({rule.framework.value} {rule.rule_id})",
            "cui_handling_change": f"CUI handling review required ({rule.framework.value} {rule.rule_id})",
            "crypto_change": f"Cryptographic validation required ({rule.framework.value} {rule.rule_id})",
            "security_policy_change": f"Security policy review required ({rule.framework.value} {rule.rule_id})",
            "asset_management_change": f"Asset documentation update required ({rule.framework.value} {rule.rule_id})",
        }
        return condition_texts.get(
            rule.condition, f"{rule.description} ({rule.framework.value} {rule.rule_id})"
        )

    def _required_approvers(self, violated_rules: List[PolicyRule]) -> List[str]:
        """Determine required approvers based on violated rules."""

        approvers = set()

        for rule in violated_rules:
            if rule.severity == "critical":
                approvers.add("chief_compliance_officer")
                approvers.add("safety_engineer")
            elif rule.severity == "high":
                approvers.add("compliance_officer")

            # Add framework-specific approvers
            if rule.framework == PolicyFramework.DO_178C:
                approvers.add("safety_engineer")
            elif (
                rule.framework == PolicyFramework.NIST_800_53
                or rule.framework == PolicyFramework.CMMC_2_0
            ):
                approvers.add("security_officer")
            elif rule.framework == PolicyFramework.ISO_27001:
                approvers.add("information_security_manager")

        return sorted(approvers)

    def get_rule_by_id(self, rule_id: str) -> Optional[PolicyRule]:
        """Get a policy rule by ID.

        Parameters
        ----------
        rule_id : str
            Rule identifier

        Returns
        -------
        Optional[PolicyRule]
            Policy rule or None
        """

        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None

    def get_rules_by_framework(self, framework: PolicyFramework) -> List[PolicyRule]:
        """Get all rules for a specific framework.

        Parameters
        ----------
        framework : PolicyFramework
            Compliance framework

        Returns
        -------
        List[PolicyRule]
            List of rules
        """

        return [r for r in self.rules if r.framework == framework]

    def get_statistics(self) -> Dict[str, Any]:
        """Get policy reasoner statistics.

        Returns
        -------
        Dict[str, Any]
            Statistics about loaded rules
        """

        return {
            "total_rules": len(self.rules),
            "rules_by_framework": {
                framework.value: len(self.get_rules_by_framework(framework))
                for framework in PolicyFramework
            },
            "rules_by_severity": {
                "critical": len([r for r in self.rules if r.severity == "critical"]),
                "high": len([r for r in self.rules if r.severity == "high"]),
                "medium": len([r for r in self.rules if r.severity == "medium"]),
                "low": len([r for r in self.rules if r.severity == "low"]),
            },
        }
