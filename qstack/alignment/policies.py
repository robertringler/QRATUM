"""Alignment policies enforcing constitutional constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from qstack.alignment import constraints
from qstack.alignment.constitution import (
    ARTICLE_QNX_SAFETY,
    ARTICLE_QUASIM_BOUNDS,
    ARTICLE_QUNIMBUS_GOVERNANCE,
)
from qstack.alignment.violations import AlignmentViolation, ViolationSeverity
from qstack.config import QStackConfig


@dataclass(frozen=True)
class AlignmentPolicy:
    """Base alignment policy definition."""

    policy_id: str
    description: str

    def evaluate(
        self, operation: str, config: QStackConfig, context: Dict
    ) -> List[AlignmentViolation]:
        raise NotImplementedError


class SafetyFirstPolicy(AlignmentPolicy):
    def __init__(self) -> None:
        super().__init__(
            "policy.safety_first", "Enforces baseline safety invariants across QNX/QuASIM/QuNimbus."
        )

    def evaluate(
        self, operation: str, config: QStackConfig, context: Dict
    ) -> List[AlignmentViolation]:
        violations: List[AlignmentViolation] = []
        if operation.startswith("qnx."):
            for article_id, message, severity in constraints.check_qnx_config(config.qnx):
                violations.append(
                    AlignmentViolation(
                        operation=operation,
                        article_id=article_id,
                        policy_id=self.policy_id,
                        message=message,
                        severity=severity,
                    )
                )
        if operation == "quasim.simulation":
            for article_id, message, severity in constraints.check_quasim_config(config.quasim):
                violations.append(
                    AlignmentViolation(
                        operation=operation,
                        article_id=article_id,
                        policy_id=self.policy_id,
                        message=message,
                        severity=severity,
                    )
                )
        if operation == "qunimbus.synthetic_market":
            for article_id, message, severity in constraints.check_qunimbus_config(config.qunimbus):
                violations.append(
                    AlignmentViolation(
                        operation=operation,
                        article_id=article_id,
                        policy_id=self.policy_id,
                        message=message,
                        severity=severity,
                    )
                )
        return violations


class DeterminismPolicy(AlignmentPolicy):
    def __init__(self) -> None:
        super().__init__(
            "policy.determinism", "Ensures deterministic seeds are configured where applicable."
        )

    def evaluate(
        self, operation: str, config: QStackConfig, context: Dict
    ) -> List[AlignmentViolation]:
        violations: List[AlignmentViolation] = []
        if operation.startswith("qnx.") and config.qnx.seed is None:
            violations.append(
                AlignmentViolation(
                    operation=operation,
                    article_id=ARTICLE_QNX_SAFETY,
                    policy_id=self.policy_id,
                    message="QNX seed must be set for deterministic execution.",
                    severity=ViolationSeverity.ERROR,
                )
            )
        if operation == "quasim.simulation" and config.quasim.seed is None:
            violations.append(
                AlignmentViolation(
                    operation=operation,
                    article_id=ARTICLE_QUASIM_BOUNDS,
                    policy_id=self.policy_id,
                    message="QuASIM seed should be provided for deterministic simulations.",
                    severity=ViolationSeverity.WARNING,
                )
            )
        return violations


class GovernancePolicy(AlignmentPolicy):
    def __init__(self) -> None:
        super().__init__(
            "policy.governance", "Maintains governance enforcement for economic evaluations."
        )

    def evaluate(
        self, operation: str, config: QStackConfig, context: Dict
    ) -> List[AlignmentViolation]:
        violations: List[AlignmentViolation] = []
        if operation == "qunimbus.synthetic_market" and not config.qunimbus.enable_node_governance:
            violations.append(
                AlignmentViolation(
                    operation=operation,
                    article_id=ARTICLE_QUNIMBUS_GOVERNANCE,
                    policy_id=self.policy_id,
                    message="Node governance must remain enabled before running QuNimbus scenarios.",
                    severity=ViolationSeverity.FATAL,
                )
            )
        return violations


def default_policies() -> List[AlignmentPolicy]:
    return [SafetyFirstPolicy(), DeterminismPolicy(), GovernancePolicy()]
