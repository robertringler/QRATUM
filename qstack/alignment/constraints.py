"""Low-level alignment constraints for Q-Stack subsystems."""
from __future__ import annotations

from typing import List, Tuple

from qstack.alignment.constitution import (
    ARTICLE_QNX_SAFETY,
    ARTICLE_QUASIM_BOUNDS,
    ARTICLE_QUNIMBUS_GOVERNANCE,
)
from qstack.alignment.violations import ViolationSeverity
from qstack.config import QNXConfig, QuASIMConfig, QuNimbusConfig

ConstraintFinding = Tuple[str, str, ViolationSeverity]


def check_qnx_config(config: QNXConfig) -> List[ConstraintFinding]:
    findings: List[ConstraintFinding] = []
    if config.timesteps <= 0:
        findings.append(
            (
                ARTICLE_QNX_SAFETY,
                "QNX timesteps must be greater than zero to maintain safety envelope.",
                ViolationSeverity.FATAL,
            )
        )
    if not config.security_level:
        findings.append(
            (
                ARTICLE_QNX_SAFETY,
                "QNX security_level must be set for runtime safety.",
                ViolationSeverity.ERROR,
            )
        )
    return findings


def check_quasim_config(config: QuASIMConfig) -> List[ConstraintFinding]:
    findings: List[ConstraintFinding] = []
    if config.max_workspace_mb <= 0:
        findings.append(
            (
                ARTICLE_QUASIM_BOUNDS,
                "QuASIM max_workspace_mb must be positive to bound runtime memory.",
                ViolationSeverity.FATAL,
            )
        )
    if not config.backend:
        findings.append(
            (
                ARTICLE_QUASIM_BOUNDS,
                "QuASIM backend must be specified for deterministic simulation.",
                ViolationSeverity.ERROR,
            )
        )
    return findings


def check_qunimbus_config(config: QuNimbusConfig) -> List[ConstraintFinding]:
    findings: List[ConstraintFinding] = []
    if not config.enable_node_governance:
        findings.append(
            (
                ARTICLE_QUNIMBUS_GOVERNANCE,
                "QuNimbus node governance must remain enabled for evaluation.",
                ViolationSeverity.FATAL,
            )
        )
    return findings
