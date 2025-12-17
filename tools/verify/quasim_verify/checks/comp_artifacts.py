"""Compliance artifacts check (COMP-101).

Validates presence of required DO-178C compliance artifacts.
"""

import os
from typing import Any

from ..models import CheckResult


def run(cfg: dict[str, Any]) -> CheckResult:
    """Run compliance artifacts check.

    Validates that required DO-178C Level A artifacts exist.

    Args:
        cfg: Configuration dictionary containing:
            - inputs.compliance: Dict with paths to PSAC, SAS, DER documents
            - policy.require_der_for_level_a: Whether DER is required

    Returns:
        CheckResult with pass/fail status and missing artifacts list
    """

    c = cfg["inputs"]["compliance"]
    need = ["psac_id", "sas_id", "der_letter"]
    missing = [k for k in need if not os.path.exists(c.get(k, ""))]

    require_der = cfg["policy"].get("require_der_for_level_a", True)
    ok = (len(missing) == 0) or (not require_der)

    evidence_paths = [c[k] for k in need if k in c and os.path.exists(c[k])]

    return CheckResult(
        id="COMP-101",
        passed=ok,
        details={
            "missing": missing,
            "require_der": require_der,
            "checked_artifacts": need,
        },
        evidence_paths=evidence_paths,
    )
