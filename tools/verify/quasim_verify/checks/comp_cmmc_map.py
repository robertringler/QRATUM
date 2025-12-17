"""CMMC mapping validation check (COMP-102).

Validates CMMC Level 2 practice mapping completeness.
"""

import os
from typing import Any

import yaml

from ..models import CheckResult


def run(cfg: dict[str, Any]) -> CheckResult:
    """Run CMMC mapping validation check.

    Validates that CMMC Level 2 practice mapping contains required number of practices.

    Args:
        cfg: Configuration dictionary containing:
            - inputs.compliance.cmmc_map: Path to CMMC mapping YAML
            - policy.tolerances.cmmc_practices_min: Minimum practices required (default: 110)

    Returns:
        CheckResult with pass/fail status and practice count
    """

    path = cfg["inputs"]["compliance"]["cmmc_map"]
    min_practices = cfg["policy"]["tolerances"].get("cmmc_practices_min", 110)

    if not os.path.exists(path):
        return CheckResult(
            id="COMP-102", passed=False, details={"error": f"CMMC map not found: {path}"}
        )

    try:
        with open(path) as f:
            y = yaml.safe_load(f)

        practices = y.get("practices", [])
        count = len(practices)
        ok = count >= min_practices

        return CheckResult(
            id="COMP-102",
            passed=ok,
            details={"count": count, "min_required": min_practices},
            evidence_paths=[path],
        )

    except Exception as e:
        return CheckResult(id="COMP-102", passed=False, details={"error": str(e)})
