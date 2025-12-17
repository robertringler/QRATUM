"""RL convergence validation check (TECH-003).

Validates that reinforcement learning convergence meets minimum threshold.
"""

import json
from typing import Any

from ..models import CheckResult


def run(cfg: dict[str, Any]) -> CheckResult:
    """Run RL convergence validation check.

    Validates that RL training has converged to acceptable level.

    Args:
        cfg: Configuration dictionary containing:
            - inputs.artifacts.rl_convergence_json: Path to RL convergence JSON
            - policy.tolerances.rl_convergence_min: Minimum convergence threshold

    Returns:
        CheckResult with pass/fail status and convergence value
    """

    min_conv = cfg["policy"]["tolerances"]["rl_convergence_min"]
    path = cfg["inputs"]["artifacts"]["rl_convergence_json"]

    try:
        with open(path) as f:
            j = json.load(f)  # expects {"final_convergence": 0.993,...}

        conv = float(j.get("final_convergence", 0.0))
        ok = conv >= min_conv

        return CheckResult(
            id="TECH-003",
            passed=ok,
            details={"convergence": conv, "min_required": min_conv, "metadata": j},
            evidence_paths=[path],
        )

    except Exception as e:
        return CheckResult(id="TECH-003", passed=False, details={"error": str(e)})
