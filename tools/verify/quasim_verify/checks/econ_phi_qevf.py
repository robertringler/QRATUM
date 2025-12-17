"""Φ_QEVF validation check (ECON-201).

Validates quantum economic value function parameters and calculation.
"""

from typing import Any

import yaml

from ..models import CheckResult


def run(cfg: dict[str, Any]) -> CheckResult:
    """Run Φ_QEVF validation check.

    Validates that quantum economic value function is properly parameterized.

    Args:
        cfg: Configuration dictionary containing:
            - inputs.economics.phi_inputs_yaml: Path to Φ_QEVF inputs YAML

    Returns:
        CheckResult with pass/fail status and Φ_QEVF value
    """

    y = cfg["inputs"]["economics"]["phi_inputs_yaml"]

    try:
        with open(y) as f:
            d = yaml.safe_load(f)

        base = float(d["base_value_per_eph"])
        eta = float(d["eta_ent"])
        eta0 = float(d["eta_baseline"])
        penalty = float(d["coherence_penalty"])
        runtime = float(d["runtime_factor"])
        market = float(d["market_multiplier"])

        phi = base * (eta / eta0) * penalty * runtime * market

        # Validate parameters are reasonable
        ok = base > 0 and 0.8 <= (eta / eta0) <= 1.2 and penalty > 0 and runtime > 0

        return CheckResult(
            id="ECON-201",
            passed=ok,
            details={"phi_qevf": phi, "inputs": d},
            evidence_paths=[y],
        )

    except Exception as e:
        return CheckResult(id="ECON-201", passed=False, details={"error": str(e)})
