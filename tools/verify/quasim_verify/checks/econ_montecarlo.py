"""Monte Carlo valuation check (ECON-202).

Validates enterprise valuation via Monte Carlo simulation.
"""

import random
from statistics import median
from typing import Any

import yaml

from ..models import CheckResult


def run(cfg: dict[str, Any]) -> CheckResult:
    """Run Monte Carlo valuation check.

    Runs Monte Carlo simulation to estimate enterprise value P50.

    Args:
        cfg: Configuration dictionary containing:
            - inputs.economics.montecarlo_params_yaml: Path to MC params YAML
            - policy.tolerances.valuation_p50_min: Minimum P50 value
            - policy.tolerances.valuation_p50_max: Maximum P50 value

    Returns:
        CheckResult with pass/fail status and P50 valuation
    """

    try:
        params_path = cfg["inputs"]["economics"]["montecarlo_params_yaml"]
        with open(params_path) as f:
            params = yaml.safe_load(f)

        trials = int(params.get("trials", 5000))
        seed = int(params.get("seed", 42))
        scenarios = params["scenarios"]

        random.seed(seed)
        samples = []

        for _ in range(trials):
            r = random.random()
            acc = 0.0
            for sc in scenarios:
                acc += sc["prob"]
                if r <= acc:
                    samples.append(sc["value"])
                    break

        p50 = float(median(samples))
        lo = float(cfg["policy"]["tolerances"]["valuation_p50_min"])
        hi = float(cfg["policy"]["tolerances"]["valuation_p50_max"])
        ok = lo <= p50 <= hi

        return CheckResult(
            id="ECON-202",
            passed=ok,
            details={
                "p50": p50,
                "bounds": [lo, hi],
                "trials": trials,
                "scenarios": len(scenarios),
            },
            evidence_paths=[params_path],
        )

    except Exception as e:
        return CheckResult(id="ECON-202", passed=False, details={"error": str(e)})
