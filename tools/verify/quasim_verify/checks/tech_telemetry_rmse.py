"""Telemetry RMSE validation check (TECH-002).

Validates RMSE between QuASIM predictions and SpaceX/NASA public telemetry.
"""

from typing import Any

import numpy as np
import pandas as pd

from ..models import CheckResult


def rmse(a, b):
    """Calculate root mean squared error."""

    a, b = np.asarray(a), np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run(cfg: dict[str, Any]) -> CheckResult:
    """Run telemetry RMSE validation check.

    Compares QuASIM telemetry predictions against reference data from SpaceX and NASA.

    Args:
        cfg: Configuration dictionary containing:
            - inputs.telemetry.spacex_csv: Path to SpaceX telemetry CSV
            - inputs.telemetry.nasa_csv: Path to NASA telemetry CSV
            - policy.tolerances.rmse_max: Maximum allowed RMSE

    Returns:
        CheckResult with pass/fail status and RMSE values
    """

    tol = cfg["policy"]["tolerances"]["rmse_max"]
    paths = cfg["inputs"]["telemetry"]

    try:
        f9 = pd.read_csv(paths["spacex_csv"])
        orion = pd.read_csv(paths["nasa_csv"])

        # Expect columns: time_s, altitude_km, altitude_km_ref (or similar reference)
        # Check which columns exist
        f9_cols = f9.columns.tolist()
        orion_cols = orion.columns.tolist()

        rmse_results = {}

        # SpaceX F9 validation
        if "altitude_km" in f9_cols and "altitude_km_ref" in f9_cols:
            r1 = rmse(f9["altitude_km"], f9["altitude_km_ref"])
            rmse_results["spacex_f9"] = r1
        else:
            rmse_results["spacex_f9_error"] = f"Missing columns. Found: {f9_cols}"

        # NASA Orion validation
        if "altitude_km" in orion_cols and "altitude_km_ref" in orion_cols:
            r2 = rmse(orion["altitude_km"], orion["altitude_km_ref"])
            rmse_results["nasa_orion"] = r2
        else:
            rmse_results["nasa_orion_error"] = f"Missing columns. Found: {orion_cols}"

        # Check if we have valid RMSE values
        has_errors = any("error" in k for k in rmse_results.keys())
        if has_errors:
            return CheckResult(
                id="TECH-002",
                passed=False,
                details={"errors": rmse_results, "tolerance": tol},
                evidence_paths=[paths["spacex_csv"], paths["nasa_csv"]],
            )

        # Check if all RMSE values are within tolerance
        ok = all(v <= tol for k, v in rmse_results.items() if not isinstance(v, str))

        return CheckResult(
            id="TECH-002",
            passed=ok,
            details={**rmse_results, "tolerance": tol},
            evidence_paths=[paths["spacex_csv"], paths["nasa_csv"]],
        )

    except Exception as e:
        return CheckResult(id="TECH-002", passed=False, details={"error": str(e)})
