"""Observable comparison and validation utilities."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from quasim.io.hdf5 import read_snapshot


def compare_observables(hdf5_path: str, cfg_yaml: str, tol_default: float = 0.03) -> Dict[str, Any]:
    """Compare observables from HDF5 snapshot against expected values.

    Parameters
    ----------
    hdf5_path : str
        Path to HDF5 snapshot file
    cfg_yaml : str
        Path to observables configuration YAML
    tol_default : float
        Default relative tolerance (default: 0.03 = 3%)

    Returns
    -------
    Dict[str, Any]
        Results dictionary with format:
        {name: {"value": float, "expected": float, "pass": bool, "delta": float}}

    Examples
    --------
    >>> results = compare_observables("snapshot.hdf5", "config.yml", 0.03)
    >>> assert all(r["pass"] for r in results.values())
    """

    # Load configuration
    cfg_path = Path(cfg_yaml)
    if not cfg_path.exists():
        return {}

    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    observables = config.get("observables", {})

    # Load snapshot
    try:
        snapshot = read_snapshot(hdf5_path)
    except Exception:
        # Return failing results if snapshot can't be loaded
        return {
            name: {
                "value": 0.0,
                "expected": obs.get("expected", 0.0),
                "pass": False,
                "delta": float("inf"),
                "error": "Failed to load snapshot",
            }
            for name, obs in observables.items()
        }

    results = {}

    for name, obs_config in observables.items():
        source = obs_config.get("source", "")
        reduce_op = obs_config.get("reduce", "mean")
        expected = float(obs_config.get("expected", 0.0))
        tol_abs = obs_config.get("tolerance_abs")
        if tol_abs is not None:
            tol_abs = float(tol_abs)
        tol_rel = float(obs_config.get("tolerance_rel", tol_default))

        # Extract value from snapshot
        value = _extract_observable(snapshot, source, reduce_op)

        # Compute delta and check tolerance
        delta = abs(value - expected)

        if tol_abs is not None:
            passed = delta <= tol_abs
        else:
            # Use relative tolerance
            if expected != 0:
                rel_delta = delta / abs(expected)
                passed = rel_delta <= tol_rel
            else:
                passed = delta == 0

        results[name] = {
            "value": float(value),
            "expected": float(expected),
            "pass": bool(passed),
            "delta": float(delta),
        }

    return results


def _extract_observable(snapshot: Dict[str, Any], source: str, reduce_op: str) -> float:
    """Extract and reduce observable from snapshot data.

    Parameters
    ----------
    snapshot : Dict[str, Any]
        Snapshot data
    source : str
        Source path (e.g., "/agents", "/climate/cells/*/state")
    reduce_op : str
        Reduction operation (e.g., "count", "mean", "sum")

    Returns
    -------
    float
        Computed observable value
    """

    # Handle special operations
    if reduce_op == "count":
        # Count rows in dataset
        for key in snapshot:
            if source.strip("/") in key:
                data = snapshot[key]
                if isinstance(data, np.ndarray):
                    return float(len(data))
        return 0.0

    # Extract data from snapshot
    data = None
    source_key = source.strip("/").split("/")[0]

    for key in snapshot:
        if source_key in key and key != "meta":
            data = snapshot[key]
            break

    if data is None or not isinstance(data, np.ndarray):
        return 0.0

    # Apply reduction
    if reduce_op == "mean":
        return float(np.mean(data))
    elif reduce_op == "sum":
        return float(np.sum(data))
    elif reduce_op == "std":
        return float(np.std(data))
    elif reduce_op.startswith("global_weighted_mean"):
        # Extract field name from operation like "global_weighted_mean(temp)"
        return float(np.mean(data))
    elif reduce_op == "return_ytd":
        # Compute year-to-date return
        if len(data) > 0:
            return float((data[-1] - data[0]) / data[0]) if data[0] != 0 else 0.0
        return 0.0
    else:
        return float(np.mean(data))
