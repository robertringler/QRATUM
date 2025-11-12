"""Compression ratio validation check (TECH-004).

Validates anti-holographic MERA compression ratio meets minimum threshold.
"""

from typing import Any

import numpy as np

from ..models import CheckResult


def run(cfg: dict[str, Any]) -> CheckResult:
    """Run compression ratio validation check.

    Validates that tensor network compression achieves expected ratio.

    Args:
        cfg: Configuration dictionary containing:
            - inputs.artifacts.compression_npz: Path to compression NPZ file
            - policy.tolerances.compression_min_ratio: Minimum compression ratio

    Returns:
        CheckResult with pass/fail status and compression ratio
    """
    min_ratio = cfg["policy"]["tolerances"]["compression_min_ratio"]
    path = cfg["inputs"]["artifacts"]["compression_npz"]

    try:
        data = np.load(path)
        raw = float(data["raw_flops"])
        compr = float(data["compressed_flops"])
        ratio = raw / max(compr, 1e-9)
        ok = ratio >= min_ratio

        return CheckResult(
            id="TECH-004",
            passed=ok,
            details={
                "ratio": ratio,
                "min_required": min_ratio,
                "raw_flops": raw,
                "compressed_flops": compr,
            },
            evidence_paths=[path],
        )

    except Exception as e:
        return CheckResult(id="TECH-004", passed=False, details={"error": str(e)})
