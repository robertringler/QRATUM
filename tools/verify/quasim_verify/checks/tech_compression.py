"""Compression ratio validation check (TECH-004).

Validates anti-holographic MERA compression ratio meets minimum threshold.
Supports both single file and directory of .npz artifacts.
"""

from pathlib import Path
from typing import Any

import numpy as np

from ..models import CheckResult


def run(cfg: dict[str, Any]) -> CheckResult:
    """Run compression ratio validation check.

    Validates that tensor network compression achieves expected ratio.

    Args:
        cfg: Configuration dictionary containing:
            - inputs.artifacts.compression_npz: Path to compression NPZ file or directory
            - policy.tolerances.compression_min_ratio: Minimum compression ratio

    Returns:
        CheckResult with pass/fail status and compression ratio statistics
    """
    min_ratio = cfg["policy"]["tolerances"]["compression_min_ratio"]
    path = Path(cfg["inputs"]["artifacts"]["compression_npz"])

    try:
        # Check if path is a directory or single file
        if path.is_dir():
            return _validate_directory(path, min_ratio)
        else:
            return _validate_single_file(path, min_ratio)

    except Exception as e:
        return CheckResult(id="TECH-004", passed=False, details={"error": str(e)})


def _validate_single_file(path: Path, min_ratio: float) -> CheckResult:
    """Validate a single compression artifact file."""
    data = np.load(path, allow_pickle=True)

    # Try old format first (raw_flops/compressed_flops)
    if "raw_flops" in data and "compressed_flops" in data:
        raw = float(data["raw_flops"])
        compr = float(data["compressed_flops"])
        ratio = raw / max(compr, 1e-9)

    # Try new format (metadata with compression_ratio)
    elif "metadata" in data:
        metadata = data["metadata"]
        if isinstance(metadata, np.ndarray) and len(metadata) > 0:
            metadata = metadata[0]
        ratio = float(metadata.get("compression_ratio", 0.0))

    else:
        return CheckResult(
            id="TECH-004",
            passed=False,
            details={"error": f"Unknown NPZ format in {path}"},
        )

    ok = ratio >= min_ratio

    return CheckResult(
        id="TECH-004",
        passed=ok,
        details={
            "ratio": ratio,
            "min_required": min_ratio,
            "source": "single_file",
        },
        evidence_paths=[str(path)],
    )


def _validate_directory(directory: Path, min_ratio: float) -> CheckResult:
    """Validate compression artifacts from a directory of .npz files."""
    npz_files = list(directory.glob("*.npz"))

    if not npz_files:
        return CheckResult(
            id="TECH-004",
            passed=False,
            details={"error": f"No .npz files found in {directory}"},
        )

    ratios = []
    fidelities = []
    evidence_paths = []

    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)

            # Extract metadata
            if "metadata" in data:
                metadata = data["metadata"]
                if isinstance(metadata, np.ndarray) and len(metadata) > 0:
                    metadata = metadata[0]

                ratio = float(metadata.get("compression_ratio", 0.0))
                fidelity = float(metadata.get("fidelity_achieved", 0.0))

                ratios.append(ratio)
                fidelities.append(fidelity)
                evidence_paths.append(str(npz_file))

        except Exception:
            # Skip files that can't be loaded
            continue

    if not ratios:
        return CheckResult(
            id="TECH-004",
            passed=False,
            details={"error": "No valid compression data found in directory"},
        )

    # Compute aggregate statistics
    ratios_array = np.array(ratios)
    fidelities_array = np.array(fidelities)

    median_ratio = float(np.median(ratios_array))
    mean_ratio = float(np.mean(ratios_array))
    min_ratio_found = float(np.min(ratios_array))
    max_ratio_found = float(np.max(ratios_array))
    std_ratio = float(np.std(ratios_array))

    # Pass if median meets minimum
    ok = median_ratio >= min_ratio

    return CheckResult(
        id="TECH-004",
        passed=ok,
        details={
            "median_ratio": median_ratio,
            "mean_ratio": mean_ratio,
            "min_ratio_found": min_ratio_found,
            "max_ratio_found": max_ratio_found,
            "std_ratio": std_ratio,
            "min_required": min_ratio,
            "n_artifacts": len(ratios),
            "mean_fidelity": float(np.mean(fidelities_array)),
            "source": "directory",
        },
        evidence_paths=evidence_paths[:10],  # Limit to first 10 for brevity
    )
