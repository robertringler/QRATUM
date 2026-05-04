"""Audit chain generation check (DOC-401).

Generates SHA256 audit chain for all evidence files.
"""

import hashlib
import os
from typing import Any

from ..io import sha256_file
from ..models import CheckResult


def run(cfg: dict[str, Any]) -> CheckResult:
    """Run audit chain generation check.

    Creates SHA256 hash chain of all evidence files.

    Args:
        cfg: Configuration dictionary containing:
            - inputs: All input file paths
            - outputs.audit_chain_file: Output path for audit chain

    Returns:
        CheckResult with pass/fail status and audit chain path
    """

    files = []

    # Collect all input files
    try:
        inputs = cfg["inputs"]

        # Brief paths
        files.extend(inputs.get("brief_paths", []))

        # Economics files
        if "economics" in inputs:
            econ = inputs["economics"]
            for key in ["phi_inputs_yaml", "montecarlo_params_yaml"]:
                if key in econ:
                    files.append(econ[key])

        # Artifacts
        if "artifacts" in inputs:
            art = inputs["artifacts"]
            for key in ["rl_convergence_json", "compression_npz"]:
                if key in art:
                    files.append(art[key])

        # Filter to existing files
        existing_files = [f for f in files if os.path.exists(f)]

        chain_path = cfg["outputs"]["audit_chain_file"]
        os.makedirs(os.path.dirname(chain_path), exist_ok=True)

        with open(chain_path, "w") as f:
            f.write("# QuASIM Verification Audit Chain\n")
            f.write("# Format: filepath,sha256,chain_link\n")
            prev = ""
            for p in existing_files:
                h = sha256_file(p)
                link_input = (prev + h).encode("utf-8")
                link = hashlib.sha256(link_input).hexdigest()
                f.write(f"{p},{h},{link}\n")
                prev = link

        return CheckResult(
            id="DOC-401",
            passed=True,
            evidence_paths=[chain_path],
            details={"files_hashed": len(existing_files), "chain_file": chain_path},
        )

    except Exception as e:
        return CheckResult(id="DOC-401", passed=False, details={"error": str(e)})
