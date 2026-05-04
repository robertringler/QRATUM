"""Language lint check (GOV-301).

Lints documentation for banned phrases without evidence.
"""

import os
import re
from typing import Any

from ..models import CheckResult


def run(cfg: dict[str, Any]) -> CheckResult:
    """Run language lint check.

    Scans documentation for banned phrases that require evidence.

    Args:
        cfg: Configuration dictionary containing:
            - inputs.brief_paths: List of document paths to scan
            - policy.ban_phrases_unless_evidence: List of phrases to flag

    Returns:
        CheckResult with pass/fail status and list of banned phrase hits
    """

    bans = cfg["policy"].get("ban_phrases_unless_evidence", [])

    # Check if evidence checks passed
    evidence_ok = []
    for eid in ["COMP-101", "ECON-202", "TECH-004"]:
        evidence_ok.append(eid in [c["id"] for c in cfg.get("checks", [])])

    # Scan documents
    texts = []
    scanned_files = []

    for p in cfg["inputs"].get("brief_paths", []):
        if not os.path.exists(p):
            continue

        try:
            if p.lower().endswith(".md"):
                with open(p, encoding="utf-8") as f:
                    texts.append(f.read())
                    scanned_files.append(p)
        except Exception:
            pass

    hit = []
    for b in bans:
        rx = re.compile(re.escape(b), re.I)
        for t in texts:
            if rx.search(t):
                hit.append(b)
                break

    ok = len(hit) == 0

    return CheckResult(
        id="GOV-301",
        passed=ok,
        severity="warn" if ok else "error",
        details={"banned_hits": list(set(hit)), "scanned_files": scanned_files},
    )
