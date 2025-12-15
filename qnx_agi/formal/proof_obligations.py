"""Proof obligations for constitutional versions."""

from __future__ import annotations

from typing import Dict, List


def obligations(version_id: str) -> List[str]:
    base = ["safety-envelope-satisfied", "ledger-audit-enabled"]
    return [f"{version_id}:{ob}" for ob in base]


def fulfilled(version_id: str, evidence: Dict[str, bool]) -> bool:
    return all(evidence.get(req, False) for req in obligations(version_id))
