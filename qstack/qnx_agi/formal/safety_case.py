"""Safety case artifact generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SafetyClaim:
    identifier: str
    statement: str
    evidence: List[str]


@dataclass
class SafetyCase:
    system: str
    claims: List[SafetyClaim] = field(default_factory=list)

    def add_claim(self, claim: SafetyClaim) -> None:
        self.claims.append(claim)

    def render(self) -> Dict[str, object]:
        return {
            "system": self.system,
            "claims": [
                {"id": c.identifier, "statement": c.statement, "evidence": list(c.evidence)}
                for c in self.claims
            ],
        }
