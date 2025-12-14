"""Safety case artifact generator."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SafetyClaim:
    identifier: str
    statement: str
    evidence: list[str]


@dataclass
class SafetyCase:
    system: str
    claims: list[SafetyClaim] = field(default_factory=list)

    def add_claim(self, claim: SafetyClaim) -> None:
        self.claims.append(claim)

    def render(self) -> dict[str, object]:
        return {
            "system": self.system,
            "claims": [
                {"id": c.identifier, "statement": c.statement, "evidence": list(c.evidence)}
                for c in self.claims
            ],
        }
