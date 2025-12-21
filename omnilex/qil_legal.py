"""Extended QIL (QRATUM Intent Language) for legal analysis.

This module defines the legal-specific QIL intent structure that integrates
with the QRATUM core contract system.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class LegalQILIntent:
    """Legal analysis intent for QRATUM execution.

    This immutable dataclass represents a legal analysis request that will be
    executed as a QRATUM contract quartet (Create, Read, Update, Delete).

    Attributes:
        intent_id: Unique identifier for this intent (auto-generated if empty)
        compute_task: Type of legal computation (e.g., 'irac_analysis', 'contract_review')
        jurisdiction_primary: Primary jurisdiction code (ISO 3166-1 + subdivision)
        jurisdictions_secondary: Additional jurisdictions for conflict analysis
        legal_domain: Domain of law (e.g., 'contract', 'tort', 'criminal')
        reasoning_framework: Framework to use (e.g., 'irac', 'creac')
        attorney_supervised: Whether this analysis requires attorney supervision
        raw_facts: Factual scenario to analyze
        legal_question: Specific legal question to answer
    """

    intent_id: str
    compute_task: str
    jurisdiction_primary: str
    jurisdictions_secondary: tuple[str, ...]
    legal_domain: str
    reasoning_framework: str
    attorney_supervised: bool
    raw_facts: str
    legal_question: str

    def __post_init__(self) -> None:
        """Validate legal QIL intent."""
        if not self.compute_task:
            raise ValueError("compute_task cannot be empty")
        if not self.jurisdiction_primary:
            raise ValueError("jurisdiction_primary cannot be empty")
        if not self.legal_domain:
            raise ValueError("legal_domain cannot be empty")
        if not self.reasoning_framework:
            raise ValueError("reasoning_framework cannot be empty")
        if not self.raw_facts:
            raise ValueError("raw_facts cannot be empty")
        if not self.legal_question:
            raise ValueError("legal_question cannot be empty")

    def to_qil_string(self) -> str:
        """Generate QIL source representation.

        Returns:
            QIL source code representation of this intent
        """
        return f"""LEGAL_INTENT {{
    intent_id: "{self.intent_id}"
    compute_task: "{self.compute_task}"
    jurisdiction: {{
        primary: "{self.jurisdiction_primary}"
        secondary: [{', '.join(f'"{j}"' for j in self.jurisdictions_secondary)}]
    }}
    domain: "{self.legal_domain}"
    reasoning: "{self.reasoning_framework}"
    attorney_supervised: {str(self.attorney_supervised).lower()}
    facts: "{self._escape_string(self.raw_facts)}"
    question: "{self._escape_string(self.legal_question)}"
}}"""

    def serialize(self) -> dict:
        """Return deterministic dict representation.

        Returns:
            Dictionary with all intent fields in sorted order
        """
        return {
            "attorney_supervised": self.attorney_supervised,
            "compute_task": self.compute_task,
            "intent_id": self.intent_id,
            "jurisdiction_primary": self.jurisdiction_primary,
            "jurisdictions_secondary": list(self.jurisdictions_secondary),
            "legal_domain": self.legal_domain,
            "legal_question": self.legal_question,
            "raw_facts": self.raw_facts,
            "reasoning_framework": self.reasoning_framework,
        }

    def to_json(self) -> str:
        """Return JSON with sorted keys for deterministic serialization.

        Returns:
            JSON string representation with sorted keys
        """
        return json.dumps(self.serialize(), sort_keys=True, default=str)

    def compute_hash(self) -> str:
        """Return SHA-256 hash of canonical JSON serialization.

        Returns:
            Hexadecimal SHA-256 hash string
        """
        json_str = self.to_json()
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    @staticmethod
    def _escape_string(s: str) -> str:
        """Escape string for QIL representation.

        Args:
            s: String to escape

        Returns:
            Escaped string
        """
        return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')


def generate_intent_id(task: str, jurisdiction: str, timestamp: float) -> str:
    """Generate deterministic intent ID.

    Args:
        task: Compute task type
        jurisdiction: Primary jurisdiction
        timestamp: Unix timestamp

    Returns:
        Intent ID in format: OMNILEX-{hash}
    """
    payload = f"{task}:{jurisdiction}:{timestamp}"
    hash_val = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"OMNILEX-{hash_val}"
