"""Safety policy engine for output validation and filtering."""

import logging
import re
from enum import Enum
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolicyVerdict(Enum):
    """Policy evaluation verdict."""

    APPROVED = "approved"
    BLOCKED = "blocked"
    FLAGGED = "flagged"


class PolicyEngine:
    """Engine for evaluating safety policies on model outputs."""

    def __init__(self):
        """Initialize policy engine with default rules."""
        self.blocked_patterns = [
            r"\b(secret|password|api[_-]?key)\b",  # Credentials
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN-like patterns
            r"\b[A-Z0-9]{20,}\b",  # Long tokens
        ]
        self.flagged_patterns = [
            r"\b(confidential|proprietary|internal[_-]?only)\b",
            r"\b(TODO|FIXME|HACK)\b",
        ]
        logger.info("Initialized PolicyEngine with default rules")

    def evaluate(self, text: str, metadata: Optional[Dict] = None) -> Dict:
        """Evaluate text against safety policies.

        Args:
            text: Text to evaluate
            metadata: Optional metadata for context-aware evaluation

        Returns:
            Dictionary with verdict, violations, and sanitized text
        """
        violations = []
        verdict = PolicyVerdict.APPROVED

        # Check blocked patterns
        for pattern in self.blocked_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                violations.append(
                    {
                        "type": "blocked",
                        "pattern": pattern,
                        "match": match.group(),
                        "position": match.span(),
                    }
                )
                verdict = PolicyVerdict.BLOCKED

        # Check flagged patterns (only if not already blocked)
        if verdict != PolicyVerdict.BLOCKED:
            for pattern in self.flagged_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    violations.append(
                        {
                            "type": "flagged",
                            "pattern": pattern,
                            "match": match.group(),
                            "position": match.span(),
                        }
                    )
                    if verdict == PolicyVerdict.APPROVED:
                        verdict = PolicyVerdict.FLAGGED

        # Sanitize if needed
        sanitized_text = text
        if violations:
            for violation in violations:
                if violation["type"] == "blocked":
                    # Redact blocked content
                    sanitized_text = sanitized_text.replace(violation["match"], "[REDACTED]")

        result = {
            "verdict": verdict.value,
            "violations": violations,
            "violation_count": len(violations),
            "sanitized_text": sanitized_text,
            "original_length": len(text),
            "sanitized_length": len(sanitized_text),
        }

        logger.info(f"Policy evaluation: {verdict.value} ({len(violations)} violations)")
        return result

    def add_blocked_pattern(self, pattern: str):
        """Add a new blocked pattern.

        Args:
            pattern: Regex pattern to block
        """
        self.blocked_patterns.append(pattern)
        logger.info(f"Added blocked pattern: {pattern}")

    def add_flagged_pattern(self, pattern: str):
        """Add a new flagged pattern.

        Args:
            pattern: Regex pattern to flag
        """
        self.flagged_patterns.append(pattern)
        logger.info(f"Added flagged pattern: {pattern}")


def evaluate_safety(text: str) -> Dict:
    """Convenience function for safety evaluation.

    Args:
        text: Text to evaluate

    Returns:
        Evaluation result dictionary
    """
    engine = PolicyEngine()
    return engine.evaluate(text)
