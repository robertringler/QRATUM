"""Q-Core Authorization Engine - NEVER Executes.

This module implements the authorization engine for QRATUM.
It authorizes intents but NEVER executes workloads.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qil.ast import Intent


class AuthorizationError(Exception):
    """Exception raised for authorization failures (FATAL)."""

    pass


@dataclass
class AuthorizationResult:
    """Result of intent authorization.

    Attributes:
        authorized: Whether intent is authorized
        reason: Reason for authorization decision
        proof: Authorization proof token
        violations: List of policy violations (if any)
    """

    authorized: bool
    reason: str
    proof: str
    violations: list[str]


class AuthorizationEngine:
    """Sovereign authorization engine (NEVER executes)."""

    def __init__(self, policies: dict[str, Any] | None = None) -> None:
        """Initialize authorization engine.

        Args:
            policies: Optional policy configuration
        """
        self.policies = policies or {}
        self._authorized_intents: dict[str, AuthorizationResult] = {}

    def authorize_intent(self, intent: Intent) -> AuthorizationResult:
        """Authorize an intent against policies.

        CRITICAL: This method ONLY authorizes. It NEVER executes.

        Args:
            intent: Intent to authorize

        Returns:
            AuthorizationResult with authorization decision

        Raises:
            AuthorizationError: If authorization fails fatally
        """
        violations: list[str] = []

        # Check authority requirements
        if not intent.authorities:
            violations.append("No authority specified in intent")

        # Validate trust level
        if intent.trust and intent.trust.level == "untrusted":
            violations.append("Untrusted intent cannot be authorized")

        # Check hardware requirements
        if (
            intent.hardware
            and not intent.hardware.only_clusters
            and not intent.hardware.not_clusters
        ):
            violations.append("Hardware specification is empty")

        # Check for conflicting constraints
        constraint_names = [c.name for c in intent.constraints]
        if len(constraint_names) != len(set(constraint_names)):
            violations.append("Duplicate constraint names found")

        # Validate time specifications
        for time_spec in intent.time_specs:
            if time_spec.value <= 0:
                violations.append(f"Invalid time value: {time_spec.value}")

        # Make authorization decision
        authorized = len(violations) == 0

        if authorized:
            proof = f"AUTH_{intent.name}_{len(self._authorized_intents)}"
            result = AuthorizationResult(
                authorized=True,
                reason="Intent authorized successfully",
                proof=proof,
                violations=[],
            )
        else:
            proof = ""
            result = AuthorizationResult(
                authorized=False,
                reason=f"Authorization failed: {'; '.join(violations)}",
                proof=proof,
                violations=violations,
            )

        # Cache authorization result
        self._authorized_intents[intent.name] = result

        # Raise FATAL error if not authorized
        if not authorized:
            raise AuthorizationError(
                f"Intent '{intent.name}' authorization FAILED: {result.reason}"
            )

        return result

    def verify_authorization(self, intent_name: str, proof: str) -> bool:
        """Verify an authorization proof.

        Args:
            intent_name: Name of intent
            proof: Authorization proof to verify

        Returns:
            True if proof is valid, False otherwise
        """
        if intent_name not in self._authorized_intents:
            return False

        result = self._authorized_intents[intent_name]
        return result.authorized and result.proof == proof

    def revoke_authorization(self, intent_name: str) -> None:
        """Revoke authorization for an intent.

        Args:
            intent_name: Name of intent to revoke
        """
        if intent_name in self._authorized_intents:
            del self._authorized_intents[intent_name]

    def get_authorization_status(self, intent_name: str) -> AuthorizationResult | None:
        """Get authorization status for an intent.

        Args:
            intent_name: Name of intent

        Returns:
            AuthorizationResult if intent was authorized, None otherwise
        """
        return self._authorized_intents.get(intent_name)

    def list_authorized_intents(self) -> list[str]:
        """List all authorized intent names.

        Returns:
            List of authorized intent names
        """
        return [name for name, result in self._authorized_intents.items() if result.authorized]


def check_authority_constraints(intent: Intent) -> list[str]:
    """Check authority constraints for an intent.

    Args:
        intent: Intent to check

    Returns:
        List of constraint violations (empty if all pass)
    """
    violations: list[str] = []

    # Check for required authority
    if not intent.authorities:
        violations.append("Intent must specify at least one authority")

    # Check for valid trust level
    if intent.trust and intent.trust.level not in ["verified", "trusted"]:
        violations.append(f"Invalid trust level: {intent.trust.level}")

    return violations
