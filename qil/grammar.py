"""QIL (Q Intent Language) Grammar Definition.

This module defines the EBNF grammar for the QIL intent language,
which is used to express computational intents for the QRATUM system.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

# EBNF Grammar for QIL
QIL_GRAMMAR = r"""
intent          = "INTENT" identifier "{" intent_body "}"
intent_body     = statement*
statement       = objective | constraint | capability | time_spec | authority | trust | hardware
objective       = "OBJECTIVE" identifier
constraint      = "CONSTRAINT" identifier comparison_op value
capability      = "CAPABILITY" identifier
time_spec       = "TIME" time_key ":" value time_unit
authority       = "AUTHORITY" auth_key ":" identifier
trust           = "TRUST" "level" ":" trust_level
hardware        = "HARDWARE" hardware_spec
hardware_spec   = only_clause? not_clause?
only_clause     = "ONLY" cluster_list
not_clause      = "NOT" cluster_list
cluster_list    = cluster_type ("AND" cluster_type)*
cluster_type    = "GB200" | "MI300X" | "QPU" | "IPU" | "GAUDI3" | "CPU" | "CEREBRAS"
comparison_op   = ">=" | "<=" | "==" | ">" | "<" | "!="
identifier      = /[a-zA-Z_][a-zA-Z0-9_]*/
value           = number | string | identifier
number          = /[0-9]+(\.[0-9]+)?/
string          = /"[^"]*"/
time_key        = "deadline" | "budget" | "window"
time_unit       = "s" | "ms" | "h" | "m"
auth_key        = "user" | "group" | "role"
trust_level     = "verified" | "trusted" | "untrusted" | "sandbox"
"""


class GrammarConstants:
    """Constants for QIL grammar tokens and keywords."""

    # Keywords
    KEYWORDS = {
        "INTENT",
        "OBJECTIVE",
        "CONSTRAINT",
        "CAPABILITY",
        "TIME",
        "AUTHORITY",
        "TRUST",
        "HARDWARE",
        "ONLY",
        "NOT",
        "AND",
    }

    # Cluster types
    CLUSTER_TYPES = {
        "GB200",
        "MI300X",
        "QPU",
        "IPU",
        "GAUDI3",
        "CPU",
        "CEREBRAS",
    }

    # Comparison operators
    COMPARISON_OPS = {">=", "<=", "==", ">", "<", "!="}

    # Time keywords
    TIME_KEYS = {"deadline", "budget", "window"}

    # Time units
    TIME_UNITS = {"s", "ms", "h", "m"}

    # Authority keys
    AUTH_KEYS = {"user", "group", "role"}

    # Trust levels
    TRUST_LEVELS = {"verified", "trusted", "untrusted", "sandbox"}


def validate_grammar_token(token_type: str, value: str) -> bool:
    """Validate a grammar token against QIL rules.

    Args:
        token_type: Type of token to validate
        value: Token value to validate

    Returns:
        True if token is valid, False otherwise
    """
    token_validators = {
        "keyword": GrammarConstants.KEYWORDS,
        "cluster_type": GrammarConstants.CLUSTER_TYPES,
        "comparison_op": GrammarConstants.COMPARISON_OPS,
        "time_key": GrammarConstants.TIME_KEYS,
        "time_unit": GrammarConstants.TIME_UNITS,
        "auth_key": GrammarConstants.AUTH_KEYS,
        "trust_level": GrammarConstants.TRUST_LEVELS,
    }
    if token_type in token_validators:
        return value in token_validators[token_type]
    return False
