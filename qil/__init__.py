"""QIL - Q Intent Language.

This module provides the QIL intent language parser, AST, and serialization
for expressing computational intents in the QRATUM system.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from qil.ast import (Authority, Capability, Constraint, HardwareSpec, Intent,
                     Objective, TimeSpec, Trust)
from qil.grammar import QIL_GRAMMAR, GrammarConstants, validate_grammar_token
from qil.parser import ParseError, QILParser, parse_intent
from qil.serializer import (compute_hash, intent_to_canonical_form,
                            serialize_intent, to_json)

__all__ = [
    # AST nodes
    "Intent",
    "Objective",
    "Constraint",
    "Capability",
    "TimeSpec",
    "Authority",
    "Trust",
    "HardwareSpec",
    # Parser
    "QILParser",
    "parse_intent",
    "ParseError",
    # Grammar
    "QIL_GRAMMAR",
    "GrammarConstants",
    "validate_grammar_token",
    # Serialization
    "serialize_intent",
    "to_json",
    "compute_hash",
    "intent_to_canonical_form",
]

__version__ = "1.0.0"
