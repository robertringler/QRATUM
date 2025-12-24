"""QIL Parser and Validator.

This module implements a parser for the QIL intent language,
converting QIL text into validated AST structures.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import re
from typing import Any

from qil.ast import (Authority, Capability, Constraint, HardwareSpec, Intent,
                     Objective, TimeSpec, Trust)
from qil.grammar import GrammarConstants


class ParseError(Exception):
    """Exception raised for QIL parsing errors."""

    pass


class QILParser:
    """Parser for QIL intent language."""

    def __init__(self, qil_text: str) -> None:
        """Initialize parser with QIL text.

        Args:
            qil_text: QIL source text to parse
        """
        self.text = qil_text
        self.pos = 0
        self.line = 1
        self.column = 1

    def parse(self) -> Intent:
        """Parse QIL text into an Intent AST.

        Returns:
            Parsed Intent object

        Raises:
            ParseError: If parsing fails
        """
        self._skip_whitespace()

        # Parse INTENT keyword and name
        if not self._consume_keyword("INTENT"):
            raise ParseError(f"Expected 'INTENT' at line {self.line}, column {self.column}")

        self._skip_whitespace()
        intent_name = self._parse_identifier()
        if not intent_name:
            raise ParseError(
                f"Expected intent name at line {self.line}, column {self.column}"
            )

        self._skip_whitespace()
        if not self._consume_char("{"):
            raise ParseError(
                f"Expected '{{' after intent name at line {self.line}, column {self.column}"
            )

        # Parse intent body
        objective: Objective | None = None
        constraints: list[Constraint] = []
        capabilities: list[Capability] = []
        time_specs: list[TimeSpec] = []
        authorities: list[Authority] = []
        trust: Trust | None = None
        hardware: HardwareSpec | None = None

        while True:
            self._skip_whitespace()

            # Check for end of intent
            if self._peek_char() == "}":
                self._consume_char("}")
                break

            # Parse statement
            if self._peek_keyword("OBJECTIVE"):
                if objective is not None:
                    raise ParseError(
                        f"Duplicate OBJECTIVE at line {self.line}, column {self.column}"
                    )
                objective = self._parse_objective()
            elif self._peek_keyword("CONSTRAINT"):
                constraints.append(self._parse_constraint())
            elif self._peek_keyword("CAPABILITY"):
                capabilities.append(self._parse_capability())
            elif self._peek_keyword("TIME"):
                time_specs.append(self._parse_time_spec())
            elif self._peek_keyword("AUTHORITY"):
                authorities.append(self._parse_authority())
            elif self._peek_keyword("TRUST"):
                if trust is not None:
                    raise ParseError(f"Duplicate TRUST at line {self.line}, column {self.column}")
                trust = self._parse_trust()
            elif self._peek_keyword("HARDWARE"):
                if hardware is not None:
                    raise ParseError(
                        f"Duplicate HARDWARE at line {self.line}, column {self.column}"
                    )
                hardware = self._parse_hardware()
            else:
                raise ParseError(
                    f"Unexpected token at line {self.line}, column {self.column}: "
                    f"{self._peek_text(20)}"
                )

        # Validate required fields
        if objective is None:
            raise ParseError("Intent must have an OBJECTIVE")

        return Intent(
            name=intent_name,
            objective=objective,
            constraints=constraints,
            capabilities=capabilities,
            time_specs=time_specs,
            authorities=authorities,
            trust=trust,
            hardware=hardware,
        )

    def _parse_objective(self) -> Objective:
        """Parse OBJECTIVE statement."""
        self._consume_keyword("OBJECTIVE")
        self._skip_whitespace()
        name = self._parse_identifier()
        if not name:
            raise ParseError(
                f"Expected objective name at line {self.line}, column {self.column}"
            )
        return Objective(name=name)

    def _parse_constraint(self) -> Constraint:
        """Parse CONSTRAINT statement."""
        self._consume_keyword("CONSTRAINT")
        self._skip_whitespace()

        name = self._parse_identifier()
        if not name:
            raise ParseError(
                f"Expected constraint name at line {self.line}, column {self.column}"
            )

        self._skip_whitespace()
        operator = self._parse_comparison_op()
        if not operator:
            raise ParseError(
                f"Expected comparison operator at line {self.line}, column {self.column}"
            )

        self._skip_whitespace()
        value = self._parse_value()
        if value is None:
            raise ParseError(f"Expected value at line {self.line}, column {self.column}")

        return Constraint(name=name, operator=operator, value=value)

    def _parse_capability(self) -> Capability:
        """Parse CAPABILITY statement."""
        self._consume_keyword("CAPABILITY")
        self._skip_whitespace()
        name = self._parse_identifier()
        if not name:
            raise ParseError(
                f"Expected capability name at line {self.line}, column {self.column}"
            )
        return Capability(name=name)

    def _parse_time_spec(self) -> TimeSpec:
        """Parse TIME specification."""
        self._consume_keyword("TIME")
        self._skip_whitespace()

        key = self._parse_identifier()
        if key not in GrammarConstants.TIME_KEYS:
            raise ParseError(
                f"Invalid time key '{key}' at line {self.line}, column {self.column}. "
                f"Must be one of: {GrammarConstants.TIME_KEYS}"
            )

        self._skip_whitespace()
        if not self._consume_char(":"):
            raise ParseError(f"Expected ':' at line {self.line}, column {self.column}")

        self._skip_whitespace()
        value_text = self._parse_number()
        if value_text is None:
            raise ParseError(f"Expected time value at line {self.line}, column {self.column}")

        value = float(value_text)

        # Parse time unit
        unit = self._parse_identifier()
        if unit not in GrammarConstants.TIME_UNITS:
            raise ParseError(
                f"Invalid time unit '{unit}' at line {self.line}, column {self.column}. "
                f"Must be one of: {GrammarConstants.TIME_UNITS}"
            )

        return TimeSpec(key=key, value=value, unit=unit)

    def _parse_authority(self) -> Authority:
        """Parse AUTHORITY statement."""
        self._consume_keyword("AUTHORITY")
        self._skip_whitespace()

        key = self._parse_identifier()
        if key not in GrammarConstants.AUTH_KEYS:
            raise ParseError(
                f"Invalid authority key '{key}' at line {self.line}, column {self.column}. "
                f"Must be one of: {GrammarConstants.AUTH_KEYS}"
            )

        self._skip_whitespace()
        if not self._consume_char(":"):
            raise ParseError(f"Expected ':' at line {self.line}, column {self.column}")

        self._skip_whitespace()
        value = self._parse_identifier()
        if not value:
            raise ParseError(
                f"Expected authority value at line {self.line}, column {self.column}"
            )

        return Authority(key=key, value=value)

    def _parse_trust(self) -> Trust:
        """Parse TRUST statement."""
        self._consume_keyword("TRUST")
        self._skip_whitespace()

        if not self._consume_identifier("level"):
            raise ParseError(f"Expected 'level' at line {self.line}, column {self.column}")

        self._skip_whitespace()
        if not self._consume_char(":"):
            raise ParseError(f"Expected ':' at line {self.line}, column {self.column}")

        self._skip_whitespace()
        level = self._parse_identifier()
        if level not in GrammarConstants.TRUST_LEVELS:
            raise ParseError(
                f"Invalid trust level '{level}' at line {self.line}, column {self.column}. "
                f"Must be one of: {GrammarConstants.TRUST_LEVELS}"
            )

        return Trust(level=level)

    def _parse_hardware(self) -> HardwareSpec:
        """Parse HARDWARE specification."""
        self._consume_keyword("HARDWARE")
        self._skip_whitespace()

        only_clusters: list[str] = []
        not_clusters: list[str] = []

        # Parse ONLY clause if present
        if self._peek_keyword("ONLY"):
            self._consume_keyword("ONLY")
            self._skip_whitespace()
            only_clusters = self._parse_cluster_list()
            self._skip_whitespace()

        # Parse NOT clause if present
        if self._peek_keyword("NOT"):
            self._consume_keyword("NOT")
            self._skip_whitespace()
            not_clusters = self._parse_cluster_list()

        return HardwareSpec(only_clusters=only_clusters, not_clusters=not_clusters)

    def _parse_cluster_list(self) -> list[str]:
        """Parse cluster type list (cluster1 AND cluster2 AND ...)."""
        clusters = []

        # Parse first cluster
        cluster = self._parse_cluster_type()
        if not cluster:
            raise ParseError(
                f"Expected cluster type at line {self.line}, column {self.column}"
            )
        clusters.append(cluster)

        # Parse additional clusters with AND
        while True:
            self._skip_whitespace()
            if not self._peek_keyword("AND"):
                break
            self._consume_keyword("AND")
            self._skip_whitespace()
            cluster = self._parse_cluster_type()
            if not cluster:
                raise ParseError(
                    f"Expected cluster type after AND at line {self.line}, column {self.column}"
                )
            clusters.append(cluster)

        return clusters

    def _parse_cluster_type(self) -> str | None:
        """Parse cluster type identifier."""
        cluster = self._parse_identifier()
        if cluster and cluster in GrammarConstants.CLUSTER_TYPES:
            return cluster
        return None

    def _parse_comparison_op(self) -> str | None:
        """Parse comparison operator."""
        for op in [">=", "<=", "==", "!=", ">", "<"]:
            if self._peek_text(len(op)) == op:
                self.pos += len(op)
                self.column += len(op)
                return op
        return None

    def _parse_identifier(self) -> str | None:
        """Parse identifier (alphanumeric + underscore, starting with letter or _)."""
        match = re.match(r"[a-zA-Z_][a-zA-Z0-9_]*", self.text[self.pos :])
        if match:
            identifier = match.group(0)
            self.pos += len(identifier)
            self.column += len(identifier)
            return identifier
        return None

    def _parse_number(self) -> str | None:
        """Parse number (integer or float)."""
        match = re.match(r"[0-9]+(\.[0-9]+)?", self.text[self.pos :])
        if match:
            number = match.group(0)
            self.pos += len(number)
            self.column += len(number)
            return number
        return None

    def _parse_value(self) -> Any:
        """Parse value (number, string, or identifier)."""
        # Try number first
        num = self._parse_number()
        if num is not None:
            return int(num) if "." not in num else float(num)

        # Try string
        if self._peek_char() == '"':
            return self._parse_string()

        # Try identifier
        return self._parse_identifier()

    def _parse_string(self) -> str:
        """Parse quoted string."""
        if not self._consume_char('"'):
            raise ParseError(f"Expected '\"' at line {self.line}, column {self.column}")

        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] != '"':
            self.pos += 1
            self.column += 1

        if self.pos >= len(self.text):
            raise ParseError(f"Unterminated string at line {self.line}")

        value = self.text[start : self.pos]
        self._consume_char('"')
        return value

    def _consume_keyword(self, keyword: str) -> bool:
        """Consume expected keyword."""
        if self._peek_keyword(keyword):
            self.pos += len(keyword)
            self.column += len(keyword)
            return True
        return False

    def _consume_identifier(self, identifier: str) -> bool:
        """Consume expected identifier."""
        saved_pos = self.pos
        saved_col = self.column
        parsed = self._parse_identifier()
        if parsed == identifier:
            return True
        self.pos = saved_pos
        self.column = saved_col
        return False

    def _consume_char(self, char: str) -> bool:
        """Consume expected character."""
        if self.pos < len(self.text) and self.text[self.pos] == char:
            self.pos += 1
            self.column += 1
            if char == "\n":
                self.line += 1
                self.column = 1
            return True
        return False

    def _peek_keyword(self, keyword: str) -> bool:
        """Check if next token is a keyword."""
        saved_pos = self.pos
        saved_col = self.column
        saved_line = self.line

        # Try to parse identifier
        parsed = self._parse_identifier()

        # Restore position
        self.pos = saved_pos
        self.column = saved_col
        self.line = saved_line

        return parsed == keyword

    def _peek_char(self) -> str:
        """Peek at current character."""
        if self.pos < len(self.text):
            return self.text[self.pos]
        return ""

    def _peek_text(self, length: int) -> str:
        """Peek at next N characters."""
        return self.text[self.pos : self.pos + length]

    def _skip_whitespace(self) -> None:
        """Skip whitespace and comments."""
        while self.pos < len(self.text):
            char = self.text[self.pos]
            if char in " \t\r\n":
                if char == "\n":
                    self.line += 1
                    self.column = 1
                else:
                    self.column += 1
                self.pos += 1
            elif char == "#":
                # Skip comment line
                while self.pos < len(self.text) and self.text[self.pos] != "\n":
                    self.pos += 1
            else:
                break


def parse_intent(qil_text: str) -> Intent:
    """Parse QIL text into an Intent object.

    Args:
        qil_text: QIL source text

    Returns:
        Parsed Intent object

    Raises:
        ParseError: If parsing fails
    """
    parser = QILParser(qil_text)
    return parser.parse()
