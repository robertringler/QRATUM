"""Deterministic lexer for QDL."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

TOKEN_SPEC = [
    ("NUMBER", r"\d+(?:\.\d+)?"),
    ("ID", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("OP", r"==|<=|>=|[+\-*/<>]"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("COLON", r":"),
    ("COMMA", r","),
    ("NEWLINE", r"\n"),
    ("SKIP", r"[ \t]+"),
    ("MISMATCH", r"."),
]

KEYWORDS = {
    "worldmodel",
    "simulate",
    "economic",
    "safety",
    "if",
    "else",
}


tok_regex = re.compile("|".join(f"(?P<{name}>{regex})" for name, regex in TOKEN_SPEC))


@dataclass
class Token:
    type: str
    value: str

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value})"


class Lexer:
    def __init__(self, source: str):
        self.source = source

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        for match in tok_regex.finditer(self.source):
            kind = match.lastgroup
            value = match.group()
            if kind == "NUMBER":
                tokens.append(Token("NUMBER", value))
            elif kind == "ID":
                if value in KEYWORDS:
                    tokens.append(Token(value.upper(), value))
                else:
                    tokens.append(Token("ID", value))
            elif kind == "OP":
                tokens.append(Token("OP", value))
            elif kind in {"LPAREN", "RPAREN", "LBRACE", "RBRACE", "COLON", "COMMA"}:
                tokens.append(Token(kind, value))
            elif kind in {"NEWLINE", "SKIP"}:
                continue
            elif kind == "MISMATCH":
                raise SyntaxError(f"Unexpected character: {value}")
        tokens.append(Token("EOF", ""))
        return tokens

    @staticmethod
    def lex(source: str) -> Iterable[Token]:
        return Lexer(source).tokenize()
