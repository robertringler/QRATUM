"""Minimal type system for QDL."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QDLType:
    name: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


NumberType = QDLType("number")
BooleanType = QDLType("bool")
UnknownType = QDLType("unknown")


class TypeEnv:
    def __init__(self):
        self._types: dict[str, QDLType] = {}

    def declare(self, name: str, type_: QDLType):
        self._types[name] = type_

    def resolve(self, name: str) -> Optional[QDLType]:
        return self._types.get(name, UnknownType)


class TypeChecker:
    def __init__(self, env: Optional[TypeEnv] = None):
        self.env = env or TypeEnv()

    def infer_number(self, value) -> QDLType:
        if isinstance(value, bool):
            return BooleanType
        if isinstance(value, (int, float)):
            return NumberType
        return UnknownType
