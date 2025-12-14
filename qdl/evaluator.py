"""Evaluator for QDL AST nodes."""
from __future__ import annotations

from typing import Any

from .ast import ASTNode, Program
from .type_system import TypeChecker


class Evaluator:
    def __init__(self, context: dict[str, Any] | None = None):
        self.context = context or {}
        self.type_checker = TypeChecker()

    def evaluate(self, program: Program) -> Any:
        if not isinstance(program, Program):
            raise TypeError("Evaluator expects a Program node")
        return program.evaluate(self.context)

    def evaluate_node(self, node: ASTNode) -> Any:
        return node.evaluate(self.context)
