"""Recursive descent parser for QDL."""

from __future__ import annotations

from .ast import (
    BinaryOp,
    EconomicPrimitive,
    Identifier,
    Number,
    Program,
    SafetyGuard,
    SimulationKernel,
    WorldModelCall,
)
from .lexer import Lexer, Token


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    @classmethod
    def parse(cls, source: str) -> Program:
        tokens = Lexer.lex(source)
        return cls(list(tokens)).program()

    def current(self) -> Token:
        return self.tokens[self.pos]

    def consume(self, expected: str) -> Token:
        token = self.current()
        if token.type != expected:
            raise SyntaxError(f"Expected {expected}, found {token.type}")
        self.pos += 1
        return token

    def match(self, types):
        if self.current().type in types:
            tok = self.current()
            self.pos += 1
            return tok
        return None

    def program(self) -> Program:
        statements = []
        while self.current().type != "EOF":
            statements.append(self.statement())
        return Program(statements)

    def statement(self):
        tok = self.current()
        if tok.type == "SAFETY":
            return self.safety_guard()
        if tok.type == "WORLDMODEL":
            return self.worldmodel_call()
        if tok.type == "SIMULATE":
            return self.simulation_kernel()
        if tok.type == "ECONOMIC":
            return self.economic_primitive()
        return self.expression()

    def expression(self):
        node = self.term()
        while self.current().type == "OP" and self.current().value in {"+", "-"}:
            op = self.current().value
            self.consume("OP")
            right = self.term()
            node = BinaryOp(op, node, right)
        return node

    def term(self):
        node = self.factor()
        while self.current().type == "OP" and self.current().value in {"*", "/"}:
            op = self.current().value
            self.consume("OP")
            right = self.factor()
            node = BinaryOp(op, node, right)
        return node

    def factor(self):
        tok = self.current()
        if tok.type == "NUMBER":
            self.consume("NUMBER")
            return Number(float(tok.value))
        if tok.type == "ID":
            self.consume("ID")
            return Identifier(tok.value)
        if tok.type == "LPAREN":
            self.consume("LPAREN")
            expr = self.expression()
            self.consume("RPAREN")
            return expr
        raise SyntaxError(f"Unexpected token {tok}")

    def worldmodel_call(self):
        self.consume("WORLDMODEL")
        self.consume("COLON")
        name = self.consume("ID").value
        args = self.arg_list()
        return WorldModelCall(name, args)

    def simulation_kernel(self):
        self.consume("SIMULATE")
        self.consume("COLON")
        kernel = self.consume("ID").value
        self.consume("LPAREN")
        params = {}
        while self.current().type != "RPAREN":
            key = self.consume("ID").value
            self.consume("COLON")
            params[key] = self.expression()
            if self.current().type == "COMMA":
                self.consume("COMMA")
            else:
                break
        self.consume("RPAREN")
        return SimulationKernel(kernel, params)

    def economic_primitive(self):
        self.consume("ECONOMIC")
        self.consume("COLON")
        prim = self.consume("ID").value
        amount = self.expression()
        return EconomicPrimitive(prim, amount)

    def safety_guard(self):
        self.consume("SAFETY")
        condition = self.expression()
        self.consume("LBRACE")
        action = self.statement()
        failure = None
        if self.current().type == "ELSE":
            self.consume("ELSE")
            failure = self.statement()
        self.consume("RBRACE")
        return SafetyGuard(condition, action, failure)

    def arg_list(self):
        args = []
        if self.current().type == "LPAREN":
            self.consume("LPAREN")
            if self.current().type == "RPAREN":
                self.consume("RPAREN")
                return args
            while True:
                args.append(self.expression())
                if self.current().type == "COMMA":
                    self.consume("COMMA")
                    continue
                break
            self.consume("RPAREN")
        return args
