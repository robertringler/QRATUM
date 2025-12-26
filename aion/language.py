"""AION Source Language Module.

High-level intent-first syntax ("vibe coding") supporting:
- Python-style, SQL-style, Rust/CUDA/FPGA semantics
- Inline cross-language/hardware fusion
- Zero-copy memory annotations
- Parallelism and affinity metadata

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from .sir.edges import HyperEdge, ParallelismKind
from .sir.hypergraph import HyperGraph
from .sir.vertices import (
    AIONType,
    EffectKind,
    HardwareAffinity,
    Provenance,
    Vertex,
)


class TokenKind(Enum):
    """AION token kinds."""

    # Keywords
    PARALLEL = auto()
    WARP = auto()
    REDUCE = auto()
    REGION = auto()
    MOVE = auto()
    INTO = auto()
    APPLY = auto()
    LET = auto()
    FN = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    RETURN = auto()
    QUERY = auto()
    SELECT = auto()
    FROM = auto()
    WHERE = auto()

    # Hardware targets
    GPU = auto()
    GPU_STREAM0 = auto()
    GPU_STREAM1 = auto()
    FPGA = auto()
    FPGA_LUT = auto()
    CPU = auto()
    THREAD = auto()
    STATIC = auto()

    # Region modifiers
    STACK = auto()
    HEAP = auto()
    SHARED = auto()
    GLOBAL = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    ASSIGN = auto()
    ARROW = auto()
    COLON = auto()
    SEMICOLON = auto()
    COMMA = auto()
    DOT = auto()

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()

    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    IDENTIFIER = auto()

    # Special
    NEWLINE = auto()
    EOF = auto()


@dataclass
class Token:
    """AION source token."""

    kind: TokenKind
    value: Any = None
    line: int = 0
    column: int = 0


@dataclass
class AIONASTNode:
    """AION AST node base."""

    kind: str
    children: list[AIONASTNode] = field(default_factory=list)
    value: Any = None
    attributes: dict[str, Any] = field(default_factory=dict)
    line: int = 0


class AIONLexer:
    """Lexer for AION source language."""

    KEYWORDS = {
        "parallel": TokenKind.PARALLEL,
        "warp": TokenKind.WARP,
        "reduce": TokenKind.REDUCE,
        "region": TokenKind.REGION,
        "move": TokenKind.MOVE,
        "into": TokenKind.INTO,
        "apply": TokenKind.APPLY,
        "let": TokenKind.LET,
        "fn": TokenKind.FN,
        "if": TokenKind.IF,
        "else": TokenKind.ELSE,
        "while": TokenKind.WHILE,
        "for": TokenKind.FOR,
        "in": TokenKind.IN,
        "return": TokenKind.RETURN,
        "query": TokenKind.QUERY,
        "select": TokenKind.SELECT,
        "from": TokenKind.FROM,
        "where": TokenKind.WHERE,
        # Hardware
        "GPU": TokenKind.GPU,
        "GPU_Stream0": TokenKind.GPU_STREAM0,
        "GPU_Stream1": TokenKind.GPU_STREAM1,
        "FPGA": TokenKind.FPGA,
        "FPGA_LUT": TokenKind.FPGA_LUT,
        "CPU": TokenKind.CPU,
        "Thread": TokenKind.THREAD,
        "Static": TokenKind.STATIC,
        # Memory
        "stack": TokenKind.STACK,
        "heap": TokenKind.HEAP,
        "shared": TokenKind.SHARED,
        "global": TokenKind.GLOBAL,
    }

    def __init__(self, source: str) -> None:
        """Initialize lexer."""
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1

    def tokenize(self) -> list[Token]:
        """Tokenize source code."""
        tokens = []

        while self.pos < len(self.source):
            char = self.source[self.pos]

            # Skip whitespace (but track newlines)
            if char in " \t\r":
                self.pos += 1
                self.column += 1
                continue

            if char == "\n":
                tokens.append(Token(TokenKind.NEWLINE, line=self.line, column=self.column))
                self.pos += 1
                self.line += 1
                self.column = 1
                continue

            # Comments
            if char == "/" and self.pos + 1 < len(self.source):
                if self.source[self.pos + 1] == "/":
                    while self.pos < len(self.source) and self.source[self.pos] != "\n":
                        self.pos += 1
                    continue

            # Operators
            if char == "+":
                tokens.append(Token(TokenKind.PLUS, "+", self.line, self.column))
                self.pos += 1
                self.column += 1
            elif char == "-":
                if self.pos + 1 < len(self.source) and self.source[self.pos + 1] == ">":
                    tokens.append(Token(TokenKind.ARROW, "->", self.line, self.column))
                    self.pos += 2
                    self.column += 2
                else:
                    tokens.append(Token(TokenKind.MINUS, "-", self.line, self.column))
                    self.pos += 1
                    self.column += 1
            elif char == "*":
                tokens.append(Token(TokenKind.STAR, "*", self.line, self.column))
                self.pos += 1
                self.column += 1
            elif char == "/":
                tokens.append(Token(TokenKind.SLASH, "/", self.line, self.column))
                self.pos += 1
                self.column += 1
            elif char == "%":
                tokens.append(Token(TokenKind.PERCENT, "%", self.line, self.column))
                self.pos += 1
                self.column += 1
            elif char == "=":
                if self.pos + 1 < len(self.source) and self.source[self.pos + 1] == "=":
                    tokens.append(Token(TokenKind.EQ, "==", self.line, self.column))
                    self.pos += 2
                    self.column += 2
                else:
                    tokens.append(Token(TokenKind.ASSIGN, "=", self.line, self.column))
                    self.pos += 1
                    self.column += 1
            elif char == "!":
                if self.pos + 1 < len(self.source) and self.source[self.pos + 1] == "=":
                    tokens.append(Token(TokenKind.NE, "!=", self.line, self.column))
                    self.pos += 2
                    self.column += 2
                else:
                    self.pos += 1
                    self.column += 1
            elif char == "<":
                if self.pos + 1 < len(self.source) and self.source[self.pos + 1] == "=":
                    tokens.append(Token(TokenKind.LE, "<=", self.line, self.column))
                    self.pos += 2
                    self.column += 2
                else:
                    tokens.append(Token(TokenKind.LT, "<", self.line, self.column))
                    self.pos += 1
                    self.column += 1
            elif char == ">":
                if self.pos + 1 < len(self.source) and self.source[self.pos + 1] == "=":
                    tokens.append(Token(TokenKind.GE, ">=", self.line, self.column))
                    self.pos += 2
                    self.column += 2
                else:
                    tokens.append(Token(TokenKind.GT, ">", self.line, self.column))
                    self.pos += 1
                    self.column += 1

            # Delimiters
            elif char == "(":
                tokens.append(Token(TokenKind.LPAREN, "(", self.line, self.column))
                self.pos += 1
                self.column += 1
            elif char == ")":
                tokens.append(Token(TokenKind.RPAREN, ")", self.line, self.column))
                self.pos += 1
                self.column += 1
            elif char == "{":
                tokens.append(Token(TokenKind.LBRACE, "{", self.line, self.column))
                self.pos += 1
                self.column += 1
            elif char == "}":
                tokens.append(Token(TokenKind.RBRACE, "}", self.line, self.column))
                self.pos += 1
                self.column += 1
            elif char == "[":
                tokens.append(Token(TokenKind.LBRACKET, "[", self.line, self.column))
                self.pos += 1
                self.column += 1
            elif char == "]":
                tokens.append(Token(TokenKind.RBRACKET, "]", self.line, self.column))
                self.pos += 1
                self.column += 1
            elif char == ":":
                tokens.append(Token(TokenKind.COLON, ":", self.line, self.column))
                self.pos += 1
                self.column += 1
            elif char == ";":
                tokens.append(Token(TokenKind.SEMICOLON, ";", self.line, self.column))
                self.pos += 1
                self.column += 1
            elif char == ",":
                tokens.append(Token(TokenKind.COMMA, ",", self.line, self.column))
                self.pos += 1
                self.column += 1
            elif char == ".":
                tokens.append(Token(TokenKind.DOT, ".", self.line, self.column))
                self.pos += 1
                self.column += 1

            # String literals
            elif char == '"':
                tokens.append(self._scan_string())

            # Numbers
            elif char.isdigit():
                tokens.append(self._scan_number())

            # Identifiers and keywords
            elif char.isalpha() or char == "_":
                tokens.append(self._scan_identifier())

            else:
                self.pos += 1
                self.column += 1

        tokens.append(Token(TokenKind.EOF, line=self.line, column=self.column))
        return tokens

    def _scan_string(self) -> Token:
        """Scan string literal."""
        start_col = self.column
        self.pos += 1  # Skip opening quote
        self.column += 1

        value = ""
        while self.pos < len(self.source) and self.source[self.pos] != '"':
            if self.source[self.pos] == "\\":
                self.pos += 1
                self.column += 1
                if self.pos < len(self.source):
                    escape_map = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"'}
                    value += escape_map.get(self.source[self.pos], self.source[self.pos])
            else:
                value += self.source[self.pos]
            self.pos += 1
            self.column += 1

        self.pos += 1  # Skip closing quote
        self.column += 1

        return Token(TokenKind.STRING, value, self.line, start_col)

    def _scan_number(self) -> Token:
        """Scan number literal."""
        start_col = self.column
        start_pos = self.pos

        while self.pos < len(self.source) and self.source[self.pos].isdigit():
            self.pos += 1
            self.column += 1

        # Check for float
        if self.pos < len(self.source) and self.source[self.pos] == ".":
            self.pos += 1
            self.column += 1
            while self.pos < len(self.source) and self.source[self.pos].isdigit():
                self.pos += 1
                self.column += 1
            value = float(self.source[start_pos : self.pos])
            return Token(TokenKind.FLOAT, value, self.line, start_col)

        value = int(self.source[start_pos : self.pos])
        return Token(TokenKind.INTEGER, value, self.line, start_col)

    def _scan_identifier(self) -> Token:
        """Scan identifier or keyword."""
        start_col = self.column
        start_pos = self.pos

        while self.pos < len(self.source) and (
            self.source[self.pos].isalnum() or self.source[self.pos] == "_"
        ):
            self.pos += 1
            self.column += 1

        text = self.source[start_pos : self.pos]

        if text in self.KEYWORDS:
            return Token(self.KEYWORDS[text], text, self.line, start_col)

        return Token(TokenKind.IDENTIFIER, text, self.line, start_col)


class AIONParser:
    """Parser for AION source language."""

    def __init__(self) -> None:
        """Initialize parser."""
        self.tokens: list[Token] = []
        self.pos = 0

    def parse(self, source: str) -> AIONASTNode:
        """Parse AION source to AST.

        Args:
            source: AION source code

        Returns:
            Root AST node
        """
        lexer = AIONLexer(source)
        self.tokens = lexer.tokenize()
        self.pos = 0

        return self._parse_program()

    def _current(self) -> Token:
        """Get current token."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(TokenKind.EOF)

    def _advance(self) -> Token:
        """Advance to next token."""
        token = self._current()
        self.pos += 1
        return token

    def _skip_newlines(self) -> None:
        """Skip newline tokens."""
        while self._current().kind == TokenKind.NEWLINE:
            self._advance()

    def _expect(self, kind: TokenKind) -> Token:
        """Expect a specific token kind."""
        token = self._current()
        if token.kind != kind:
            raise SyntaxError(f"Expected {kind}, got {token.kind} at line {token.line}")
        return self._advance()

    def _parse_program(self) -> AIONASTNode:
        """Parse program."""
        node = AIONASTNode(kind="program", line=1)

        while self._current().kind != TokenKind.EOF:
            self._skip_newlines()
            if self._current().kind == TokenKind.EOF:
                break

            stmt = self._parse_statement()
            if stmt:
                node.children.append(stmt)

        return node

    def _parse_statement(self) -> AIONASTNode | None:
        """Parse a statement."""
        token = self._current()

        if token.kind == TokenKind.PARALLEL:
            return self._parse_parallel()
        elif token.kind == TokenKind.REDUCE:
            return self._parse_reduce()
        elif token.kind == TokenKind.REGION:
            return self._parse_region()
        elif token.kind == TokenKind.MOVE:
            return self._parse_move()
        elif token.kind == TokenKind.LET:
            return self._parse_let()
        elif token.kind == TokenKind.FN:
            return self._parse_function()
        elif token.kind == TokenKind.IF:
            return self._parse_if()
        elif token.kind == TokenKind.WHILE:
            return self._parse_while()
        elif token.kind == TokenKind.FOR:
            return self._parse_for()
        elif token.kind == TokenKind.RETURN:
            return self._parse_return()
        elif token.kind == TokenKind.IDENTIFIER:
            return self._parse_expr_statement()
        elif token.kind == TokenKind.QUERY:
            return self._parse_query()

        return None

    def _parse_parallel(self) -> AIONASTNode:
        """Parse parallel block.

        parallel warp { body }
        parallel thread { body }
        """
        self._expect(TokenKind.PARALLEL)

        parallelism = "thread"
        if self._current().kind == TokenKind.WARP:
            self._advance()
            parallelism = "warp"
        elif self._current().kind in (TokenKind.THREAD, TokenKind.IDENTIFIER):
            parallelism = self._advance().value

        self._expect(TokenKind.LBRACE)
        self._skip_newlines()

        body = []
        while self._current().kind != TokenKind.RBRACE:
            stmt = self._parse_statement()
            if stmt:
                body.append(stmt)
            self._skip_newlines()

        self._expect(TokenKind.RBRACE)

        return AIONASTNode(
            kind="parallel",
            children=body,
            attributes={"parallelism": parallelism},
        )

    def _parse_reduce(self) -> AIONASTNode:
        """Parse reduce operation.

        reduce sum expr
        reduce max expr
        """
        self._expect(TokenKind.REDUCE)

        op = self._expect(TokenKind.IDENTIFIER).value
        expr = self._parse_expression()

        return AIONASTNode(
            kind="reduce",
            children=[expr],
            attributes={"op": op},
        )

    def _parse_region(self) -> AIONASTNode:
        """Parse region declaration.

        region Name : GPU_Stream0 Thread
        region Cache : FPGA_LUT Static
        """
        self._expect(TokenKind.REGION)

        name = self._expect(TokenKind.IDENTIFIER).value
        self._expect(TokenKind.COLON)

        # Parse hardware target
        hardware = "CPU"
        if (
            self._current().kind
            in (
                TokenKind.GPU,
                TokenKind.GPU_STREAM0,
                TokenKind.GPU_STREAM1,
                TokenKind.FPGA,
                TokenKind.FPGA_LUT,
                TokenKind.CPU,
            )
            or self._current().kind == TokenKind.IDENTIFIER
        ):
            hardware = self._advance().value

        # Parse lifetime/storage class
        storage = "dynamic"
        if self._current().kind in (TokenKind.THREAD, TokenKind.STATIC, TokenKind.IDENTIFIER):
            storage = self._advance().value

        return AIONASTNode(
            kind="region",
            value=name,
            attributes={"hardware": hardware, "storage": storage},
        )

    def _parse_move(self) -> AIONASTNode:
        """Parse move statement.

        move tensor into HotData
        """
        self._expect(TokenKind.MOVE)

        source = self._expect(TokenKind.IDENTIFIER).value
        self._expect(TokenKind.INTO)
        target = self._expect(TokenKind.IDENTIFIER).value

        return AIONASTNode(
            kind="move",
            value=source,
            attributes={"target_region": target},
        )

    def _parse_let(self) -> AIONASTNode:
        """Parse let binding.

        let x = expr
        """
        self._expect(TokenKind.LET)

        name = self._expect(TokenKind.IDENTIFIER).value

        # Optional type annotation
        type_hint = None
        if self._current().kind == TokenKind.COLON:
            self._advance()
            type_hint = self._expect(TokenKind.IDENTIFIER).value

        self._expect(TokenKind.ASSIGN)
        expr = self._parse_expression()

        return AIONASTNode(
            kind="let",
            value=name,
            children=[expr],
            attributes={"type": type_hint},
        )

    def _parse_function(self) -> AIONASTNode:
        """Parse function definition.

        fn name(params) { body }
        """
        self._expect(TokenKind.FN)

        name = self._expect(TokenKind.IDENTIFIER).value
        self._expect(TokenKind.LPAREN)

        # Parse parameters
        params = []
        while self._current().kind != TokenKind.RPAREN:
            param_name = self._expect(TokenKind.IDENTIFIER).value
            param_type = None
            if self._current().kind == TokenKind.COLON:
                self._advance()
                param_type = self._expect(TokenKind.IDENTIFIER).value
            params.append((param_name, param_type))

            if self._current().kind == TokenKind.COMMA:
                self._advance()

        self._expect(TokenKind.RPAREN)

        # Optional return type
        ret_type = None
        if self._current().kind == TokenKind.ARROW:
            self._advance()
            ret_type = self._expect(TokenKind.IDENTIFIER).value

        self._expect(TokenKind.LBRACE)
        self._skip_newlines()

        # Parse body
        body = []
        while self._current().kind != TokenKind.RBRACE:
            stmt = self._parse_statement()
            if stmt:
                body.append(stmt)
            self._skip_newlines()

        self._expect(TokenKind.RBRACE)

        return AIONASTNode(
            kind="function",
            value=name,
            children=body,
            attributes={"params": params, "return_type": ret_type},
        )

    def _parse_if(self) -> AIONASTNode:
        """Parse if statement."""
        self._expect(TokenKind.IF)

        condition = self._parse_expression()

        self._expect(TokenKind.LBRACE)
        self._skip_newlines()

        then_body = []
        while self._current().kind != TokenKind.RBRACE:
            stmt = self._parse_statement()
            if stmt:
                then_body.append(stmt)
            self._skip_newlines()

        self._expect(TokenKind.RBRACE)

        else_body = []
        if self._current().kind == TokenKind.ELSE:
            self._advance()
            self._expect(TokenKind.LBRACE)
            self._skip_newlines()

            while self._current().kind != TokenKind.RBRACE:
                stmt = self._parse_statement()
                if stmt:
                    else_body.append(stmt)
                self._skip_newlines()

            self._expect(TokenKind.RBRACE)

        return AIONASTNode(
            kind="if",
            children=[condition] + then_body,
            attributes={"else_body": else_body},
        )

    def _parse_while(self) -> AIONASTNode:
        """Parse while loop."""
        self._expect(TokenKind.WHILE)

        condition = self._parse_expression()

        self._expect(TokenKind.LBRACE)
        self._skip_newlines()

        body = []
        while self._current().kind != TokenKind.RBRACE:
            stmt = self._parse_statement()
            if stmt:
                body.append(stmt)
            self._skip_newlines()

        self._expect(TokenKind.RBRACE)

        return AIONASTNode(
            kind="while",
            children=[condition] + body,
        )

    def _parse_for(self) -> AIONASTNode:
        """Parse for loop."""
        self._expect(TokenKind.FOR)

        var = self._expect(TokenKind.IDENTIFIER).value
        self._expect(TokenKind.IN)
        iterable = self._parse_expression()

        self._expect(TokenKind.LBRACE)
        self._skip_newlines()

        body = []
        while self._current().kind != TokenKind.RBRACE:
            stmt = self._parse_statement()
            if stmt:
                body.append(stmt)
            self._skip_newlines()

        self._expect(TokenKind.RBRACE)

        return AIONASTNode(
            kind="for",
            value=var,
            children=[iterable] + body,
        )

    def _parse_return(self) -> AIONASTNode:
        """Parse return statement."""
        self._expect(TokenKind.RETURN)

        expr = None
        if self._current().kind not in (TokenKind.NEWLINE, TokenKind.RBRACE, TokenKind.EOF):
            expr = self._parse_expression()

        children = [expr] if expr else []
        return AIONASTNode(kind="return", children=children)

    def _parse_query(self) -> AIONASTNode:
        """Parse inline SQL query.

        query("SELECT * FROM table")
        """
        self._expect(TokenKind.QUERY)
        self._expect(TokenKind.LPAREN)
        sql = self._expect(TokenKind.STRING).value
        self._expect(TokenKind.RPAREN)

        return AIONASTNode(kind="query", value=sql)

    def _parse_expr_statement(self) -> AIONASTNode:
        """Parse expression statement (assignment or call)."""
        expr = self._parse_expression()

        # Check for assignment
        if self._current().kind == TokenKind.ASSIGN:
            self._advance()
            value = self._parse_expression()
            return AIONASTNode(
                kind="assign",
                value=expr.value if expr.kind == "identifier" else None,
                children=[value],
            )

        return AIONASTNode(kind="expr_stmt", children=[expr])

    def _parse_expression(self) -> AIONASTNode:
        """Parse expression."""
        return self._parse_comparison()

    def _parse_comparison(self) -> AIONASTNode:
        """Parse comparison expression."""
        left = self._parse_additive()

        while self._current().kind in (
            TokenKind.EQ,
            TokenKind.NE,
            TokenKind.LT,
            TokenKind.GT,
            TokenKind.LE,
            TokenKind.GE,
        ):
            op = self._advance().value
            right = self._parse_additive()
            left = AIONASTNode(
                kind="binary",
                value=op,
                children=[left, right],
            )

        return left

    def _parse_additive(self) -> AIONASTNode:
        """Parse additive expression."""
        left = self._parse_multiplicative()

        while self._current().kind in (TokenKind.PLUS, TokenKind.MINUS):
            op = self._advance().value
            right = self._parse_multiplicative()
            left = AIONASTNode(
                kind="binary",
                value=op,
                children=[left, right],
            )

        return left

    def _parse_multiplicative(self) -> AIONASTNode:
        """Parse multiplicative expression."""
        left = self._parse_unary()

        while self._current().kind in (TokenKind.STAR, TokenKind.SLASH, TokenKind.PERCENT):
            op = self._advance().value
            right = self._parse_unary()
            left = AIONASTNode(
                kind="binary",
                value=op,
                children=[left, right],
            )

        return left

    def _parse_unary(self) -> AIONASTNode:
        """Parse unary expression."""
        if self._current().kind == TokenKind.MINUS:
            self._advance()
            operand = self._parse_unary()
            return AIONASTNode(kind="unary", value="-", children=[operand])

        return self._parse_primary()

    def _parse_primary(self) -> AIONASTNode:
        """Parse primary expression."""
        token = self._current()

        if token.kind == TokenKind.INTEGER:
            self._advance()
            return AIONASTNode(kind="integer", value=token.value)

        elif token.kind == TokenKind.FLOAT:
            self._advance()
            return AIONASTNode(kind="float", value=token.value)

        elif token.kind == TokenKind.STRING:
            self._advance()
            return AIONASTNode(kind="string", value=token.value)

        elif token.kind == TokenKind.IDENTIFIER:
            name = self._advance().value

            # Check for function call
            if self._current().kind == TokenKind.LPAREN:
                return self._parse_call(name)

            # Check for member access
            if self._current().kind == TokenKind.DOT:
                self._advance()
                member = self._expect(TokenKind.IDENTIFIER).value
                return AIONASTNode(
                    kind="member",
                    value=member,
                    children=[AIONASTNode(kind="identifier", value=name)],
                )

            return AIONASTNode(kind="identifier", value=name)

        elif token.kind == TokenKind.LPAREN:
            self._advance()
            expr = self._parse_expression()
            self._expect(TokenKind.RPAREN)
            return expr

        raise SyntaxError(f"Unexpected token {token.kind} at line {token.line}")

    def _parse_call(self, name: str) -> AIONASTNode:
        """Parse function call."""
        self._expect(TokenKind.LPAREN)

        args = []
        while self._current().kind != TokenKind.RPAREN:
            args.append(self._parse_expression())
            if self._current().kind == TokenKind.COMMA:
                self._advance()

        self._expect(TokenKind.RPAREN)

        return AIONASTNode(kind="call", value=name, children=args)


class AIONCompiler:
    """Compiler from AION AST to SIR."""

    def __init__(self) -> None:
        """Initialize compiler."""
        self.graph = HyperGraph()
        self.variables: dict[str, Vertex] = {}
        self.regions: dict[str, str] = {}

    def compile(self, source: str) -> HyperGraph:
        """Compile AION source to SIR.

        Args:
            source: AION source code

        Returns:
            AION-SIR hypergraph
        """
        parser = AIONParser()
        ast = parser.parse(source)
        return self.compile_ast(ast)

    def compile_ast(self, ast: AIONASTNode) -> HyperGraph:
        """Compile AST to SIR.

        Args:
            ast: AION AST

        Returns:
            AION-SIR hypergraph
        """
        self.graph = HyperGraph(name="aion_program")

        for child in ast.children:
            self._compile_node(child)

        return self.graph

    def _compile_node(self, node: AIONASTNode) -> Vertex | None:
        """Compile an AST node to SIR vertices."""
        provenance = Provenance(
            source_language="AION",
            source_line=node.line,
        )

        if node.kind == "program":
            for child in node.children:
                self._compile_node(child)
            return None

        elif node.kind == "parallel":
            return self._compile_parallel(node, provenance)

        elif node.kind == "reduce":
            return self._compile_reduce(node, provenance)

        elif node.kind == "region":
            self._compile_region(node)
            return None

        elif node.kind == "move":
            return self._compile_move(node, provenance)

        elif node.kind == "let":
            return self._compile_let(node, provenance)

        elif node.kind == "function":
            return self._compile_function(node, provenance)

        elif node.kind == "assign":
            return self._compile_assign(node, provenance)

        elif node.kind == "binary":
            return self._compile_binary(node, provenance)

        elif node.kind == "call":
            return self._compile_call(node, provenance)

        elif node.kind == "integer" or node.kind == "float":
            return self._compile_literal(node, provenance)

        elif node.kind == "identifier":
            return self._compile_identifier(node, provenance)

        elif node.kind == "query":
            return self._compile_query(node, provenance)

        elif node.kind == "return":
            return self._compile_return(node, provenance)

        elif node.kind == "expr_stmt":
            if node.children:
                return self._compile_node(node.children[0])

        return None

    def _compile_parallel(self, node: AIONASTNode, provenance: Provenance) -> Vertex:
        """Compile parallel block."""
        parallelism = node.attributes.get("parallelism", "thread")

        # Compile body
        body_vertices = []
        for child in node.children:
            v = self._compile_node(child)
            if v:
                body_vertices.append(v)

        # Add parallel edge
        kind = ParallelismKind.SIMT if parallelism == "warp" else ParallelismKind.THREAD_LEVEL
        affinity = HardwareAffinity.GPU if parallelism == "warp" else HardwareAffinity.CPU

        if body_vertices:
            self.graph.add_edge(
                HyperEdge.parallel_edge(
                    body_vertices,
                    kind=kind,
                    warp_size=32 if parallelism == "warp" else 1,
                    affinity=affinity,
                )
            )

        return body_vertices[0] if body_vertices else None  # type: ignore

    def _compile_reduce(self, node: AIONASTNode, provenance: Provenance) -> Vertex:
        """Compile reduce operation."""
        op = node.attributes.get("op", "sum")
        expr = self._compile_node(node.children[0]) if node.children else None

        reduce_v = Vertex.apply(
            function_name=f"reduce_{op}",
            type_info=AIONType(kind="float"),
            effects={EffectKind.PURE},
            provenance=provenance,
        )
        self.graph.add_vertex(reduce_v)

        if expr:
            self.graph.add_edge(HyperEdge.data_flow(expr, reduce_v))

        return reduce_v

    def _compile_region(self, node: AIONASTNode) -> None:
        """Compile region declaration."""
        name = node.value
        hardware = node.attributes.get("hardware", "CPU")
        self.regions[name] = hardware

    def _compile_move(self, node: AIONASTNode, provenance: Provenance) -> Vertex:
        """Compile move into region."""
        source_name = node.value
        target_region = node.attributes.get("target_region", "heap")

        # Get hardware affinity from region
        hardware = self.regions.get(target_region, "CPU")
        affinity_map = {
            "GPU": HardwareAffinity.GPU,
            "GPU_Stream0": HardwareAffinity.GPU_STREAM0,
            "GPU_Stream1": HardwareAffinity.GPU_STREAM1,
            "FPGA": HardwareAffinity.FPGA,
            "FPGA_LUT": HardwareAffinity.FPGA_LUT,
            "CPU": HardwareAffinity.CPU,
        }
        affinity = affinity_map.get(hardware, HardwareAffinity.ANY)

        move_v = Vertex.apply(
            function_name="move",
            type_info=AIONType(kind="unit"),
            effects={EffectKind.WRITE},
            provenance=provenance,
        )
        move_v.attributes["source"] = source_name
        move_v.attributes["target_region"] = target_region
        move_v.metadata.hardware_affinity = affinity

        self.graph.add_vertex(move_v)

        if source_name in self.variables:
            self.graph.add_edge(HyperEdge.data_flow(self.variables[source_name], move_v))

        return move_v

    def _compile_let(self, node: AIONASTNode, provenance: Provenance) -> Vertex:
        """Compile let binding."""
        name = node.value

        value = None
        if node.children:
            value = self._compile_node(node.children[0])

        alloc = Vertex.alloc(
            size=8,
            type_info=AIONType(kind="unit"),
            region="stack",
            provenance=provenance,
        )
        self.graph.add_vertex(alloc)
        self.variables[name] = alloc

        if value:
            self.graph.add_edge(HyperEdge.data_flow(value, alloc))

        return alloc

    def _compile_function(self, node: AIONASTNode, provenance: Provenance) -> Vertex:
        """Compile function definition."""
        name = node.value
        params = node.attributes.get("params", [])

        entry = Vertex.parameter(
            name=f"{name}_entry",
            type_info=AIONType(kind="fn"),
            index=0,
            provenance=provenance,
        )
        self.graph.add_vertex(entry)
        self.graph.entry = entry

        # Add parameters
        for i, (param_name, _) in enumerate(params):
            param = Vertex.parameter(
                name=param_name,
                type_info=AIONType(kind="unit"),
                index=i,
                provenance=provenance,
            )
            self.graph.add_vertex(param)
            self.variables[param_name] = param

        # Compile body
        for child in node.children:
            self._compile_node(child)

        return entry

    def _compile_assign(self, node: AIONASTNode, provenance: Provenance) -> Vertex:
        """Compile assignment."""
        name = node.value

        value = None
        if node.children:
            value = self._compile_node(node.children[0])

        if name in self.variables:
            target = self.variables[name]
            store = Vertex.store(
                type_info=AIONType(kind="unit"),
                region="stack",
                provenance=provenance,
            )
            self.graph.add_vertex(store)
            if value:
                self.graph.add_edge(HyperEdge.data_flow([value, target], store))
            return store

        return value  # type: ignore

    def _compile_binary(self, node: AIONASTNode, provenance: Provenance) -> Vertex:
        """Compile binary operation."""
        op = node.value
        left = self._compile_node(node.children[0]) if len(node.children) > 0 else None
        right = self._compile_node(node.children[1]) if len(node.children) > 1 else None

        op_v = Vertex.apply(
            function_name=f"op_{op}",
            type_info=AIONType(kind="int"),
            effects={EffectKind.PURE},
            provenance=provenance,
        )
        self.graph.add_vertex(op_v)

        inputs = []
        if left:
            inputs.append(left)
        if right:
            inputs.append(right)

        if inputs:
            self.graph.add_edge(HyperEdge.data_flow(inputs, op_v))

        return op_v

    def _compile_call(self, node: AIONASTNode, provenance: Provenance) -> Vertex:
        """Compile function call."""
        name = node.value
        args = [self._compile_node(arg) for arg in node.children]

        call = Vertex.apply(
            function_name=name,
            type_info=AIONType(kind="unit"),
            effects={EffectKind.IO},
            provenance=provenance,
        )
        self.graph.add_vertex(call)

        if args:
            self.graph.add_edge(HyperEdge.data_flow([a for a in args if a], call))

        return call

    def _compile_literal(self, node: AIONASTNode, provenance: Provenance) -> Vertex:
        """Compile literal value."""
        ty = AIONType.int() if node.kind == "integer" else AIONType.float()
        return Vertex.const(node.value, ty, provenance)

    def _compile_identifier(self, node: AIONASTNode, provenance: Provenance) -> Vertex:
        """Compile identifier reference."""
        name = node.value

        if name in self.variables:
            var = self.variables[name]
            load = Vertex.load(
                type_info=AIONType(kind="unit"),
                region="stack",
                provenance=provenance,
            )
            self.graph.add_vertex(load)
            self.graph.add_edge(HyperEdge.data_flow(var, load))
            return load

        # Unknown variable - create placeholder
        return Vertex.const(None, AIONType(kind="unit"), provenance)

    def _compile_query(self, node: AIONASTNode, provenance: Provenance) -> Vertex:
        """Compile inline SQL query."""
        sql = node.value

        from .lifters.sql_lifter import SQLLifter

        lifter = SQLLifter()
        sql_graph = lifter.lift(sql)

        # Merge SQL graph into main graph
        for v in sql_graph.vertices:
            self.graph.add_vertex(v)
        for e in sql_graph.edges:
            self.graph.add_edge(e)

        # Return the output vertex
        if sql_graph.exits:
            return sql_graph.exits[0]

        # Create a placeholder
        query_v = Vertex.apply(
            function_name="sql_query",
            type_info=AIONType(kind="struct"),
            effects={EffectKind.READ},
            provenance=provenance,
        )
        query_v.attributes["sql"] = sql
        self.graph.add_vertex(query_v)
        return query_v

    def _compile_return(self, node: AIONASTNode, provenance: Provenance) -> Vertex:
        """Compile return statement."""
        value = None
        if node.children:
            value = self._compile_node(node.children[0])

        ret = Vertex.ret(AIONType(kind="unit"), provenance)
        self.graph.add_vertex(ret)
        self.graph.exits.append(ret)

        if value:
            self.graph.add_edge(HyperEdge.data_flow(value, ret))

        return ret
