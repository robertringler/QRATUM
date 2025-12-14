
from qdl.lexer import Lexer
from qdl.parser import Parser


def test_lexer_tokens_basic():
    tokens = Lexer('1 + 2').tokenize()
    assert [t.type for t in tokens[:3]] == ['NUMBER', 'OP', 'NUMBER']


def test_parser_arithmetic():
    program = Parser.parse('1 + 2 * 3')
    result = program.evaluate({})
    assert result == 7
