from qdl.evaluator import Evaluator
from qdl.parser import Parser


class WM:
    def add(self, a, b):
        return a + b


def test_worldmodel_call_evaluates():
    program = Parser.parse("worldmodel:add(1, 2)")
    evaluator = Evaluator({"worldmodel": WM()})
    assert evaluator.evaluate(program) == 3
