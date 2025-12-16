from qdl.qir.ir_nodes import Constant, Graph, Operation
from qdl.qir.optimizer import optimize


def test_constant_folding():
    graph = Graph([Operation("add", [Constant(1), Constant(2)])])
    optimized = optimize(graph)
    assert isinstance(optimized.outputs[0], Constant)
    assert optimized.outputs[0].value == 3
