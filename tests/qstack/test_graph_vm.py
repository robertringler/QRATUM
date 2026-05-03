from qstack.qnx.runtime import GraphVM, OperatorGraph, OperatorLibrary, QNXState


def test_graph_vm_topological_execution():
    ops = OperatorLibrary()
    state = QNXState({"count": 0})

    def inc(state_obj, goal):
        state_obj.update("count", state_obj.read("count", 0) + 1)
        return state_obj.read("count")

    def double(state_obj, goal):
        state_obj.update("count", state_obj.read("count") * 2)
        return state_obj.read("count")

    ops.register("increment", inc)
    ops.register("double", double)

    graph = OperatorGraph()
    graph.add_edge("increment", "double")

    vm = GraphVM(operators=ops, graph=graph)
    trace = vm.run(state, goal={})

    assert trace[0]["op"] == "increment"
    assert trace[1]["op"] == "double"
    assert trace[-1]["result"] == 2
    assert vm.replay_buffer()[0]["tick"] == 0
