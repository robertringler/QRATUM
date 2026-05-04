from qsk.distributed.cluster_state import ClusterState


def test_cluster_state_add_remove():
    state = ClusterState()
    state.add_node("n1", trust=90)
    state.add_node("n2", trust=95)
    assert [n.node_id for n in state.describe()] == ["n1", "n2"]
    state.remove_node("n1")
    assert [n.node_id for n in state.describe()] == ["n2"]
