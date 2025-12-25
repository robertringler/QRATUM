from qunimbus.node_governance.node_policies import (evaluate_node_policies,
                                                    rank_nodes)


def test_node_policy_scoring_and_ranking():
    metrics_a = {"errors": 0}
    metrics_b = {"errors": 3}
    expectations = {"errors": 1}
    result_a = evaluate_node_policies(metrics_a, expectations)
    result_b = evaluate_node_policies(metrics_b, expectations)
    ranking = rank_nodes([("a", result_a), ("b", result_b)])
    assert ranking[0][0] == "a"
    assert result_b.score < result_a.score
