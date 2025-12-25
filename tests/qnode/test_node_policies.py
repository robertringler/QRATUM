from qnode.policies import (budget_policy, compose_policies,
                            syscall_allowlist_policy)


def test_policy_composition():
    policies = [syscall_allowlist_policy(["a", "b"]), budget_policy(lambda name: 2)]
    allowed = compose_policies(policies, {"syscall": "a", "cost": 1})
    blocked = compose_policies(policies, {"syscall": "c", "cost": 1})
    over_budget = compose_policies(policies, {"syscall": "a", "cost": 5})

    assert allowed.allowed
    assert not blocked.allowed
    assert not over_budget.allowed
