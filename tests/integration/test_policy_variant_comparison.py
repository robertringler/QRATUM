from qscenario.domains.network import network_partition
from qscenario.policy_sandbox.policy_variant import PolicyVariant
from qscenario.policy_sandbox.sandbox import PolicySandbox


def allow(event, state):
    return True


def block_partition(event, state):
    if event.kind == "partition":
        state.record_incident("blocked_partition", event.describe())
        return False
    return True


def test_variant_comparison_selects_best():
    sandbox = PolicySandbox(base_policies={"allow": allow})
    sandbox.variants.append(PolicyVariant(name="allow_all"))
    sandbox.variants.append(PolicyVariant(name="block_partition", rules={"block": block_partition}))
    comparison = sandbox.run(network_partition)
    best = comparison.best_outcome()
    assert best.outcome is not None
    assert len(comparison.reports) == 2
