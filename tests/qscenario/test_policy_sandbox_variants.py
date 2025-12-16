from qscenario.domains.finance import finance_liquidity_crunch
from qscenario.policy_sandbox.policy_variant import PolicyVariant
from qscenario.policy_sandbox.sandbox import PolicySandbox


def base_policy(event, state):
    return True


def restrictive_policy(event, state):
    if event.kind == "contagion":
        state.record_incident("blocked", event.describe())
        return False
    return True


def test_policy_variants_comparison():
    sandbox = PolicySandbox(base_policies={"allow_all": base_policy})
    sandbox.variants.append(PolicyVariant(name="baseline"))
    sandbox.variants.append(
        PolicyVariant(name="restrictive", rules={"restrict": restrictive_policy})
    )

    comparison = sandbox.run(finance_liquidity_crunch)
    classifications = comparison.summarize()["classifications"]
    assert classifications[0] in {"mixed", "success", "failure"}
    assert classifications[1] in {"mixed", "failure", "success"}
