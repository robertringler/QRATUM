from qstack.qunimbus.core.governance_vm import GovernanceRule, GovernanceVM


def test_governance_vm_scoring():
    vm = GovernanceVM()
    vm.register_rule(
        GovernanceRule(name="safety", condition_key="safety", threshold=0.5, weight=2.0)
    )
    vm.register_rule(
        GovernanceRule(name="compliance", condition_key="compliance", threshold=1.0, weight=1.0)
    )
    signals = {"safety": 0.7, "compliance": 1.2}
    decision = vm.decision(signals, min_score=2.0)
    assert decision["approved"]
    assert decision["score"] == 3.0
