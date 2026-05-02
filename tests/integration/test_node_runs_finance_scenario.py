from qnode.config import NodeConfig
from qnode.orchestration import NodeOrchestrator
from qnode.policies import NodePolicy
from qscenario.domains.finance import finance_liquidity_crunch


def allow(event, state):
    return True


def test_finance_scenario_through_node():
    orchestrator = NodeOrchestrator(
        NodeConfig("n2", "id-n2", ["finance"]), {"allow": NodePolicy("allow", allow, "permit")}, {}
    )
    scenario = finance_liquidity_crunch()
    report = orchestrator.run_scenario(scenario, required_metrics=["market_events"])
    assert report.outcome.metrics["market_events"] == 5
    assert report.outcome.classify() in {"success", "mixed"}
