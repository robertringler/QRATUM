from qnode.config import NodeConfig
from qnode.orchestration import NodeOrchestrator
from qnode.policies import NodePolicy
from qscenario.domains.aerospace import aerospace_launch_anomaly


def noop_policy(event, state):
    return True


def test_node_run_scenario():
    config = NodeConfig(node_id="n1", identity_ref="id-n1", allowed_syscalls=["simulate"])
    orchestrator = NodeOrchestrator(config, {"noop": NodePolicy("noop", noop_policy, "allow all")}, {})
    scenario = aerospace_launch_anomaly()
    report = orchestrator.run_scenario(scenario, required_metrics=["mission_steps"])
    assert report.outcome.classify() in {"success", "mixed"}
    assert report.state.metrics["mission_steps"] == 5
