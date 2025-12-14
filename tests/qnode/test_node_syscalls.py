from qnode.config import NodeConfig
from qnode.incident_log import IncidentLog
from qnode.policies import budget_policy, syscall_allowlist_policy
from qnode.syscalls import SyscallError, SyscallRouter


def test_syscall_policy_enforcement():
    config = NodeConfig(node_id="n1", identity_ref="id", allowed_syscalls=["ping"], policy_limits={"ping": 2})
    log = IncidentLog()
    policies = [syscall_allowlist_policy(config.allowed_syscalls), budget_policy(config.budget_for)]
    router = SyscallRouter(config, policies, log)
    router.register("ping", lambda payload: "pong")

    assert router.dispatch("ping", {"cost": 1}) == "pong"
    try:
        router.dispatch("ping", {"cost": 3})
    except SyscallError as exc:
        assert "budget" in str(exc)
    else:
        raise AssertionError("expected budget violation")
    assert log.recent(1)[0].category == "policy_violation"
