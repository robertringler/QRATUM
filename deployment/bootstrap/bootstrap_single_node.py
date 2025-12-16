"""Bootstrap a single sovereign node logically."""

from __future__ import annotations

from qnode.config import NodeConfig
from qnode.identity_binding import NodeIdentity
from qnode.incident_log import IncidentLog
from qnode.lifecycle import NodeLifecycle
from qnode.monitor import HealthMonitor
from qnode.policies import budget_policy, syscall_allowlist_policy
from qnode.syscalls import SyscallRouter


def build_node() -> SyscallRouter:
    identity = NodeIdentity(node_id="node0", public_key="pk0")
    config = NodeConfig(
        node_id=identity.node_id,
        identity_ref=identity.public_key,
        allowed_syscalls=["echo"],
        policy_limits={"echo": 5},
    )
    log = IncidentLog()
    monitor = HealthMonitor(log)
    lifecycle = NodeLifecycle(monitor)
    lifecycle.start()
    policies = [syscall_allowlist_policy(config.allowed_syscalls), budget_policy(config.budget_for)]
    router = SyscallRouter(config, policies, log)
    router.register("echo", lambda payload: payload.get("message", ""))
    return router


if __name__ == "__main__":
    node = build_node()
    print(node.dispatch("echo", {"message": "hello", "cost": 1}))
