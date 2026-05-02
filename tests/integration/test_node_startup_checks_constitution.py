import pytest

from qconstitution.validator import ValidationError
from qnode.config import NodeConfig
from qnode.incident_log import IncidentLog
from qnode.lifecycle import NodeLifecycle
from qnode.monitor import HealthMonitor


def test_node_startup_requires_syscalls():
    monitor = HealthMonitor(IncidentLog())
    lifecycle = NodeLifecycle(
        monitor=monitor, config=NodeConfig(node_id="n1", identity_ref="id", allowed_syscalls=[])
    )
    with pytest.raises(ValidationError):
        lifecycle.start()


def test_node_startup_passes_with_syscalls():
    monitor = HealthMonitor(IncidentLog())
    lifecycle = NodeLifecycle(
        monitor=monitor,
        config=NodeConfig(node_id="n1", identity_ref="id", allowed_syscalls=["read"]),
    )
    lifecycle.start()
    assert lifecycle.state == "running"
