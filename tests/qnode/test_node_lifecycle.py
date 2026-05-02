from qnode.incident_log import IncidentLog
from qnode.lifecycle import NodeLifecycle
from qnode.monitor import HealthMonitor


def test_lifecycle_transitions():
    log = IncidentLog()
    monitor = HealthMonitor(log)
    lifecycle = NodeLifecycle(monitor)
    lifecycle.start()
    lifecycle.pause()
    lifecycle.resume()
    lifecycle.shutdown()
    assert lifecycle.state == "shutdown"
    assert lifecycle.trace() == [
        "init->running",
        "running->paused",
        "paused->running",
        "running->shutdown",
    ]
    assert monitor.metrics.snapshot()["start"] == 1
