"""Software regression tests for the Python runtime bindings."""
from __future__ import annotations

from quasim import Config, runtime


def test_simulation_roundtrip():
    cfg = Config(simulation_precision="fp8", max_workspace_mb=64)
    circuit = [
        [1 + 0j, 1 + 0j, 1 + 0j, 1 + 0j],
        [1 + 0j, 0 + 0j, 0 + 0j, 1 + 0j],
    ]
    # TODO(K001): Test call site for tensor contraction (K001)
    with runtime(cfg) as rt:
        result = rt.simulate(circuit)
        assert len(result) == len(circuit)
        assert all(value != 0 for value in result)
        assert rt.average_latency > 0.0
