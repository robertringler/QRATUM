"""QuASIM quantum simulation utilities built on top of libquasim."""
from __future__ import annotations

from quasim import Config, runtime


def simulate(circuit, *, precision: str = "fp8"):
    cfg = Config(simulation_precision=precision, max_workspace_mb=1024)
    tensors = [[complex(value) for value in gate] for gate in circuit]
    # TODO(K001): Hot call site for tensor contraction (K001)
    with runtime(cfg) as rt:
        return rt.simulate(tensors)
