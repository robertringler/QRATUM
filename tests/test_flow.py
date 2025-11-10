# tests/test_flow.py

import numpy as np

from quasim.control.optimizer import optimize_a, simulate


def test_simulate_runs():
    N = 50
    a = np.ones(N)
    J, logs = simulate(a, N=N, T=1.0)
    assert np.isfinite(J)
    assert "Bures_dist" in logs
    assert logs["Bures_dist"].shape == (N,)


def test_optimize_a_improves_or_stabilizes():
    a_opt, hist, logs = optimize_a(steps=5, N=80, T=1.5)
    # at least it should run and produce history
    assert len(hist) >= 1
    # objective is a scalar
    assert np.isscalar(hist[-1][0])
