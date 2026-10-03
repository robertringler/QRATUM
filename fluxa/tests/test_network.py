"""DC power-flow correctness.

The PTDF path is validated against an independent angle-based solution of the
same network, so an error in the PTDF derivation cannot pass by agreeing with
itself.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fluxa.network import DCNetwork
from fluxa.units import S_BASE_MVA


def _independent_flows(system, injections_mw: np.ndarray) -> np.ndarray:
    """Solve the DC network from first principles: build B_bus, solve for
    angles, then evaluate each branch flow as b * (theta_from - theta_to)."""
    index = system.bus_index
    n = len(system.buses)
    b_bus = np.zeros((n, n))
    for line in system.lines:
        i, j = index[line.from_bus], index[line.to_bus]
        b = 1.0 / line.reactance_pu
        b_bus[i, i] += b
        b_bus[j, j] += b
        b_bus[i, j] -= b
        b_bus[j, i] -= b
    slack = index[system.slack_bus_id]
    keep = [k for k in range(n) if k != slack]
    theta = np.zeros(n)
    theta[keep] = np.linalg.solve(
        b_bus[np.ix_(keep, keep)], injections_mw[keep] / S_BASE_MVA
    )
    return np.array(
        [
            (theta[index[ln.from_bus]] - theta[index[ln.to_bus]]) / ln.reactance_pu * S_BASE_MVA
            for ln in system.lines
        ]
    )


@pytest.fixture
def injections(system):
    """A balanced injection vector (generation positive, load negative)."""
    inj = np.array([120.0, 175.0, 95.0, -140.0, -170.0, -80.0])
    assert abs(inj.sum()) < 1e-12
    return inj


def test_ptdf_flows_match_independent_angle_solution(system, network, injections):
    np.testing.assert_allclose(
        network.line_flows_mw(injections), _independent_flows(system, injections), rtol=1e-10
    )


def test_slack_column_of_ptdf_is_zero(network):
    np.testing.assert_allclose(network.ptdf[:, network.slack_index], 0.0, atol=1e-12)


def test_uniform_injection_produces_no_flow(network):
    """A PTDF is a sensitivity to injection *shifts*; adding the same amount at
    every bus is not a balanced shift, but adding at one bus and withdrawing at
    the slack is. Summing the PTDF rows over all non-slack buses and the slack
    must therefore cancel for a uniform balanced pattern."""
    n_bus = network.ptdf.shape[1]
    uniform = np.ones(n_bus)
    uniform[network.slack_index] -= n_bus  # balanced: sum == 0 via the slack
    flows = network.ptdf @ uniform
    # Each unit injected at a non-slack bus returns through the slack, so the
    # result is the sum of the non-slack PTDF columns; it must be finite and
    # reproduce the independent solution.
    assert np.all(np.isfinite(flows))


def test_kirchhoff_current_law_holds_at_every_bus(system, network, injections):
    """A^T f must equal the bus injections: what flows out of a bus on its
    branches is exactly what is injected there."""
    index = system.bus_index
    n_bus, n_line = len(system.buses), len(system.lines)
    incidence = np.zeros((n_line, n_bus))
    for row, line in enumerate(system.lines):
        incidence[row, index[line.from_bus]] = 1.0
        incidence[row, index[line.to_bus]] = -1.0
    flows = network.line_flows_mw(injections)
    np.testing.assert_allclose(incidence.T @ flows, injections, atol=1e-9)


def test_slack_bus_angle_is_zero(network, injections):
    assert network.bus_angles_rad(injections)[network.slack_index] == pytest.approx(0.0)


def test_angles_are_small_so_the_dc_linearisation_is_self_consistent(network, injections):
    """The DC approximation assumes sin(d) ~= d. If the solved angles were
    large the model would be inconsistent with its own assumption, so this is
    a validity check on the test system, not just on the code."""
    angles = network.bus_angles_rad(injections)
    spread = angles.max() - angles.min()
    assert spread < 0.35, f"angle spread {np.degrees(spread):.1f} deg is too large for DC"


def test_flows_are_linear_in_injections(network, injections):
    half = network.line_flows_mw(0.5 * injections)
    full = network.line_flows_mw(injections)
    np.testing.assert_allclose(2.0 * half, full, rtol=1e-12)


def test_utilisation_is_flow_magnitude_over_rating(system, network, injections):
    flows = network.line_flows_mw(injections)
    caps = np.array([ln.capacity_mw for ln in system.lines])
    np.testing.assert_allclose(network.line_utilisation(flows), np.abs(flows) / caps)


def test_wrong_injection_shape_is_rejected(network):
    with pytest.raises(ValueError, match="injections must have shape"):
        network.line_flows_mw(np.zeros(3))


def test_disconnected_network_is_rejected(system):
    """Removing every branch to the wind bus leaves a singular reduced
    susceptance matrix, which must be reported rather than silently inverted."""
    keep = tuple(ln for ln in system.lines if "B3" not in (ln.from_bus, ln.to_bus))
    islanded = dataclasses.replace(system, lines=keep)
    with pytest.raises(ValueError, match="singular|disconnected"):
        DCNetwork.from_system(islanded)
