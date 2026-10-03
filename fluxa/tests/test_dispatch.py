"""Dispatch-LP correctness: the solution returned must satisfy every modelled
constraint, and the solver's optimality claim must be checked, not trusted."""

from __future__ import annotations

import numpy as np
import pytest

from fluxa.dispatch import DispatchError, DispatchProblem, DispatchStep
from fluxa.units import POWER_TOLERANCE_MW


@pytest.fixture(scope="module")
def problem(system, network):
    return DispatchProblem(system, network, timestep_s=300.0, horizon_steps=6)


def _step(system, load_mw: float, solar_mw: float, wind_mw: float, avail=(1.0, 1.0)):
    shares = np.array([b.load_share for b in system.buses])
    return DispatchStep(
        bus_load_mw=load_mw * shares,
        variable_availability_mw=np.array([solar_mw, wind_mw]),
        dispatchable_availability=np.array(avail, dtype=float),
        line_capacity_mw=np.array([ln.capacity_mw for ln in system.lines]),
    )


def _window(system, problem, **kwargs):
    return [_step(system, **kwargs) for _ in range(problem.H)]


def _solve(system, problem, *, load_mw, solar_mw, wind_mw, soc=0.5, p_prev=None, avail=(1.0, 1.0)):
    p_prev = (
        np.array(p_prev, dtype=float)
        if p_prev is not None
        else np.array([g.p_min_mw for g in system.dispatchable_generators])
    )
    return problem.solve(
        _window(system, problem, load_mw=load_mw, solar_mw=solar_mw, wind_mw=wind_mw, avail=avail),
        np.array([soc]),
        p_prev,
        np.array(avail, dtype=float),
    )


def test_solver_reports_optimal_status(system, problem):
    sol = _solve(system, problem, load_mw=320.0, solar_mw=80.0, wind_mw=60.0)
    assert sol.solver_status == 0
    assert "optimal" in sol.solver_message.lower()


def test_power_balance_holds_exactly(system, problem):
    sol = _solve(system, problem, load_mw=320.0, solar_mw=80.0, wind_mw=60.0)
    supply = (
        sol.p_disp_mw.sum() + sol.p_var_mw.sum() + sol.p_discharge_mw.sum()
        + sol.p_import_mw.sum() + sol.unserved_mw.sum()
    )
    demand = 320.0 + sol.p_charge_mw.sum() + sol.p_export_mw.sum()
    assert supply == pytest.approx(demand, abs=1e-6)


def test_generator_output_respects_capacity_and_must_run_floor(system, problem):
    sol = _solve(system, problem, load_mw=400.0, solar_mw=0.0, wind_mw=0.0)
    for i, gen in enumerate(system.dispatchable_generators):
        assert sol.p_disp_mw[i] <= gen.p_max_mw + POWER_TOLERANCE_MW
        assert sol.p_disp_mw[i] >= gen.p_min_mw - POWER_TOLERANCE_MW


def test_variable_output_cannot_exceed_availability(system, problem):
    sol = _solve(system, problem, load_mw=420.0, solar_mw=40.0, wind_mw=25.0)
    assert sol.p_var_mw[0] <= 40.0 + POWER_TOLERANCE_MW
    assert sol.p_var_mw[1] <= 25.0 + POWER_TOLERANCE_MW


def test_unavailable_generator_is_forced_to_zero(system, problem):
    sol = _solve(system, problem, load_mw=300.0, solar_mw=60.0, wind_mw=60.0, avail=(0.0, 1.0))
    assert sol.p_disp_mw[0] == pytest.approx(0.0, abs=1e-9)


def test_generator_trip_is_feasible_despite_ramp_limit(system, problem):
    """A trip is not a ramp. Without the availability-dependent ramp
    relaxation this solve is infeasible, which would make forced-outage
    scenarios unsimulatable."""
    sol = _solve(
        system, problem, load_mw=300.0, solar_mw=60.0, wind_mw=60.0,
        p_prev=[200.0, 0.0], avail=(0.0, 1.0),
    )
    assert sol.solver_status == 0
    assert sol.p_disp_mw[0] == pytest.approx(0.0, abs=1e-9)


def test_ramp_limit_binds_when_unit_is_available(system, problem):
    """From 40 MW with a 4 MW/min ramp over a 5-minute step the CCGT cannot
    exceed 60 MW, even though demand would justify far more."""
    ccgt = system.dispatchable_generators[0]
    sol = _solve(system, problem, load_mw=420.0, solar_mw=0.0, wind_mw=0.0, p_prev=[40.0, 0.0])
    limit = 40.0 + ccgt.ramp_mw_per_min * 5.0
    assert sol.p_disp_mw[0] <= limit + 1e-6


def test_storage_conservation_matches_declared_efficiency(system, problem):
    bat = system.batteries[0]
    sol = _solve(system, problem, load_mw=420.0, solar_mw=0.0, wind_mw=0.0, soc=0.8)
    expected = 0.8 * bat.energy_capacity_mwh + (
        bat.charge_efficiency * sol.p_charge_mw[0] * (300.0 / 3600.0)
        - sol.p_discharge_mw[0] * (300.0 / 3600.0) / bat.discharge_efficiency
    )
    assert sol.soc_end[0] * bat.energy_capacity_mwh == pytest.approx(expected, abs=1e-9)


def test_soc_stays_within_declared_window(system, problem):
    bat = system.batteries[0]
    for soc in (bat.soc_min, 0.5, bat.soc_max):
        sol = _solve(system, problem, load_mw=420.0, solar_mw=0.0, wind_mw=0.0, soc=soc)
        assert bat.soc_min - 1e-9 <= sol.soc_end[0] <= bat.soc_max + 1e-9


def test_empty_battery_cannot_discharge(system, problem):
    bat = system.batteries[0]
    sol = _solve(system, problem, load_mw=420.0, solar_mw=0.0, wind_mw=0.0, soc=bat.soc_min)
    assert sol.p_discharge_mw[0] == pytest.approx(0.0, abs=1e-6)


def test_charge_and_discharge_are_not_simultaneous(system, problem):
    """No binary enforces exclusivity; round-trip loss plus cycle cost should
    make it strictly suboptimal. Verified rather than assumed."""
    for load in (260.0, 320.0, 380.0, 420.0):
        sol = _solve(system, problem, load_mw=load, solar_mw=90.0, wind_mw=70.0)
        assert sol.simultaneous_charge_discharge_mw < POWER_TOLERANCE_MW


def test_network_limits_are_respected_or_the_overload_is_reported(system, problem):
    sol = _solve(system, problem, load_mw=420.0, solar_mw=170.0, wind_mw=140.0)
    caps = np.array([ln.capacity_mw for ln in system.lines])
    excess = np.maximum(np.abs(sol.line_flow_mw) - caps, 0.0)
    np.testing.assert_allclose(sol.overload_mw, excess, atol=1e-4)


def test_line_flows_are_consistent_with_the_reported_injections(system, network, problem):
    sol = _solve(system, problem, load_mw=340.0, solar_mw=100.0, wind_mw=80.0)
    np.testing.assert_allclose(
        sol.line_flow_mw, network.line_flows_mw(sol.bus_injection_mw), atol=1e-9
    )


def test_injections_sum_to_zero(system, problem):
    sol = _solve(system, problem, load_mw=340.0, solar_mw=100.0, wind_mw=80.0)
    assert abs(sol.bus_injection_mw.sum()) < 1e-9


def test_battery_reserve_contribution_is_energy_tested(system, problem):
    """A nearly empty battery may claim only the reserve its stored energy can
    actually sustain for the policy's duration."""
    bat = system.batteries[0]
    soc = bat.soc_min + 0.01
    sol = _solve(system, problem, load_mw=300.0, solar_mw=60.0, wind_mw=60.0, soc=soc)
    deliverable = (
        (sol.soc_end[0] - bat.soc_min) * bat.energy_capacity_mwh
        * bat.discharge_efficiency / system.reserve.duration_h
    )
    assert sol.r_batt_mw[0] <= deliverable + 1e-6


def test_unserved_load_appears_only_when_capacity_is_exhausted(system, problem):
    comfortable = _solve(system, problem, load_mw=300.0, solar_mw=90.0, wind_mw=80.0)
    assert comfortable.unserved_mw.sum() < POWER_TOLERANCE_MW
    # Both gas units out, no sun, no wind: the battery and the tie cannot cover
    # a 420 MW peak, so shedding is the only feasible outcome.
    starved = _solve(
        system, problem, load_mw=420.0, solar_mw=0.0, wind_mw=0.0, avail=(0.0, 0.0)
    )
    assert starved.unserved_mw.sum() > 1.0


def test_cheaper_unit_is_dispatched_before_the_peaker(system, problem):
    sol = _solve(system, problem, load_mw=300.0, solar_mw=0.0, wind_mw=0.0, p_prev=[220.0, 0.0])
    ccgt, peaker = sol.p_disp_mw[0], sol.p_disp_mw[1]
    assert ccgt > peaker, f"merit order violated: CCGT {ccgt}, peaker {peaker}"


def test_wrong_window_length_is_rejected(system, problem):
    with pytest.raises(DispatchError, match="window must have"):
        problem.solve(
            _window(system, problem, load_mw=300.0, solar_mw=0.0, wind_mw=0.0)[:2],
            np.array([0.5]),
            np.array([40.0, 0.0]),
        )


def test_wrong_avail_prev_shape_is_rejected(system, problem):
    with pytest.raises(DispatchError, match="avail_prev must have shape"):
        problem.solve(
            _window(system, problem, load_mw=300.0, solar_mw=0.0, wind_mw=0.0),
            np.array([0.5]),
            np.array([40.0, 0.0]),
            np.array([1.0]),
        )


def test_horizon_must_be_at_least_one(system, network):
    with pytest.raises(ValueError, match="horizon_steps must be >= 1"):
        DispatchProblem(system, network, 300.0, horizon_steps=0)


def test_solver_metadata_records_versions_and_problem_size(problem):
    meta = problem.solver_metadata()
    assert meta["method"] == "highs"
    assert meta["library"] == "scipy.optimize.linprog"
    assert meta["scipy_version"] and meta["numpy_version"]
    assert meta["n_variables"] == problem.n_vars
    assert meta["n_equality_rows"] > 0 and meta["n_inequality_rows"] > 0


def test_single_step_horizon_reduces_to_myopic_dispatch(system, network):
    myopic = DispatchProblem(system, network, 300.0, horizon_steps=1)
    sol = _solve(system, myopic, load_mw=320.0, solar_mw=80.0, wind_mw=60.0)
    assert sol.solver_status == 0
    assert myopic.n_vars == myopic.vars_per_step
