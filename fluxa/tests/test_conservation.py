"""Conservation laws over a full simulated trajectory.

These are the physics acceptance tests. They re-derive each law from the
recorded state rather than from the LP, so a solver returning a wrong answer
would be caught here.
"""

from __future__ import annotations

import pytest

from fluxa.engine import validate_physical_state
from fluxa.tests.conftest import make_engine
from fluxa.units import ENERGY_TOLERANCE_MWH, POWER_TOLERANCE_MW


@pytest.fixture(scope="module")
def stressed_result(loaded, scenarios):
    """A compound-disturbance run: conservation must hold under stress too."""
    return make_engine(loaded, run_id="conservation-stress", n_steps=288).run(
        scenarios["F_COMPOUND"]
    )


def test_instantaneous_power_balance_every_step(baseline_result):
    for state in baseline_result.states:
        supply = (
            state.total_generation_mw
            + state.battery_discharge_mw
            + state.net_import_mw
            + state.unserved_mw
        )
        demand = state.total_load_mw + state.battery_charge_mw + state.net_export_mw
        assert supply == pytest.approx(demand, abs=1e-6), f"step {state.step}"


def test_power_balance_holds_under_compound_disturbance(stressed_result):
    for state in stressed_result.states:
        supply = (
            state.total_generation_mw
            + state.battery_discharge_mw
            + state.net_import_mw
            + state.unserved_mw
        )
        demand = state.total_load_mw + state.battery_charge_mw + state.net_export_mw
        assert supply == pytest.approx(demand, abs=1e-6), f"step {state.step}"


def test_balance_residual_is_numerically_negligible(baseline_result):
    worst = max(abs(s.balance_residual_mw) for s in baseline_result.states)
    assert worst < 1e-6, f"worst bus-injection residual {worst:.3e} MW"


def test_served_plus_unserved_equals_demand(baseline_result):
    for state in baseline_result.states:
        assert state.served_load_mw + state.unserved_mw == pytest.approx(
            state.total_load_mw, abs=1e-9
        )


def test_energy_conservation_over_the_whole_horizon(baseline_result):
    """Integrated: generation + storage withdrawal + net import == served load
    + storage injection, to within accumulated floating-point tolerance."""
    dt = baseline_result.run_config.timestep_h
    gen = sum(s.total_generation_mw for s in baseline_result.states) * dt
    discharge = sum(s.battery_discharge_mw for s in baseline_result.states) * dt
    charge = sum(s.battery_charge_mw for s in baseline_result.states) * dt
    imported = sum(s.net_import_mw for s in baseline_result.states) * dt
    exported = sum(s.net_export_mw for s in baseline_result.states) * dt
    served = sum(s.served_load_mw for s in baseline_result.states) * dt
    assert gen + discharge + imported == pytest.approx(
        served + charge + exported, rel=1e-9, abs=1e-6
    )


def test_storage_energy_conservation_step_by_step(baseline_result):
    system = baseline_result.system
    dt = baseline_result.run_config.timestep_h
    bat = system.batteries[0]
    previous = bat.soc_initial
    for state in baseline_result.states:
        expected = previous * bat.energy_capacity_mwh + (
            bat.charge_efficiency * state.battery_charge_mw * dt
            - state.battery_discharge_mw * dt / bat.discharge_efficiency
        )
        actual = state.battery_soc[bat.battery_id] * bat.energy_capacity_mwh
        assert actual == pytest.approx(expected, abs=ENERGY_TOLERANCE_MWH * 320.0), (
            f"step {state.step}"
        )
        previous = state.battery_soc[bat.battery_id]


def test_storage_energy_balance_closes_over_the_horizon(stressed_result):
    """The integrated storage balance must close exactly:

        E_final - E_initial == eta_ch * charge - discharge / eta_dis

    A non-zero residual would mean the storage model created or destroyed
    energy over the run, which per-step checks could miss if the error
    alternated sign.
    """
    ren = stressed_result.metrics.renewables
    capacity = sum(b.energy_capacity_mwh for b in stressed_result.system.batteries)
    assert abs(ren.storage_energy_balance_residual_mwh) < 1e-6 * capacity


def test_storage_delivers_no_more_than_stock_plus_charged_energy(stressed_result):
    """Discharged energy may exceed charged energy only by the stock drawn
    down from the initial state of charge, scaled by discharge efficiency."""
    ren = stressed_result.metrics.renewables
    bat = stressed_result.system.batteries[0]
    ceiling = (
        bat.charge_efficiency * ren.storage_charge_mwh + ren.storage_net_withdrawal_mwh
    ) * bat.discharge_efficiency
    assert ren.storage_discharge_mwh <= ceiling + 1e-6


def test_soc_never_leaves_the_declared_window(stressed_result):
    bat = stressed_result.system.batteries[0]
    for state in stressed_result.states:
        soc = state.battery_soc[bat.battery_id]
        assert bat.soc_min - 1e-9 <= soc <= bat.soc_max + 1e-9, f"step {state.step}: soc {soc}"


def test_no_generator_exceeds_its_nameplate(stressed_result):
    system = stressed_result.system
    for state in stressed_result.states:
        for gen in system.generators:
            p = state.generation_by_unit_mw[gen.gen_id]
            assert -POWER_TOLERANCE_MW <= p <= gen.p_max_mw + POWER_TOLERANCE_MW


def test_renewable_output_never_exceeds_available_resource(stressed_result):
    for state in stressed_result.states:
        assert state.renewable_generation_mw <= state.renewable_available_mw + POWER_TOLERANCE_MW
        assert state.curtailment_mw >= -POWER_TOLERANCE_MW


def test_curtailment_identity(stressed_result):
    for state in stressed_result.states:
        assert state.curtailment_mw == pytest.approx(
            max(state.renewable_available_mw - state.renewable_generation_mw, 0.0), abs=1e-9
        )


def test_generation_by_source_sums_to_total(baseline_result):
    for state in baseline_result.states:
        assert sum(state.generation_by_source_mw.values()) == pytest.approx(
            state.total_generation_mw, abs=1e-9
        )
        assert sum(state.generation_by_unit_mw.values()) == pytest.approx(
            state.total_generation_mw, abs=1e-9
        )


def test_reported_overload_matches_flow_and_rating(stressed_result):
    system = stressed_result.system
    for state in stressed_result.states:
        for line in system.lines:
            flow = abs(state.line_flow_mw[line.line_id])
            util = state.line_utilisation[line.line_id]
            # utilisation is defined against the step's (possibly derated)
            # rating, so rating = flow / utilisation where utilisation > 0.
            if util > 1e-9:
                rating = flow / util
                assert state.overload_mw[line.line_id] == pytest.approx(
                    max(flow - rating, 0.0), abs=1e-4
                )


def test_every_state_passes_the_engine_validator(baseline_result):
    """The validator used in-loop is re-applied to the recorded trajectory, so
    a state that only passes because of loop-local context would fail here."""
    import numpy as np

    system = baseline_result.system
    dt = baseline_result.run_config.timestep_h
    caps = np.array([ln.capacity_mw for ln in system.lines])
    previous = None
    for state in baseline_result.states:
        avail = np.array([state.renewable_available_mw])
        failures = validate_physical_state(system, state, previous, dt, caps, avail)
        assert failures == [], f"step {state.step}: {failures}"
        previous = state


def test_co2_accounting_matches_dispatch_and_emission_factors(baseline_result):
    system = baseline_result.system
    dt = baseline_result.run_config.timestep_h
    expected = sum(
        state.generation_by_unit_mw[g.gen_id] * dt * g.co2_tonnes_per_mwh
        for state in baseline_result.states
        for g in system.generators
    )
    assert baseline_result.metrics.economics.co2_tonnes == pytest.approx(expected, rel=1e-12)
