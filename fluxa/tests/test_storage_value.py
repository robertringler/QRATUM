"""The terminal stored-energy value dominates the storage result.

This is a measured limitation of the dispatch formulation, pinned by a test so
the report's claim cannot drift from the code. A receding-horizon LP must price
energy left in the battery at the end of each window, or it empties the battery
at the window edge as an artefact. The *value* chosen for that price then
determines the storage behaviour, and the base case's choice
(``stored_energy_value_usd_per_mwh = 48``, equal to the CCGT marginal cost) is
not the cost-minimising one.
"""

from __future__ import annotations

import dataclasses

import pytest

from fluxa.engine import RunConfig, SimulationEngine


def _run(loaded, scenario, stored_value: float, *, n_steps: int = 288):
    economics = dataclasses.replace(
        loaded.system.economics, stored_energy_value_usd_per_mwh=stored_value
    )
    variant = dataclasses.replace(
        loaded, system=dataclasses.replace(loaded.system, economics=economics)
    )
    config = RunConfig(run_id="vsoc", n_steps=n_steps, authorized=True)
    return SimulationEngine(variant, config).run(scenario)


@pytest.fixture(scope="module")
def sweep(loaded, scenarios):
    return {
        value: _run(loaded, scenarios["A_BASELINE"], value) for value in (48.0, 60.0)
    }


def test_base_case_battery_drains_to_its_floor_and_does_not_refill(sweep):
    """At the base-case stored-energy value the battery delivers its initial
    stock and then sits at ``soc_min``: recharging costs more than the stored
    energy is priced at, so the LP correctly refuses to do it. The consequence
    is that the reported storage contribution is an artefact of the price, not
    an optimal storage dispatch."""
    ren = sweep[48.0].metrics.renewables
    battery = sweep[48.0].system.batteries[0]
    assert ren.storage_soc_final == pytest.approx(battery.soc_min, abs=1e-6)
    assert ren.storage_charge_mwh < ren.storage_discharge_mwh / 4.0
    assert ren.storage_cycle_equivalents < 0.35


def test_raising_the_stored_energy_value_materially_increases_cycling(sweep):
    low, high = sweep[48.0].metrics.renewables, sweep[60.0].metrics.renewables
    assert high.storage_cycle_equivalents > 2.5 * low.storage_cycle_equivalents
    assert high.storage_soc_max > low.storage_soc_max + 0.4
    assert high.storage_contribution > 2.0 * low.storage_contribution


def test_the_base_case_is_not_the_cost_minimising_configuration(sweep):
    """The decisive negative result: a different value of a documented
    modelling parameter produces a cheaper dispatch of the same physical
    system. The base case's operating cost must therefore not be presented as
    an optimum."""
    low = sweep[48.0].metrics.economics.total_operating_cost_usd
    high = sweep[60.0].metrics.economics.total_operating_cost_usd
    assert high < low, "expected v_stored=60 to beat the base case"
    improvement = (low - high) / low
    assert improvement > 0.005, f"improvement {improvement:.4%} smaller than reported"


def test_physics_remains_valid_across_the_sweep(sweep):
    """The parameter changes the dispatch, never the conservation laws."""
    for result in sweep.values():
        ren = result.metrics.renewables
        capacity = sum(b.energy_capacity_mwh for b in result.system.batteries)
        assert abs(ren.storage_energy_balance_residual_mwh) < 1e-6 * capacity
        assert result.metrics.reliability.unserved_energy_mwh == 0.0
        for state in result.states:
            assert abs(state.balance_residual_mw) < 1e-6
