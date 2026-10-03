"""Metric definitions. A metric that silently means something other than its
name is worse than a missing metric."""

from __future__ import annotations

import math

import pytest

from fluxa.metrics import (
    ReliabilityMetrics,
    compute_all,
    compute_economics,
    compute_reliability,
    compute_renewables,
)
from fluxa.state import OperationalState
from fluxa.tests.conftest import make_engine


@pytest.fixture(scope="module")
def disturbed(loaded, scenarios):
    return make_engine(loaded, run_id="metrics-disturbed", n_steps=288, authorized=True).run(
        scenarios["F_COMPOUND"]
    )


def test_energy_demanded_equals_served_plus_unserved(disturbed):
    rel = disturbed.metrics.reliability
    assert rel.energy_demanded_mwh == pytest.approx(
        rel.energy_served_mwh + rel.unserved_energy_mwh, rel=1e-12
    )


def test_unserved_fraction_is_consistent(disturbed):
    rel = disturbed.metrics.reliability
    assert rel.unserved_energy_fraction == pytest.approx(
        rel.unserved_energy_mwh / rel.energy_demanded_mwh, rel=1e-12
    )


def test_operational_state_counts_sum_to_the_step_count(disturbed):
    rel = disturbed.metrics.reliability
    assert sum(rel.steps_by_operational_state.values()) == rel.n_steps == len(disturbed.states)


def test_violation_durations_are_multiples_of_the_timestep(disturbed):
    rel = disturbed.metrics.reliability
    dt_min = disturbed.run_config.timestep_h * 60.0
    for duration in (
        rel.violation_duration_min,
        rel.unserved_duration_min,
        rel.overload_duration_min,
    ):
        assert duration % dt_min == pytest.approx(0.0, abs=1e-9)


def test_unserved_duration_matches_the_counted_steps(disturbed):
    rel = disturbed.metrics.reliability
    dt_min = disturbed.run_config.timestep_h * 60.0
    counted = sum(1 for s in disturbed.states if s.unserved_mw > 1e-6)
    assert rel.unserved_duration_min == pytest.approx(counted * dt_min)


def test_baseline_has_no_violations_and_no_unserved_energy(baseline_result):
    rel = baseline_result.metrics.reliability
    assert rel.unserved_energy_mwh == 0.0
    assert rel.n_steps_with_hard_violation == 0
    assert rel.max_line_overload_mw == 0.0
    assert rel.steps_by_operational_state == {OperationalState.NORMAL.value: rel.n_steps}


def test_disturbance_increases_unserved_energy_over_the_baseline(baseline_result, disturbed):
    assert (
        disturbed.metrics.reliability.unserved_energy_mwh
        > baseline_result.metrics.reliability.unserved_energy_mwh
    )


def test_operating_cost_excludes_penalties_and_voll(disturbed):
    """Operating cost must be a production cost. Mixing in a soft-constraint
    penalty or VOLL would make the economic result meaningless."""
    econ = disturbed.metrics.economics
    expected = (
        econ.generation_cost_usd
        + econ.storage_cost_usd
        + econ.import_cost_usd
        - econ.export_revenue_usd
    )
    assert econ.total_operating_cost_usd == pytest.approx(expected, rel=1e-9)
    assert econ.unserved_energy_cost_usd > 0.0
    assert econ.total_cost_with_externalities_usd > econ.total_operating_cost_usd


def test_generation_cost_by_source_sums_to_the_total(disturbed):
    econ = disturbed.metrics.economics
    assert sum(econ.generation_cost_by_source_usd.values()) == pytest.approx(
        econ.generation_cost_usd, rel=1e-12
    )


def test_unserved_cost_is_voll_times_unserved_energy(disturbed):
    econ = disturbed.metrics.economics
    rel = disturbed.metrics.reliability
    voll = disturbed.system.economics.value_of_lost_load_usd_per_mwh
    assert econ.unserved_energy_cost_usd == pytest.approx(
        rel.unserved_energy_mwh * voll, rel=1e-12
    )


def test_average_cost_per_mwh_is_in_a_physically_sensible_band(baseline_result):
    econ = baseline_result.metrics.economics
    assert 0.0 < econ.average_cost_usd_per_mwh_served < 200.0


def test_renewable_utilisation_is_generation_over_availability(disturbed):
    ren = disturbed.metrics.renewables
    assert ren.renewable_utilisation == pytest.approx(
        ren.renewable_generation_mwh / ren.renewable_available_mwh, rel=1e-12
    )
    assert 0.0 <= ren.renewable_utilisation <= 1.0 + 1e-12


def test_curtailment_and_generation_account_for_all_availability(disturbed):
    ren = disturbed.metrics.renewables
    assert ren.renewable_generation_mwh + ren.curtailment_mwh == pytest.approx(
        ren.renewable_available_mwh, rel=1e-9
    )


def test_curtailment_fraction_is_consistent(loaded, scenarios):
    result = make_engine(loaded, run_id="metrics-curtail", n_steps=288).run(
        scenarios["G_LINE_DERATE"]
    )
    ren = result.metrics.renewables
    assert ren.curtailment_mwh > 0.0, "the congestion scenario must actually curtail"
    assert ren.curtailment_fraction == pytest.approx(
        ren.curtailment_mwh / ren.renewable_available_mwh, rel=1e-12
    )
    assert ren.renewable_utilisation + ren.curtailment_fraction == pytest.approx(1.0, rel=1e-9)


def test_storage_cycle_equivalents_are_throughput_over_twice_capacity(disturbed):
    ren = disturbed.metrics.renewables
    capacity = sum(b.energy_capacity_mwh for b in disturbed.system.batteries)
    assert ren.storage_cycle_equivalents == pytest.approx(
        ren.storage_throughput_mwh / (2.0 * capacity), rel=1e-12
    )


def test_resilience_fields_are_consistent_for_an_undisturbed_run(baseline_result):
    res = baseline_result.metrics.resilience
    assert res.disturbance_onset_h is None
    assert res.detection_step is None
    assert res.episode_had_degradation is False
    assert res.stabilization_step is None
    assert res.recovery_time_min is None
    assert res.stabilised is True
    assert res.energy_deficit_mwh == 0.0


def test_resilience_ordering_under_a_disturbance(disturbed):
    res = disturbed.metrics.resilience
    assert res.disturbance_onset_step is not None
    assert res.detection_step is not None and res.detection_step >= res.disturbance_onset_step
    assert res.first_degraded_step is not None
    assert res.first_violation_step is not None
    assert res.first_violation_step >= res.disturbance_onset_step
    assert res.stabilization_step is not None
    assert res.stabilization_step > res.first_violation_step
    assert res.recovery_time_min is not None and res.recovery_time_min > 0.0


def test_energy_deficit_does_not_exceed_total_unserved_energy(disturbed):
    assert (
        disturbed.metrics.resilience.energy_deficit_mwh
        <= disturbed.metrics.reliability.unserved_energy_mwh + 1e-9
    )


def test_detection_delay_is_at_most_one_timestep_for_a_ramped_perturbation(disturbed):
    """A trapezoid has zero intensity exactly at its onset, so the earliest
    observable step is onset+1. A larger delay would mean the detection
    threshold is masking the disturbance."""
    res = disturbed.metrics.resilience
    assert res.detection_delay_min == pytest.approx(disturbed.run_config.timestep_h * 60.0)


def test_frequency_proxy_out_of_range_steps_are_counted(disturbed):
    rel = disturbed.metrics.reliability
    assert rel.n_steps_frequency_proxy_out_of_range > 0
    assert rel.n_steps_frequency_proxy_out_of_range == sum(
        1 for s in disturbed.states if not s.frequency_proxy_in_range
    )


def test_baseline_frequency_proxy_stays_in_range(baseline_result):
    assert baseline_result.metrics.reliability.n_steps_frequency_proxy_out_of_range == 0
    assert baseline_result.metrics.reliability.max_frequency_deviation_hz == 0.0


def test_metrics_reject_an_empty_trajectory(system):
    with pytest.raises(ValueError, match="empty trajectory"):
        compute_all(
            system, [], 1 / 12, scenario_id="X", run_id="X",
            onset_h=None, detection_step=None, stabilization_steps=6,
        )


def test_no_standard_reliability_index_is_fabricated():
    """SAIDI/SAIFI/CAIDI/ENS/LOLP/LOLE/EUE have regulatory definitions this
    model does not represent. Reporting one would be a fabrication, so none
    may appear in the metric schema."""
    forbidden = {"saidi", "saifi", "caidi", "ens", "lolp", "lole", "eue", "asai"}
    for cls in (ReliabilityMetrics,):
        for field in cls.__dataclass_fields__:
            tokens = set(field.lower().split("_"))
            assert not (tokens & forbidden), f"{cls.__name__}.{field} names a standard index"


def test_metric_payload_excludes_the_run_label(disturbed):
    payload = disturbed.metrics.output_payload()
    assert "run_id" not in payload
    assert payload["scenario_id"] == disturbed.scenario.scenario_id


def test_reserve_margin_is_finite_or_explicitly_infinite(disturbed):
    for state in disturbed.states:
        assert math.isfinite(state.reserve_margin) or math.isinf(state.reserve_margin)


def test_metric_families_can_be_computed_independently(disturbed):
    dt = disturbed.run_config.timestep_h
    system, states = disturbed.system, disturbed.states
    assert compute_reliability(system, states, dt).n_steps == len(states)
    assert compute_economics(system, states, dt).generation_cost_usd > 0.0
    assert compute_renewables(system, states, dt).renewable_available_mwh > 0.0
