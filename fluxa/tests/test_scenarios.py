"""Scenario-engine correctness: perturbation shapes, application, and the
safety classification that gates them."""

from __future__ import annotations

import numpy as np
import pytest

from fluxa.scenarios import (
    Perturbation,
    PerturbationKind,
    Scenario,
    ScenarioApplicationError,
    apply_scenario,
    max_safety_level,
    rescale_onsets,
    scenario_from_dict,
)


@pytest.fixture
def hours():
    return np.arange(0, 24, 1.0 / 12.0)


def test_trapezoid_rises_holds_and_recovers(hours):
    pert = Perturbation(
        "P", PerturbationKind.RENEWABLE_DERATE, start_h=10.0,
        ramp_min=30.0, hold_min=60.0, recovery_min=30.0,
        magnitude=0.5, targets=("SOLAR_PV_1",),
    )
    i = pert.intensity(hours)
    assert i.min() == 0.0 and i.max() == pytest.approx(1.0)
    assert i[hours < 10.0].max() == 0.0, "intensity before onset must be zero"
    assert pert.intensity(np.array([10.0]))[0] == pytest.approx(0.0)
    assert pert.intensity(np.array([10.25]))[0] == pytest.approx(0.5)
    assert pert.intensity(np.array([10.5]))[0] == pytest.approx(1.0)
    assert pert.intensity(np.array([11.5]))[0] == pytest.approx(1.0)
    assert pert.intensity(np.array([11.75]))[0] == pytest.approx(0.5)
    assert pert.intensity(np.array([12.1]))[0] == pytest.approx(0.0)


def test_zero_recovery_means_the_perturbation_persists(hours):
    pert = Perturbation(
        "P", PerturbationKind.LOAD_SCALE, start_h=6.0,
        ramp_min=0.0, hold_min=0.0, recovery_min=0.0, magnitude=0.2,
    )
    i = pert.intensity(hours)
    assert i[hours >= 6.0].min() == pytest.approx(1.0)
    assert i[hours < 6.0].max() == 0.0


def test_intensity_is_bounded_to_unit_interval(hours):
    pert = Perturbation(
        "P", PerturbationKind.RENEWABLE_DERATE, start_h=1.0, ramp_min=10.0,
        hold_min=10.0, recovery_min=10.0, magnitude=1.0, targets=("WIND_1",),
    )
    i = pert.intensity(hours)
    assert i.min() >= 0.0 and i.max() <= 1.0


def test_renewable_derate_reduces_availability_by_the_declared_depth(system, series):
    pert = Perturbation(
        "PV70", PerturbationKind.RENEWABLE_DERATE, start_h=0.0,
        ramp_min=0.0, hold_min=0.0, recovery_min=0.0,
        magnitude=0.7, targets=("SOLAR_PV_1",),
    )
    out = apply_scenario(system, series, Scenario("S", "s", "", (pert,)))
    np.testing.assert_allclose(
        out.availability_mw["SOLAR_PV_1"], series.availability_mw["SOLAR_PV_1"] * 0.3, rtol=1e-12
    )
    np.testing.assert_allclose(
        out.availability_mw["WIND_1"], series.availability_mw["WIND_1"], rtol=1e-12
    )


def test_generator_outage_zeroes_dispatchable_availability(system, series):
    pert = Perturbation(
        "TRIP", PerturbationKind.GENERATOR_OUTAGE, start_h=0.0, targets=("GAS_CCGT_1",)
    )
    out = apply_scenario(system, series, Scenario("S", "s", "", (pert,)))
    assert out.dispatchable_availability["GAS_CCGT_1"].max() == 0.0
    assert out.dispatchable_availability["GAS_GT_2"].min() == 1.0


def test_load_scale_multiplies_demand_and_preserves_bus_shares(system, series):
    pert = Perturbation("UP", PerturbationKind.LOAD_SCALE, start_h=0.0, magnitude=0.25)
    out = apply_scenario(system, series, Scenario("S", "s", "", (pert,)))
    np.testing.assert_allclose(out.total_load_mw, series.total_load_mw * 1.25, rtol=1e-12)
    np.testing.assert_allclose(out.bus_load_mw.sum(axis=1), out.total_load_mw, rtol=1e-12)


def test_line_derate_reduces_only_the_named_line(system, series):
    pert = Perturbation(
        "D", PerturbationKind.LINE_DERATE, start_h=0.0, magnitude=0.5, targets=("L5",)
    )
    out = apply_scenario(system, series, Scenario("S", "s", "", (pert,)))
    index = {ln.line_id: i for i, ln in enumerate(system.lines)}
    assert out.line_capacity_mw[0, index["L5"]] == pytest.approx(75.0)
    assert out.line_capacity_mw[0, index["L6"]] == pytest.approx(180.0)


def test_apply_scenario_does_not_mutate_the_baseline(system, series):
    before_load = series.total_load_mw.copy()
    before_solar = series.availability_mw["SOLAR_PV_1"].copy()
    apply_scenario(
        system,
        series,
        Scenario(
            "S", "s", "",
            (
                Perturbation("U", PerturbationKind.LOAD_SCALE, start_h=0.0, magnitude=0.5),
                Perturbation(
                    "D", PerturbationKind.RENEWABLE_DERATE, start_h=0.0,
                    magnitude=0.9, targets=("SOLAR_PV_1",),
                ),
            ),
        ),
    )
    np.testing.assert_array_equal(series.total_load_mw, before_load)
    np.testing.assert_array_equal(series.availability_mw["SOLAR_PV_1"], before_solar)


def test_compound_perturbations_all_apply(system, series, scenarios):
    out = apply_scenario(system, series, scenarios["F_COMPOUND"])
    assert len(out.perturbation_intensity) == 3
    assert set(out.perturbation_intensity) == {
        "WIND_COLLAPSE_80", "CCGT_TRIP_COMPOUND", "LOAD_SURGE_25"
    }


def test_unknown_generator_target_is_rejected(system, series):
    pert = Perturbation(
        "X", PerturbationKind.GENERATOR_OUTAGE, start_h=0.0, targets=("NO_SUCH_UNIT",)
    )
    with pytest.raises(ScenarioApplicationError, match="unknown generator"):
        apply_scenario(system, series, Scenario("S", "s", "", (pert,)))


def test_unknown_line_target_is_rejected(system, series):
    pert = Perturbation(
        "X", PerturbationKind.LINE_DERATE, start_h=0.0, magnitude=0.5, targets=("L99",)
    )
    with pytest.raises(ScenarioApplicationError, match="unknown line"):
        apply_scenario(system, series, Scenario("S", "s", "", (pert,)))


def test_derating_a_dispatchable_unit_as_renewable_is_rejected(system, series):
    pert = Perturbation(
        "X", PerturbationKind.RENEWABLE_DERATE, start_h=0.0,
        magnitude=0.5, targets=("GAS_CCGT_1",),
    )
    with pytest.raises(ScenarioApplicationError, match="not a variable-renewable"):
        apply_scenario(system, series, Scenario("S", "s", "", (pert,)))


def test_derate_requires_targets():
    with pytest.raises(ValueError, match="derate requires explicit targets"):
        Perturbation("X", PerturbationKind.RENEWABLE_DERATE, start_h=0.0, magnitude=0.5)


def test_load_scale_rejects_targets():
    with pytest.raises(ValueError, match="LOAD_SCALE is system-wide"):
        Perturbation(
            "X", PerturbationKind.LOAD_SCALE, start_h=0.0, magnitude=0.5, targets=("L1",)
        )


def test_derate_magnitude_must_be_a_fraction():
    with pytest.raises(ValueError, match="derate magnitude must be in"):
        Perturbation(
            "X", PerturbationKind.RENEWABLE_DERATE, start_h=0.0,
            magnitude=1.5, targets=("WIND_1",),
        )


def test_negative_onset_is_rejected():
    with pytest.raises(ValueError, match="start_h must be >= 0"):
        Perturbation("X", PerturbationKind.LOAD_SCALE, start_h=-1.0, magnitude=0.1)


def test_generator_outage_is_classified_sensitive(scenarios):
    assert scenarios["D_GENERATOR_OUTAGE"].required_safety_level == "SENSITIVE"
    assert scenarios["F_COMPOUND"].required_safety_level == "SENSITIVE"


def test_baseline_is_routine_and_derates_are_elevated(scenarios):
    assert scenarios["A_BASELINE"].required_safety_level == "ROUTINE"
    assert scenarios["B_SOLAR_SHOCK"].required_safety_level == "ELEVATED"
    assert scenarios["E_LOAD_SPIKE"].required_safety_level == "ELEVATED"


def test_no_scenario_claims_critical_or_existential(scenarios):
    """An ordinary grid contingency is not an existential risk. The scenario
    library must not inflate its own safety classification."""
    for scenario in scenarios.values():
        assert scenario.required_safety_level in ("ROUTINE", "ELEVATED", "SENSITIVE")


def test_max_safety_level_respects_the_declared_order():
    assert max_safety_level(["ROUTINE", "SENSITIVE", "ELEVATED"]) == "SENSITIVE"
    assert max_safety_level([]) == "ROUTINE"
    with pytest.raises(ValueError, match="unknown safety level"):
        max_safety_level(["WHATEVER"])


def test_scenario_hash_is_content_addressed(scenarios):
    a = scenarios["A_BASELINE"]
    assert a.scenario_hash() == a.scenario_hash()
    assert a.scenario_hash() != scenarios["B_SOLAR_SHOCK"].scenario_hash()


def test_scenario_hash_changes_when_a_perturbation_changes(scenarios):
    import dataclasses

    original = scenarios["B_SOLAR_SHOCK"]
    nudged = dataclasses.replace(
        original,
        perturbations=(
            dataclasses.replace(original.perturbations[0], magnitude=0.71),
        ),
    )
    assert nudged.scenario_hash() != original.scenario_hash()


def test_rescale_onsets_shifts_every_perturbation(scenarios):
    original = scenarios["F_COMPOUND"]
    shifted = rescale_onsets(original, 2.0)
    for before, after in zip(original.perturbations, shifted.perturbations, strict=True):
        assert after.start_h == pytest.approx(before.start_h * 2.0)
    assert shifted.scenario_hash() != original.scenario_hash()


def test_scenario_round_trips_through_its_dict_form(scenarios):
    for original in scenarios.values():
        assert scenario_from_dict(original.to_dict()).scenario_hash() == original.scenario_hash()


def test_duplicate_scenario_ids_are_rejected(tmp_path):
    import json

    from fluxa.scenarios import load_scenarios

    path = tmp_path / "dup.json"
    entry = {"scenario_id": "X", "name": "x", "perturbations": []}
    path.write_text(json.dumps({"scenarios": [entry, entry]}))
    with pytest.raises(ScenarioApplicationError, match="duplicate scenario_id"):
        load_scenarios(path)


def test_empty_scenario_library_is_rejected(tmp_path):
    import json

    from fluxa.scenarios import load_scenarios

    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"scenarios": []}))
    with pytest.raises(ScenarioApplicationError, match="no scenarios defined"):
        load_scenarios(path)
