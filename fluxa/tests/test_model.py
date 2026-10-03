"""Model validation: every physically impossible description must be rejected."""

from __future__ import annotations

import pytest

from fluxa.config import ConfigError, load_system
from fluxa.model import Battery, Bus, EnergySystem, Generator, GeneratorKind, Line


def test_default_system_loads_and_validates(loaded):
    assert loaded.system.system_id == "FLUXA-6BUS-ISLAND-A"
    assert len(loaded.system.buses) == 6
    assert len(loaded.system.lines) == 8
    assert len(loaded.system.generators) == 4
    assert len(loaded.system.batteries) == 1


def test_model_hash_is_stable_across_calls(system):
    assert system.model_hash() == system.model_hash()


def test_model_hash_changes_with_physics(system):
    import dataclasses

    perturbed = dataclasses.replace(system, peak_load_mw=system.peak_load_mw + 1.0)
    assert perturbed.model_hash() != system.model_hash()


def test_bus_load_shares_must_sum_to_one(system):
    import dataclasses

    broken = tuple(
        dataclasses.replace(b, load_share=b.load_share * 0.5) for b in system.buses
    )
    with pytest.raises(ValueError, match="load_share must sum"):
        dataclasses.replace(system, buses=broken)


def test_exactly_one_slack_bus_required(system):
    import dataclasses

    two_slacks = tuple(dataclasses.replace(b, is_slack=True) for b in system.buses)
    with pytest.raises(ValueError, match="exactly one slack bus"):
        dataclasses.replace(system, buses=two_slacks)


def test_line_rejects_non_positive_reactance():
    with pytest.raises(ValueError, match="reactance_pu must be > 0"):
        Line("X", "B1", "B2", 0.0, 100.0)


def test_line_rejects_self_loop():
    with pytest.raises(ValueError, match="self-loop"):
        Line("X", "B1", "B1", 0.05, 100.0)


def test_generator_rejects_pmin_above_pmax():
    with pytest.raises(ValueError, match="p_min_mw <= p_max_mw"):
        Generator("G", "B1", GeneratorKind.GAS_CCGT, 100.0, 150.0, 40.0, 4.0)


def test_variable_generator_must_have_zero_pmin():
    with pytest.raises(ValueError, match="variable units must have p_min_mw == 0"):
        Generator("S", "B2", GeneratorKind.SOLAR_PV, 100.0, 10.0, 0.0, 100.0)


def test_battery_rejects_inverted_soc_window():
    with pytest.raises(ValueError, match="soc_min < soc_max"):
        Battery("B", "B5", 100.0, 10.0, 10.0, 0.95, 0.95, 0.9, 0.1, 0.5)


def test_battery_rejects_initial_soc_outside_window():
    with pytest.raises(ValueError, match="soc_initial"):
        Battery("B", "B5", 100.0, 10.0, 10.0, 0.95, 0.95, 0.2, 0.8, 0.95)


def test_battery_rejects_efficiency_above_one():
    with pytest.raises(ValueError, match="charge_efficiency must be in"):
        Battery("B", "B5", 100.0, 10.0, 10.0, 1.4, 0.95, 0.1, 0.9, 0.5)


def test_battery_round_trip_efficiency_is_product_of_one_way():
    bat = Battery("B", "B5", 100.0, 10.0, 10.0, 0.95, 0.90, 0.1, 0.9, 0.5)
    assert bat.round_trip_efficiency == pytest.approx(0.855)


def test_generator_referencing_unknown_bus_is_rejected(system):
    import dataclasses

    bad = system.generators + (
        Generator("GHOST", "B99", GeneratorKind.GAS_PEAKER, 10.0, 0.0, 100.0, 10.0),
    )
    with pytest.raises(ValueError, match="unknown bus B99"):
        dataclasses.replace(system, generators=bad)


def test_unsupported_schema_version_is_rejected(tmp_path):
    import json

    from fluxa.config import DEFAULT_SYSTEM_CONFIG

    doc = json.loads(DEFAULT_SYSTEM_CONFIG.read_text())
    doc["schema_version"] = "9.9.9"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ConfigError, match="unsupported schema_version"):
        load_system(path)


def test_malformed_json_is_rejected(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    with pytest.raises(ConfigError, match="invalid JSON"):
        load_system(path)


def test_physically_inconsistent_document_is_rejected(tmp_path):
    import json

    from fluxa.config import DEFAULT_SYSTEM_CONFIG

    doc = json.loads(DEFAULT_SYSTEM_CONFIG.read_text())
    doc["batteries"][0]["soc_initial"] = 1.5
    path = tmp_path / "bad_soc.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(ConfigError, match="physically inconsistent"):
        load_system(path)
