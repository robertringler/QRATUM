"""Unit-conversion correctness. A silent MW/MWh confusion is the most common
defect class in power-system code, so the conversions are tested directly."""

from __future__ import annotations

import math

import pytest

from fluxa import units


def test_mw_to_mwh_over_five_minutes():
    assert units.mw_to_mwh(60.0, 300.0) == pytest.approx(5.0)


def test_mw_to_mwh_over_one_hour_is_identity():
    assert units.mw_to_mwh(123.456, 3600.0) == pytest.approx(123.456)


def test_mwh_to_mw_inverts_mw_to_mwh():
    for power, duration in ((42.0, 300.0), (0.5, 900.0), (310.75, 3600.0)):
        assert units.mwh_to_mw(units.mw_to_mwh(power, duration), duration) == pytest.approx(power)


def test_mwh_to_mw_rejects_zero_duration():
    with pytest.raises(ValueError, match="duration must be positive"):
        units.mwh_to_mw(10.0, 0.0)


def test_kilo_mega_round_trip():
    assert units.kw_to_mw(units.mw_to_kw(7.25)) == pytest.approx(7.25)
    assert units.kwh_to_mwh(units.mwh_to_kwh(7.25)) == pytest.approx(7.25)


def test_per_unit_round_trip_uses_declared_base():
    assert units.mw_to_pu(units.S_BASE_MVA) == pytest.approx(1.0)
    assert units.pu_to_mw(units.mw_to_pu(250.0)) == pytest.approx(250.0)


def test_seconds_to_hours():
    assert units.seconds_to_hours(900.0) == pytest.approx(0.25)


def test_tolerances_are_physically_negligible_but_above_solver_noise():
    # 1e-6 MW is one watt: far below anything meaningful, far above LP noise.
    assert 1e-9 < units.POWER_TOLERANCE_MW < 1e-3
    assert math.isfinite(units.ENERGY_TOLERANCE_MWH)
