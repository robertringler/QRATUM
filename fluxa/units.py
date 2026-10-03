"""Explicit unit system for the FLUXA energy-system vertical.

Every quantity carried through FLUXA has exactly one canonical unit. Mixing
MW/kW or MWh/kWh silently is the single most common defect class in
power-system code, so all conversions are funnelled through this module and
every public dataclass field name carries its unit as a suffix.

Canonical units
---------------
power          : MW   (megawatt)
energy         : MWh  (megawatt-hour)
time (step)    : s    (seconds), derived hours exposed as ``*_h``
cost / price   : USD, USD/MWh
voltage angle  : rad internally, deg in reported state
frequency      : Hz
reactance      : per-unit on ``S_BASE_MVA``
state of charge: dimensionless fraction in [0, 1]

Version: 1.0.0
"""

from __future__ import annotations

from typing import Final

#: Three-phase apparent-power base for the per-unit network model (MVA).
S_BASE_MVA: Final[float] = 100.0

#: Nominal system frequency (Hz).
F_NOMINAL_HZ: Final[float] = 50.0

#: Seconds per hour, used for every power <-> energy conversion.
SECONDS_PER_HOUR: Final[float] = 3600.0

#: Tolerance (MW) below which a power residual is treated as numerically zero.
#: Chosen as 1e-6 MW == 1 W, far below any physically meaningful quantity and
#: comfortably above LP solver primal feasibility (~1e-9 on this problem size).
POWER_TOLERANCE_MW: Final[float] = 1e-6

#: Tolerance (MWh) for energy-conservation assertions.
ENERGY_TOLERANCE_MWH: Final[float] = 1e-6

#: Tolerance (fraction) for state-of-charge bound assertions.
SOC_TOLERANCE: Final[float] = 1e-9


def seconds_to_hours(seconds: float) -> float:
    """Convert a duration in seconds to hours."""
    return seconds / SECONDS_PER_HOUR


def mw_to_mwh(power_mw: float, duration_s: float) -> float:
    """Integrate a constant power (MW) over a duration (s) to energy (MWh)."""
    return power_mw * seconds_to_hours(duration_s)


def mwh_to_mw(energy_mwh: float, duration_s: float) -> float:
    """Average an energy (MWh) over a duration (s) to power (MW)."""
    hours = seconds_to_hours(duration_s)
    if hours <= 0.0:
        raise ValueError(f"duration must be positive, got {duration_s} s")
    return energy_mwh / hours


def kw_to_mw(power_kw: float) -> float:
    """Convert kW to MW."""
    return power_kw / 1000.0


def mw_to_kw(power_mw: float) -> float:
    """Convert MW to kW."""
    return power_mw * 1000.0


def kwh_to_mwh(energy_kwh: float) -> float:
    """Convert kWh to MWh."""
    return energy_kwh / 1000.0


def mwh_to_kwh(energy_mwh: float) -> float:
    """Convert MWh to kWh."""
    return energy_mwh * 1000.0


def mw_to_pu(power_mw: float) -> float:
    """Convert MW to per-unit on ``S_BASE_MVA``."""
    return power_mw / S_BASE_MVA


def pu_to_mw(power_pu: float) -> float:
    """Convert per-unit power on ``S_BASE_MVA`` to MW."""
    return power_pu * S_BASE_MVA
