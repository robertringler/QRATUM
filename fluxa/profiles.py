"""Deterministic exogenous time series: load, solar availability, wind availability.

Every series is produced by a closed-form analytic shape plus, optionally, a
seeded stochastic component drawn from an explicitly constructed
``numpy.random.Generator(PCG64(seed))``. No global RNG state is touched, so
two processes with the same seed produce bit-identical series regardless of
import order or other code in the process.

These are *synthetic* profiles chosen to be physically plausible. They are
not measurements from any real grid; see the report's Limitations section.

Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from fluxa.model import EnergySystem, GeneratorKind
from fluxa.units import seconds_to_hours


@dataclass(frozen=True)
class TimeGrid:
    """Simulation clock.

    Attributes:
        start_iso: ISO-8601 UTC timestamp of the first timestep's start.
        timestep_s: Timestep length in seconds.
        n_steps: Number of timesteps.
    """

    start_iso: str
    timestep_s: float
    n_steps: int

    def __post_init__(self) -> None:
        if self.timestep_s <= 0.0:
            raise ValueError("timestep_s must be > 0")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be > 0")
        # Fail fast on an unparseable start time rather than deep in the loop.
        self.start_datetime()

    def start_datetime(self) -> datetime:
        dt = datetime.fromisoformat(self.start_iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @property
    def timestep_h(self) -> float:
        return seconds_to_hours(self.timestep_s)

    @property
    def horizon_h(self) -> float:
        return self.n_steps * self.timestep_h

    def timestamp(self, step: int) -> str:
        """ISO-8601 UTC timestamp at the start of ``step``.

        Derived arithmetically from ``start_iso``; never reads a wall clock.
        """
        return (
            self.start_datetime() + timedelta(seconds=self.timestep_s * step)
        ).isoformat().replace("+00:00", "Z")

    def hours_elapsed(self) -> np.ndarray:
        """(n_steps,) hours from simulation start to each timestep's start."""
        return np.arange(self.n_steps, dtype=np.float64) * self.timestep_h

    def to_dict(self) -> dict[str, object]:
        return {
            "start_iso": self.start_iso,
            "timestep_s": self.timestep_s,
            "n_steps": self.n_steps,
            "horizon_h": self.horizon_h,
        }


@dataclass(frozen=True)
class ProfileParameters:
    """Shape parameters for the synthetic profiles. All values dimensionless
    unless a unit is named."""

    # Load
    load_base_fraction: float = 0.62
    load_morning_peak_h: float = 8.5
    load_evening_peak_h: float = 19.0
    load_morning_amplitude: float = 0.16
    load_evening_amplitude: float = 0.38
    load_peak_width_h: float = 3.0
    load_weekend_factor: float = 0.88
    load_noise_sigma: float = 0.012

    # Solar
    solar_sunrise_h: float = 6.0
    solar_sunset_h: float = 19.5
    solar_clear_sky_peak: float = 0.92
    solar_cloud_sigma: float = 0.10
    solar_cloud_correlation: float = 0.90

    # Wind
    wind_mean: float = 0.42
    wind_diurnal_amplitude: float = 0.14
    wind_diurnal_phase_h: float = 2.0
    wind_synoptic_amplitude: float = 0.18
    wind_synoptic_period_h: float = 38.0
    wind_noise_sigma: float = 0.07
    wind_correlation: float = 0.93


@dataclass(frozen=True)
class ExogenousSeries:
    """Baseline exogenous drivers for one simulation.

    Attributes:
        time: The simulation clock.
        total_load_mw: (n_steps,) system-wide demand.
        bus_load_mw: (n_steps, n_buses) demand allocated by bus load_share.
        availability_mw: {gen_id: (n_steps,)} upper bound on variable-renewable
            output, i.e. the resource that is physically present before any
            dispatch decision.
        capacity_factor: {gen_id: (n_steps,)} availability_mw / p_max_mw.
        seeds: The seeds actually used, for the provenance record.
    """

    time: TimeGrid
    total_load_mw: np.ndarray
    bus_load_mw: np.ndarray
    availability_mw: dict[str, np.ndarray]
    capacity_factor: dict[str, np.ndarray]
    seeds: dict[str, int]


def _ar1(gen: np.random.Generator, n: int, rho: float, sigma: float) -> np.ndarray:
    """Zero-mean AR(1) series with correlation ``rho`` and innovation ``sigma``.

    Used to give solar and wind realistic temporal persistence instead of
    white noise, which would be smoothed away by the dispatch and understate
    ramping stress.
    """
    out = np.zeros(n, dtype=np.float64)
    innovations = gen.normal(0.0, sigma, size=n)
    for k in range(1, n):
        out[k] = rho * out[k - 1] + innovations[k]
    return out


def load_shape(hours: np.ndarray, params: ProfileParameters) -> np.ndarray:
    """Dimensionless daily demand shape in (0, 1], peaking at 1.0.

    Two Gaussian bumps (morning and evening) on a flat base, modulated by a
    weekday/weekend factor derived from the elapsed-hours index.
    """
    hour_of_day = np.mod(hours, 24.0)
    day_index = np.floor(hours / 24.0).astype(np.int64)

    def bump(centre: float, amplitude: float) -> np.ndarray:
        # Wrap the distance so a peak near midnight is continuous.
        d = np.abs(hour_of_day - centre)
        d = np.minimum(d, 24.0 - d)
        return amplitude * np.exp(-0.5 * (d / params.load_peak_width_h) ** 2)

    shape = (
        params.load_base_fraction
        + bump(params.load_morning_peak_h, params.load_morning_amplitude)
        + bump(params.load_evening_peak_h, params.load_evening_amplitude)
    )
    # Day 0 of the simulation is treated as a Monday.
    is_weekend = np.isin(np.mod(day_index, 7), (5, 6))
    shape = np.where(is_weekend, shape * params.load_weekend_factor, shape)
    return shape


def solar_clear_sky(hours: np.ndarray, params: ProfileParameters) -> np.ndarray:
    """Dimensionless clear-sky PV capacity factor in [0, solar_clear_sky_peak]."""
    hour_of_day = np.mod(hours, 24.0)
    day_length = params.solar_sunset_h - params.solar_sunrise_h
    phase = np.pi * (hour_of_day - params.solar_sunrise_h) / day_length
    cf = params.solar_clear_sky_peak * np.sin(np.clip(phase, 0.0, np.pi))
    return np.where(
        (hour_of_day >= params.solar_sunrise_h) & (hour_of_day <= params.solar_sunset_h),
        cf,
        0.0,
    )


def wind_shape(hours: np.ndarray, params: ProfileParameters) -> np.ndarray:
    """Dimensionless wind capacity factor before stochastic perturbation."""
    diurnal = params.wind_diurnal_amplitude * np.sin(
        2.0 * np.pi * (hours - params.wind_diurnal_phase_h) / 24.0
    )
    synoptic = params.wind_synoptic_amplitude * np.sin(
        2.0 * np.pi * hours / params.wind_synoptic_period_h
    )
    return params.wind_mean + diurnal + synoptic


def build_series(
    system: EnergySystem,
    time: TimeGrid,
    params: ProfileParameters | None = None,
    *,
    load_seed: int = 10_001,
    solar_seed: int = 20_002,
    wind_seed: int = 30_003,
    stochastic: bool = True,
) -> ExogenousSeries:
    """Construct the baseline exogenous drivers for ``system`` over ``time``.

    Args:
        system: The energy system (supplies peak load, bus shares, capacities).
        time: Simulation clock.
        params: Shape parameters; defaults to :class:`ProfileParameters`.
        load_seed, solar_seed, wind_seed: Independent PCG64 seeds. Each driver
            gets its own generator so that changing one driver's seed does not
            shift the others.
        stochastic: When False, all noise terms are zero and the series are
            purely analytic. Used by the unit tests that must not depend on
            RNG behaviour.

    Returns:
        An :class:`ExogenousSeries`.
    """
    params = params or ProfileParameters()
    hours = time.hours_elapsed()
    n = time.n_steps

    if stochastic:
        load_noise = _ar1(
            np.random.Generator(np.random.PCG64(load_seed)), n, 0.8, params.load_noise_sigma
        )
        cloud = _ar1(
            np.random.Generator(np.random.PCG64(solar_seed)),
            n,
            params.solar_cloud_correlation,
            params.solar_cloud_sigma,
        )
        wind_noise = _ar1(
            np.random.Generator(np.random.PCG64(wind_seed)),
            n,
            params.wind_correlation,
            params.wind_noise_sigma,
        )
    else:
        load_noise = cloud = wind_noise = np.zeros(n, dtype=np.float64)

    total_load_mw = system.peak_load_mw * np.maximum(load_shape(hours, params) + load_noise, 0.05)

    shares = np.array([b.load_share for b in system.buses], dtype=np.float64)
    bus_load_mw = total_load_mw[:, None] * shares[None, :]

    solar_cf = np.clip(solar_clear_sky(hours, params) * (1.0 + cloud), 0.0, 1.0)
    # Clouds must not create generation where there is no sun.
    solar_cf = np.where(solar_clear_sky(hours, params) > 0.0, solar_cf, 0.0)
    wind_cf = np.clip(wind_shape(hours, params) + wind_noise, 0.0, 1.0)

    availability: dict[str, np.ndarray] = {}
    capacity_factor: dict[str, np.ndarray] = {}
    for gen in system.variable_generators:
        cf = solar_cf if gen.kind is GeneratorKind.SOLAR_PV else wind_cf
        capacity_factor[gen.gen_id] = cf.copy()
        availability[gen.gen_id] = cf * gen.p_max_mw

    return ExogenousSeries(
        time=time,
        total_load_mw=total_load_mw,
        bus_load_mw=bus_load_mw,
        availability_mw=availability,
        capacity_factor=capacity_factor,
        seeds={"load": load_seed, "solar": solar_seed, "wind": wind_seed},
    )
