"""Monte Carlo campaign over randomised FLUXA conditions.

Design
------
Each sample is a *fully specified deterministic simulation*. Randomness enters
only through the sample's seeds and its drawn perturbation parameters, both of
which are derived from a single ``campaign_seed`` via
``numpy.random.SeedSequence``. Replaying the campaign with the same
``campaign_seed`` and sample count reproduces every sample exactly, and sample
``k`` is independent of how many samples were drawn before it -- so a 10-sample
and a 100-sample campaign agree on their first 10 samples.

Randomised quantities
---------------------
* load level          -- multiplicative factor, lognormal around 1.0
* solar availability  -- multiplicative factor on the PV series
* wind availability   -- multiplicative factor on the wind series
* generator availability -- the CCGT is forced out with probability
  ``outage_probability``; when it is, the outage duration is drawn uniformly
* disturbance timing  -- the onset hour is drawn uniformly over the day
* driver seeds        -- each sample re-draws the load/solar/wind AR(1) seeds

Reported statistics are **simulation-derived**: they describe the behaviour of
this model under this sampling distribution. They are not empirical grid
statistics and must not be read as frequencies of real-world events. The
sampling distribution itself is an assumption, stated in
:class:`CampaignConfig`.

Percentile reporting
--------------------
A percentile is only reported when the sample size supports it. With ``n``
samples the engine reports a percentile ``p`` only if
``n * (1 - p/100) >= 1``, i.e. at least one sample lies in the upper tail:
P99 needs n >= 100, P95 needs n >= 20, P90 needs n >= 10. Unsupported
percentiles are reported as ``None`` rather than as an interpolated value that
no observation backs.

Version: 1.0.0
"""

from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Sequence

import numpy as np

from fluxa.config import DEFAULT_SYSTEM_CONFIG, LoadedSystem, load_system
from fluxa.engine import RunConfig, SimulationEngine
from fluxa.profiles import ProfileParameters
from fluxa.scenarios import Perturbation, PerturbationKind, Scenario

LOGGER = logging.getLogger("fluxa.montecarlo")

#: Percentiles the campaign attempts to report, with the minimum sample size
#: at which each is backed by at least one observation in the tail.
PERCENTILE_MIN_SAMPLES: dict[int, int] = {50: 2, 90: 10, 95: 20, 99: 100}


@dataclass(frozen=True)
class CampaignConfig:
    """Sampling distribution and execution settings for a campaign.

    Every distribution parameter here is a modelling assumption, not an
    observation. They are chosen to span conditions a small island system
    plausibly sees over a year, wide enough that the tail statistics are not
    all driven by the same mechanism.
    """

    campaign_seed: int = 777_001
    n_samples: int = 100
    n_steps: int = 288
    timestep_s: float = 300.0
    horizon_steps: int = 12

    load_factor_log_sigma: float = 0.10
    load_factor_min: float = 0.70
    load_factor_max: float = 1.35

    solar_factor_min: float = 0.25
    solar_factor_max: float = 1.05
    wind_factor_min: float = 0.15
    wind_factor_max: float = 1.15

    outage_probability: float = 0.25
    outage_duration_min_minutes: float = 60.0
    outage_duration_max_minutes: float = 240.0

    disturbance_onset_min_h: float = 4.0
    disturbance_onset_max_h: float = 21.0
    vre_shock_probability: float = 0.50
    vre_shock_depth_min: float = 0.30
    vre_shock_depth_max: float = 0.90

    #: Monte Carlo samples are authorized by construction: the campaign is a
    #: planning study whose authorization is granted once, for the campaign,
    #: and recorded on every sample's SCENARIO_INITIALIZED event.
    authorizer_id: str = "campaign_authority"
    emit_per_step_events: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CampaignSample:
    """One fully specified sample, derived deterministically from the seed."""

    index: int
    load_factor: float
    solar_factor: float
    wind_factor: float
    outage: bool
    outage_duration_min: float
    vre_shock: bool
    vre_shock_depth: float
    vre_shock_target: str
    onset_h: float
    load_seed: int
    solar_seed: int
    wind_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def scenario(self) -> Scenario:
        """Build the scenario this sample represents."""
        perturbations: list[Perturbation] = []
        if self.vre_shock:
            perturbations.append(
                Perturbation(
                    perturbation_id=f"MC_VRE_SHOCK_{self.index}",
                    kind=PerturbationKind.RENEWABLE_DERATE,
                    targets=(self.vre_shock_target,),
                    start_h=self.onset_h,
                    ramp_min=15.0,
                    hold_min=120.0,
                    recovery_min=45.0,
                    magnitude=self.vre_shock_depth,
                )
            )
        if self.outage:
            perturbations.append(
                Perturbation(
                    perturbation_id=f"MC_OUTAGE_{self.index}",
                    kind=PerturbationKind.GENERATOR_OUTAGE,
                    targets=("GAS_CCGT_1",),
                    start_h=self.onset_h + 0.25,
                    ramp_min=0.0,
                    hold_min=self.outage_duration_min,
                    recovery_min=15.0,
                )
            )
        return Scenario(
            scenario_id=f"MC_{self.index:04d}",
            name=f"Monte Carlo sample {self.index}",
            description=(
                f"load x{self.load_factor:.4f}, solar x{self.solar_factor:.4f}, "
                f"wind x{self.wind_factor:.4f}, outage={self.outage}, "
                f"vre_shock={self.vre_shock}, onset={self.onset_h:.4f} h"
            ),
            perturbations=tuple(perturbations),
            tags=("monte-carlo",),
        )


def draw_samples(config: CampaignConfig) -> list[CampaignSample]:
    """Derive every sample from ``campaign_seed``.

    Each sample gets its own child ``SeedSequence``, so sample ``k`` is
    identical regardless of ``n_samples``. That property is what lets the
    scaling benchmark run 1, 10 and 100 samples and still compare like with
    like.
    """
    root = np.random.SeedSequence(config.campaign_seed)
    children = root.spawn(config.n_samples)
    samples: list[CampaignSample] = []
    for index, child in enumerate(children):
        gen = np.random.Generator(np.random.PCG64(child))
        load_factor = float(
            np.clip(
                math.exp(gen.normal(0.0, config.load_factor_log_sigma)),
                config.load_factor_min,
                config.load_factor_max,
            )
        )
        solar_factor = float(gen.uniform(config.solar_factor_min, config.solar_factor_max))
        wind_factor = float(gen.uniform(config.wind_factor_min, config.wind_factor_max))
        outage = bool(gen.random() < config.outage_probability)
        outage_duration = float(
            gen.uniform(config.outage_duration_min_minutes, config.outage_duration_max_minutes)
        )
        vre_shock = bool(gen.random() < config.vre_shock_probability)
        depth = float(gen.uniform(config.vre_shock_depth_min, config.vre_shock_depth_max))
        target = "SOLAR_PV_1" if gen.random() < 0.5 else "WIND_1"
        onset = float(gen.uniform(config.disturbance_onset_min_h, config.disturbance_onset_max_h))
        seeds = gen.integers(1, 2**31 - 1, size=3)
        samples.append(
            CampaignSample(
                index=index,
                load_factor=load_factor,
                solar_factor=solar_factor,
                wind_factor=wind_factor,
                outage=outage,
                outage_duration_min=outage_duration,
                vre_shock=vre_shock,
                vre_shock_depth=depth,
                vre_shock_target=target,
                onset_h=onset,
                load_seed=int(seeds[0]),
                solar_seed=int(seeds[1]),
                wind_seed=int(seeds[2]),
            )
        )
    return samples


def _scaled_system(loaded: LoadedSystem, sample: CampaignSample) -> LoadedSystem:
    """Apply the sample's load factor by scaling the system's peak load.

    Scaling ``peak_load_mw`` rather than the realised series keeps the system
    object self-consistent, so the model hash changes with the sample and the
    provenance record cannot conflate two different load levels.
    """
    scaled = replace(loaded.system, peak_load_mw=loaded.system.peak_load_mw * sample.load_factor)
    return replace(loaded, system=scaled)


def _scaled_profile_params(sample: CampaignSample) -> ProfileParameters:
    """Apply the sample's resource factors to the profile shape parameters."""
    base = ProfileParameters()
    return replace(
        base,
        solar_clear_sky_peak=min(base.solar_clear_sky_peak * sample.solar_factor, 1.0),
        wind_mean=base.wind_mean * sample.wind_factor,
        wind_diurnal_amplitude=base.wind_diurnal_amplitude * sample.wind_factor,
        wind_synoptic_amplitude=base.wind_synoptic_amplitude * sample.wind_factor,
    )


def run_sample(
    sample: CampaignSample,
    config: CampaignConfig,
    system_config_path: str = str(DEFAULT_SYSTEM_CONFIG),
) -> dict[str, Any]:
    """Run one sample and return its summary row.

    Loads the system from disk rather than taking it as an argument so the
    function is picklable and usable as a ``ProcessPoolExecutor`` task.
    """
    loaded = _scaled_system(load_system(system_config_path), sample)
    run_config = RunConfig(
        run_id=f"mc{config.campaign_seed}-{sample.index:04d}",
        timestep_s=config.timestep_s,
        n_steps=config.n_steps,
        horizon_steps=config.horizon_steps,
        load_seed=sample.load_seed,
        solar_seed=sample.solar_seed,
        wind_seed=sample.wind_seed,
        checkpoint_every=0,
        authorized=True,
        authorizer_id=config.authorizer_id,
        emit_per_step_events=config.emit_per_step_events,
    )
    engine = SimulationEngine(loaded, run_config, _scaled_profile_params(sample))
    result = engine.run(sample.scenario())
    row = result.summary()
    row.update({f"sample.{k}": v for k, v in sample.to_dict().items()})
    row["sample_index"] = sample.index
    return row


def _run_sample_task(args: tuple[CampaignSample, CampaignConfig, str]) -> dict[str, Any]:
    """Module-level trampoline so samples can be dispatched to a process pool."""
    return run_sample(*args)


#: Metrics the campaign builds distributions over.
DISTRIBUTION_METRICS: tuple[str, ...] = (
    "unserved_energy_mwh",
    "unserved_energy_fraction",
    "curtailment_mwh",
    "total_operating_cost_usd",
    "min_reserve_margin",
    "max_line_utilisation",
    "recovery_time_min",
    "renewable_penetration",
    "energy_deficit_mwh",
    "co2_tonnes",
)


def percentiles(values: Sequence[float], n_samples: int) -> dict[str, float | None]:
    """Report only the percentiles the sample size supports.

    Uses linear interpolation between order statistics (numpy's default). A
    percentile is withheld when fewer than ``PERCENTILE_MIN_SAMPLES[p]``
    observations exist, because beyond that point the "percentile" is an
    extrapolation of the single most extreme sample.

    The gate is applied to the count of **finite observations**, not to the
    campaign's sample count. Some metrics are undefined for some samples --
    ``recovery_time_min`` does not exist for a sample that never degraded --
    so gating on the campaign size would report a P99 backed by far fewer
    than 100 observations. ``n_samples`` is accepted for call compatibility
    and is recorded as ``n_campaign_samples``.
    """
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    out: dict[str, float | None] = {
        "n": len(finite),
        "n_campaign_samples": n_samples,
        "n_excluded_non_finite": len(values) - len(finite),
    }
    if not finite:
        for p in PERCENTILE_MIN_SAMPLES:
            out[f"P{p}"] = None
        out.update({"mean": None, "min": None, "max": None, "std": None})
        return out
    arr = np.asarray(finite, dtype=np.float64)
    out["mean"] = float(arr.mean())
    out["std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    out["min"] = float(arr.min())
    out["max"] = float(arr.max())
    for p, minimum in PERCENTILE_MIN_SAMPLES.items():
        out[f"P{p}"] = float(np.percentile(arr, p)) if len(finite) >= minimum else None
    return out


@dataclass
class CampaignResult:
    """Outcome of a Monte Carlo campaign."""

    config: CampaignConfig
    samples: list[CampaignSample]
    rows: list[dict[str, Any]]
    distributions: dict[str, dict[str, float | None]]
    wall_time_s: float
    parallel: bool
    n_workers: int
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "n_samples_requested": self.config.n_samples,
            "n_samples_completed": len(self.rows),
            "wall_time_s": self.wall_time_s,
            "parallel": self.parallel,
            "n_workers": self.n_workers,
            "percentile_minimum_samples": PERCENTILE_MIN_SAMPLES,
            "distributions": self.distributions,
            "failures": self.failures,
        }


def run_campaign(
    config: CampaignConfig,
    *,
    parallel: bool = False,
    n_workers: int | None = None,
    system_config_path: str = str(DEFAULT_SYSTEM_CONFIG),
) -> CampaignResult:
    """Run a campaign and summarise the resulting distributions.

    Args:
        config: Sampling distribution and run settings.
        parallel: Dispatch samples to a process pool. Results are reordered by
            sample index afterwards, so parallel and serial runs produce
            identical rows -- which the benchmark asserts rather than assumes.
        n_workers: Pool size; defaults to ``os.cpu_count()``.
        system_config_path: System document every worker loads.

    A sample that raises is recorded in ``failures`` and excluded from the
    distributions; the campaign does not abort, but the failure is never
    silent.
    """
    samples = draw_samples(config)
    workers = n_workers or (os.cpu_count() or 1)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    start = time.perf_counter()
    if parallel and config.n_samples > 1:
        tasks = [(s, config, system_config_path) for s in samples]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for sample, outcome in zip(samples, pool.map(_run_sample_task, tasks), strict=True):
                rows.append(outcome)
    else:
        workers = 1
        for sample in samples:
            try:
                rows.append(run_sample(sample, config, system_config_path))
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                LOGGER.error("sample %d failed: %s", sample.index, exc)
                failures.append(
                    {
                        "sample_index": sample.index,
                        "error": f"{type(exc).__name__}: {exc}",
                        "sample": sample.to_dict(),
                    }
                )
    wall = time.perf_counter() - start

    rows.sort(key=lambda r: r["sample_index"])
    distributions = {
        metric: percentiles([r.get(metric) for r in rows], len(rows))
        for metric in DISTRIBUTION_METRICS
    }
    return CampaignResult(
        config=config,
        samples=samples,
        rows=rows,
        distributions=distributions,
        wall_time_s=wall,
        parallel=parallel,
        n_workers=workers,
        failures=failures,
    )
