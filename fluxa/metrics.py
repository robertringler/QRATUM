"""Reliability, economic, renewable-integration and resilience metrics.

Every metric here is computed from the recorded :class:`fluxa.state.FluxaState`
trajectory. Nothing is estimated, extrapolated, or taken from an external
source.

A deliberate omission: FLUXA does **not** report SAIDI, SAIFI, CAIDI, ENS,
LOLP, LOLE or EUE. Those indices have precise regulatory definitions tied to
customer counts, interruption events and multi-year observation windows that
this model does not represent. Reporting a number under one of those names
from a single synthetic 24-hour trajectory would be a fabrication. The
quantities actually measured -- unserved energy, violation duration, reserve
shortfall -- are reported under their own plain names.

Version: 1.0.0
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from fluxa.model import EnergySystem
from fluxa.state import FluxaState, OperationalState
from fluxa.units import POWER_TOLERANCE_MW


def _is_hard_violation(state: FluxaState) -> bool:
    """True if the timestep shed load or overloaded a line."""
    return (
        state.unserved_mw > POWER_TOLERANCE_MW
        or state.max_overload_mw > POWER_TOLERANCE_MW
    )


def _is_degraded(state: FluxaState) -> bool:
    """True if the timestep lost security margin (ALERT) or physics (EMERGENCY).

    RESTORATIVE is deliberately *not* degraded: it is the label a recovering
    system carries while it accumulates the clean timesteps needed to be
    declared stabilised. Treating it as degraded would make stabilisation
    unreachable, since a system cannot leave RESTORATIVE until it has already
    been stable.
    """
    return state.operational_state in (OperationalState.ALERT, OperationalState.EMERGENCY)


@dataclass(frozen=True)
class ReliabilityMetrics:
    """Service continuity and constraint compliance over a trajectory."""

    energy_demanded_mwh: float
    energy_served_mwh: float
    unserved_energy_mwh: float
    unserved_energy_fraction: float
    peak_unserved_mw: float
    n_steps: int
    n_steps_with_hard_violation: int
    n_steps_with_any_violation: int
    n_violation_records: int
    violation_duration_min: float
    unserved_duration_min: float
    overload_duration_min: float
    max_line_overload_mw: float
    max_line_utilisation: float
    reserve_shortfall_mwh: float
    n_steps_reserve_short: int
    min_reserve_margin: float
    min_frequency_proxy_hz: float
    max_frequency_deviation_hz: float
    n_steps_frequency_alarm: int
    n_steps_frequency_proxy_out_of_range: int
    steps_by_operational_state: dict[str, int]


@dataclass(frozen=True)
class EconomicMetrics:
    """Operating cost decomposition over a trajectory (USD).

    ``total_operating_cost_usd`` is the economically meaningful production
    cost: fuel, variable O&M, storage cycling and net interchange. Soft
    constraint penalties (overload, reserve shortfall) are *not* market
    prices and are reported separately as ``penalty_cost_usd``; unserved
    energy is valued at the configured VOLL and reported on its own.
    """

    generation_cost_usd: float
    generation_cost_by_source_usd: dict[str, float]
    storage_cost_usd: float
    import_cost_usd: float
    export_revenue_usd: float
    total_operating_cost_usd: float
    curtailment_cost_usd: float
    unserved_energy_cost_usd: float
    penalty_cost_usd: float
    total_cost_with_externalities_usd: float
    average_cost_usd_per_mwh_served: float
    co2_tonnes: float


@dataclass(frozen=True)
class RenewableMetrics:
    """Variable-renewable integration and storage contribution.

    Storage note: the ratio ``discharge_mwh / charge_mwh`` is *not* a
    round-trip efficiency and is deliberately not reported. Over a finite
    horizon the battery also consumes (or accumulates) its initial stock, so
    that ratio can exceed 1 without any energy having been created. What is
    reported instead is the exact storage energy balance::

        E_final - E_initial = eta_ch * charge_mwh - discharge_mwh / eta_dis

    whose residual (``storage_energy_balance_residual_mwh``) is the
    conservation check, plus the nameplate efficiency for reference and the
    equivalent full-cycle count for degradation context.
    """

    renewable_generation_mwh: float
    renewable_available_mwh: float
    renewable_penetration: float
    renewable_utilisation: float
    curtailment_mwh: float
    curtailment_fraction: float
    storage_charge_mwh: float
    storage_discharge_mwh: float
    storage_throughput_mwh: float
    storage_contribution: float
    storage_net_withdrawal_mwh: float
    storage_energy_balance_residual_mwh: float
    storage_nameplate_round_trip: float
    storage_cycle_equivalents: float
    storage_soc_min: float
    storage_soc_max: float
    storage_soc_initial: float
    storage_soc_final: float


@dataclass(frozen=True)
class ResilienceMetrics:
    """Disturbance response.

    Definitions (all measured, none assumed):

    * ``disturbance_onset_h`` -- earliest perturbation onset declared by the
      scenario. ``None`` for an unperturbed scenario.
    * ``detection_step`` -- first timestep at which the engine emitted
      DISTURBANCE_DETECTED, i.e. the first step at which a perturbed driver
      departed from its baseline by more than the configured detection
      threshold. This is *observability latency at a stated threshold*, not a
      claim about SCADA or telemetry delay.
    * ``first_violation_step`` -- first timestep with load shedding or a line
      overload at or after onset.
    * ``episode_had_degradation`` -- whether any timestep in the episode left
      the NORMAL state. When False the disturbance was absorbed without any
      measurable loss of security and the stabilisation and recovery fields
      are ``None`` by construction: there was nothing to recover from.
    * ``stabilization_step`` -- first timestep at or after which the required
      number of consecutive NORMAL timesteps is achieved, searching from the
      first degraded timestep. ``None`` if the trajectory never stabilises, or
      if it never degraded.
    * ``recovery_time_min`` -- from the first degraded timestep to
      stabilisation.
    * ``energy_deficit_mwh`` -- unserved energy accumulated from onset to
      stabilisation (or to end of run).
    * ``max_system_stress`` -- maximum line utilisation over the episode.
    """

    disturbance_onset_h: float | None
    disturbance_onset_step: int | None
    detection_step: int | None
    detection_delay_min: float | None
    episode_had_degradation: bool
    first_degraded_step: int | None
    first_violation_step: int | None
    time_to_first_violation_min: float | None
    stabilization_step: int | None
    time_to_stabilization_min: float | None
    recovery_time_min: float | None
    stabilised: bool
    energy_deficit_mwh: float
    max_system_stress: float
    min_reserve_margin: float
    max_frequency_deviation_hz: float
    peak_unserved_mw: float
    episode_steps: int


@dataclass(frozen=True)
class RunMetrics:
    """All metric families for one trajectory, plus the aggregate hash input."""

    scenario_id: str
    run_id: str
    reliability: ReliabilityMetrics
    economics: EconomicMetrics
    renewables: RenewableMetrics
    resilience: ResilienceMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "run_id": self.run_id,
            "reliability": asdict(self.reliability),
            "economics": asdict(self.economics),
            "renewables": asdict(self.renewables),
            "resilience": asdict(self.resilience),
        }

    def output_payload(self) -> dict[str, Any]:
        """Metric payload hashed into the provenance bundle's ``output_hash``.

        ``run_id`` is excluded so that two runs of the same experiment under
        different ids have the same ``output_hash``.
        """
        payload = self.to_dict()
        payload.pop("run_id", None)
        return payload


def compute_reliability(
    system: EnergySystem, states: Sequence[FluxaState], timestep_h: float
) -> ReliabilityMetrics:
    """Reliability metrics over ``states``."""
    dt_min = timestep_h * 60.0
    demanded = sum(s.total_load_mw for s in states) * timestep_h
    unserved = sum(s.unserved_energy_mwh for s in states)
    served = demanded - unserved

    hard = [s for s in states if _is_hard_violation(s)]
    any_v = [s for s in states if s.violations]
    n_unserved = sum(1 for s in states if s.unserved_mw > POWER_TOLERANCE_MW)
    n_overload = sum(1 for s in states if s.max_overload_mw > POWER_TOLERANCE_MW)
    n_res_short = sum(1 for s in states if s.reserve_shortfall_mw > POWER_TOLERANCE_MW)

    finite_margins = [s.reserve_margin for s in states if math.isfinite(s.reserve_margin)]
    by_state: dict[str, int] = {}
    for s in states:
        key = s.operational_state.value
        by_state[key] = by_state.get(key, 0) + 1

    freqs = [s.frequency_proxy_hz for s in states]
    nominal = system.frequency.nominal_hz
    deviations = [abs(f - nominal) for f in freqs]

    return ReliabilityMetrics(
        energy_demanded_mwh=demanded,
        energy_served_mwh=served,
        unserved_energy_mwh=unserved,
        unserved_energy_fraction=(unserved / demanded if demanded > 0 else 0.0),
        peak_unserved_mw=max((s.unserved_mw for s in states), default=0.0),
        n_steps=len(states),
        n_steps_with_hard_violation=len(hard),
        n_steps_with_any_violation=len(any_v),
        n_violation_records=sum(len(s.violations) for s in states),
        violation_duration_min=len(any_v) * dt_min,
        unserved_duration_min=n_unserved * dt_min,
        overload_duration_min=n_overload * dt_min,
        max_line_overload_mw=max((s.max_overload_mw for s in states), default=0.0),
        max_line_utilisation=max((s.max_line_utilisation for s in states), default=0.0),
        reserve_shortfall_mwh=sum(s.reserve_shortfall_mw for s in states) * timestep_h,
        n_steps_reserve_short=n_res_short,
        min_reserve_margin=min(finite_margins) if finite_margins else float("inf"),
        min_frequency_proxy_hz=min(freqs) if freqs else nominal,
        max_frequency_deviation_hz=max(deviations) if deviations else 0.0,
        n_steps_frequency_alarm=sum(
            1 for d in deviations if d >= system.frequency.alarm_deviation_hz
        ),
        n_steps_frequency_proxy_out_of_range=sum(
            1 for s in states if not s.frequency_proxy_in_range
        ),
        steps_by_operational_state=dict(sorted(by_state.items())),
    )


def compute_economics(
    system: EnergySystem, states: Sequence[FluxaState], timestep_h: float
) -> EconomicMetrics:
    """Cost decomposition over ``states``.

    Generation, storage and interchange costs are recomputed here from the
    recorded per-unit dispatch and the model's prices, rather than taken from
    the LP objective. That makes the economic report independent of the
    objective's penalty terms and of the curtailment offset constant.
    """
    econ = system.economics
    cost_by_source: dict[str, float] = {}
    gen_cost = 0.0
    co2 = 0.0
    for state in states:
        for gen in system.generators:
            p = state.generation_by_unit_mw.get(gen.gen_id, 0.0)
            energy = p * timestep_h
            unit_cost = energy * gen.marginal_cost_usd_per_mwh
            gen_cost += unit_cost
            cost_by_source[gen.kind.value] = cost_by_source.get(gen.kind.value, 0.0) + unit_cost
            co2 += energy * gen.co2_tonnes_per_mwh

    storage_cost = 0.0
    for state in states:
        throughput = (state.battery_charge_mw + state.battery_discharge_mw) * timestep_h
        # All batteries share the same throughput price in this model; if that
        # ever differs, split by battery_id here.
        price = (
            system.batteries[0].cycle_cost_usd_per_mwh if system.batteries else 0.0
        )
        storage_cost += throughput * price

    import_cost = export_rev = 0.0
    if system.interconnections:
        link = system.interconnections[0]
        for state in states:
            import_cost += state.net_import_mw * timestep_h * link.import_price_usd_per_mwh
            export_rev += state.net_export_mw * timestep_h * link.export_price_usd_per_mwh

    curtail_mwh = sum(state.curtailment_mw for state in states) * timestep_h
    curtail_cost = curtail_mwh * econ.curtailment_cost_usd_per_mwh
    unserved_mwh = sum(state.unserved_energy_mwh for state in states)
    unserved_cost = unserved_mwh * econ.value_of_lost_load_usd_per_mwh
    penalty = sum(state.violation_cost_usd for state in states) - unserved_cost

    co2_cost = co2 * econ.co2_price_usd_per_tonne
    operating = gen_cost + storage_cost + import_cost - export_rev + co2_cost
    demanded = sum(state.total_load_mw for state in states) * timestep_h
    served = demanded - unserved_mwh

    return EconomicMetrics(
        generation_cost_usd=gen_cost,
        generation_cost_by_source_usd=dict(sorted(cost_by_source.items())),
        storage_cost_usd=storage_cost,
        import_cost_usd=import_cost,
        export_revenue_usd=export_rev,
        total_operating_cost_usd=operating,
        curtailment_cost_usd=curtail_cost,
        unserved_energy_cost_usd=unserved_cost,
        penalty_cost_usd=max(penalty, 0.0),
        total_cost_with_externalities_usd=operating + curtail_cost + unserved_cost + max(penalty, 0.0),
        average_cost_usd_per_mwh_served=(operating / served if served > 0 else float("nan")),
        co2_tonnes=co2,
    )


def compute_renewables(
    system: EnergySystem, states: Sequence[FluxaState], timestep_h: float
) -> RenewableMetrics:
    """Renewable integration and storage contribution over ``states``."""
    gen_mwh = sum(s.renewable_generation_mw for s in states) * timestep_h
    avail_mwh = sum(s.renewable_available_mw for s in states) * timestep_h
    curtail_mwh = sum(s.curtailment_mw for s in states) * timestep_h
    charge_mwh = sum(s.battery_charge_mw for s in states) * timestep_h
    discharge_mwh = sum(s.battery_discharge_mw for s in states) * timestep_h
    demanded = sum(s.total_load_mw for s in states) * timestep_h
    unserved_mwh = sum(s.unserved_energy_mwh for s in states)
    served = demanded - unserved_mwh

    socs = [v for s in states for v in s.battery_soc.values()]

    # Exact storage energy balance across the horizon.
    stored_initial = sum(b.soc_initial * b.energy_capacity_mwh for b in system.batteries)
    stored_final = sum(
        states[-1].battery_soc.get(b.battery_id, b.soc_initial) * b.energy_capacity_mwh
        for b in system.batteries
    )
    total_capacity = sum(b.energy_capacity_mwh for b in system.batteries)
    eta_ch = system.batteries[0].charge_efficiency if system.batteries else 1.0
    eta_dis = system.batteries[0].discharge_efficiency if system.batteries else 1.0
    nameplate_rt = (
        system.batteries[0].round_trip_efficiency if system.batteries else float("nan")
    )
    residual = (stored_final - stored_initial) - (
        eta_ch * charge_mwh - discharge_mwh / eta_dis
    )

    return RenewableMetrics(
        renewable_generation_mwh=gen_mwh,
        renewable_available_mwh=avail_mwh,
        renewable_penetration=(gen_mwh / served if served > 0 else 0.0),
        renewable_utilisation=(gen_mwh / avail_mwh if avail_mwh > 0 else float("nan")),
        curtailment_mwh=curtail_mwh,
        curtailment_fraction=(curtail_mwh / avail_mwh if avail_mwh > 0 else 0.0),
        storage_charge_mwh=charge_mwh,
        storage_discharge_mwh=discharge_mwh,
        storage_throughput_mwh=charge_mwh + discharge_mwh,
        storage_contribution=(discharge_mwh / served if served > 0 else 0.0),
        storage_net_withdrawal_mwh=stored_initial - stored_final,
        storage_energy_balance_residual_mwh=residual,
        storage_nameplate_round_trip=nameplate_rt,
        storage_cycle_equivalents=(
            (charge_mwh + discharge_mwh) / (2.0 * total_capacity)
            if total_capacity > 0
            else 0.0
        ),
        storage_soc_min=min(socs) if socs else float("nan"),
        storage_soc_max=max(socs) if socs else float("nan"),
        storage_soc_initial=(
            min(b.soc_initial for b in system.batteries) if system.batteries else float("nan")
        ),
        storage_soc_final=(
            min(states[-1].battery_soc.values())
            if states and states[-1].battery_soc
            else float("nan")
        ),
    )


def compute_resilience(
    system: EnergySystem,
    states: Sequence[FluxaState],
    timestep_h: float,
    *,
    onset_h: float | None,
    detection_step: int | None,
    stabilization_steps: int,
) -> ResilienceMetrics:
    """Disturbance-response metrics over ``states``.

    Args:
        onset_h: Declared perturbation onset, hours from start.
        detection_step: Step at which DISTURBANCE_DETECTED fired, if it did.
        stabilization_steps: Consecutive NORMAL steps required to declare the
            system stabilised.
    """
    dt_min = timestep_h * 60.0
    onset_step = int(round(onset_h / timestep_h)) if onset_h is not None else None
    episode_start = onset_step if onset_step is not None else 0
    episode = list(states[episode_start:])

    first_violation = next(
        (s.step for s in states if s.step >= episode_start and _is_hard_violation(s)), None
    )
    first_degraded = next(
        (s.step for s in states if s.step >= episode_start and _is_degraded(s)), None
    )
    had_degradation = first_degraded is not None

    # Stabilisation is only defined once the system has actually left NORMAL.
    # Searching for it in an undisturbed trajectory would report the onset
    # step itself as "stabilisation" and yield a negative recovery time.
    stabilization_step: int | None = None
    if had_degradation:
        run_length = 0
        candidate: int | None = None
        for s in states:
            if s.step < first_degraded:
                continue
            if not _is_degraded(s):
                if run_length == 0:
                    candidate = s.step
                run_length += 1
                if run_length >= stabilization_steps:
                    stabilization_step = candidate
                    break
            else:
                run_length = 0
                candidate = None

    deficit_end = stabilization_step if stabilization_step is not None else len(states)
    deficit = sum(
        s.unserved_energy_mwh for s in states if episode_start <= s.step < max(deficit_end, episode_start)
    )

    recovery = (
        (stabilization_step - first_degraded) * dt_min
        if (stabilization_step is not None and first_degraded is not None)
        else None
    )
    finite_margins = [s.reserve_margin for s in episode if math.isfinite(s.reserve_margin)]
    nominal = system.frequency.nominal_hz

    return ResilienceMetrics(
        disturbance_onset_h=onset_h,
        disturbance_onset_step=onset_step,
        detection_step=detection_step,
        detection_delay_min=(
            (detection_step - onset_step) * dt_min
            if (detection_step is not None and onset_step is not None)
            else None
        ),
        episode_had_degradation=had_degradation,
        first_degraded_step=first_degraded,
        first_violation_step=first_violation,
        time_to_first_violation_min=(
            (first_violation - onset_step) * dt_min
            if (first_violation is not None and onset_step is not None)
            else None
        ),
        stabilization_step=stabilization_step,
        time_to_stabilization_min=(
            (stabilization_step - onset_step) * dt_min
            if (stabilization_step is not None and onset_step is not None)
            else None
        ),
        recovery_time_min=recovery,
        stabilised=(not had_degradation) or stabilization_step is not None,
        energy_deficit_mwh=deficit,
        max_system_stress=max((s.max_line_utilisation for s in episode), default=0.0),
        min_reserve_margin=min(finite_margins) if finite_margins else float("inf"),
        max_frequency_deviation_hz=max(
            (abs(s.frequency_proxy_hz - nominal) for s in episode), default=0.0
        ),
        peak_unserved_mw=max((s.unserved_mw for s in episode), default=0.0),
        episode_steps=len(episode),
    )


def compute_all(
    system: EnergySystem,
    states: Sequence[FluxaState],
    timestep_h: float,
    *,
    scenario_id: str,
    run_id: str,
    onset_h: float | None,
    detection_step: int | None,
    stabilization_steps: int,
) -> RunMetrics:
    """Compute every metric family for one trajectory."""
    if not states:
        raise ValueError("cannot compute metrics for an empty trajectory")
    return RunMetrics(
        scenario_id=scenario_id,
        run_id=run_id,
        reliability=compute_reliability(system, states, timestep_h),
        economics=compute_economics(system, states, timestep_h),
        renewables=compute_renewables(system, states, timestep_h),
        resilience=compute_resilience(
            system,
            states,
            timestep_h,
            onset_h=onset_h,
            detection_step=detection_step,
            stabilization_steps=stabilization_steps,
        ),
    )
