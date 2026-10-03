"""FLUXA machine-readable timestep state.

One :class:`FluxaState` is produced per committed timestep. It is the unit of
record: it is what goes into the CSV/Parquet output, what is hashed into the
provenance chain, and what a checkpoint restores.

Units are carried in field names. Power is MW, energy MWh, cost USD, angles
degrees (converted from the solver's radians at the boundary), frequency Hz,
state of charge a dimensionless fraction.

Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from fluxa.dispatch import DispatchSolution
from fluxa.model import EnergySystem
from fluxa.units import POWER_TOLERANCE_MW


class OperationalState(str, Enum):
    """Power-system operating state, after the classical Dy Liacco partition.

    NORMAL      -- all loads served, all constraints satisfied, reserve met.
    ALERT       -- all loads served and no constraint violated, but the
                   security margin is eroded (reserve shortfall, or a line
                   at/above the warning utilisation threshold).
    EMERGENCY   -- a physical constraint is violated: load is being shed or a
                   line is loaded beyond its rating.
    RESTORATIVE -- no violation in this timestep, but the system has not yet
                   held a clean state long enough to be declared stabilised.
    """

    NORMAL = "NORMAL"
    ALERT = "ALERT"
    EMERGENCY = "EMERGENCY"
    RESTORATIVE = "RESTORATIVE"


#: Line utilisation at or above which a constraint is reported as approached.
CONSTRAINT_WARNING_UTILISATION: float = 0.95

#: Absolute frequency deviation (Hz) beyond which the linear quasi-static
#: frequency proxy is outside its validity range. A real interconnected system
#: subjected to a deviation this large would have tripped under-frequency load
#: shedding and very likely cascaded; the proxy's linear extrapolation is then
#: an indicator of *severity ordering*, not a prediction of system frequency.
#: States beyond this bound are flagged, never silently reported as if valid.
FREQUENCY_PROXY_VALID_DEVIATION_HZ: float = 2.0


@dataclass
class FluxaState:
    """Complete observable state of the energy system at one timestep."""

    # ---- identity -------------------------------------------------------
    step: int
    timestamp: str
    scenario_id: str
    run_id: str

    # ---- demand ---------------------------------------------------------
    total_load_mw: float
    served_load_mw: float
    unserved_mw: float
    unserved_energy_mwh: float
    bus_load_mw: dict[str, float]

    # ---- generation -----------------------------------------------------
    generation_by_source_mw: dict[str, float]
    generation_by_unit_mw: dict[str, float]
    total_generation_mw: float
    renewable_generation_mw: float
    renewable_available_mw: float
    curtailment_mw: float
    co2_tonnes: float

    # ---- storage --------------------------------------------------------
    battery_soc: dict[str, float]
    battery_charge_mw: float
    battery_discharge_mw: float
    battery_stored_mwh: float

    # ---- interchange ----------------------------------------------------
    net_import_mw: float
    net_export_mw: float

    # ---- network (DC-equivalent state) ---------------------------------
    bus_angle_deg: dict[str, float]
    bus_injection_mw: dict[str, float]
    line_flow_mw: dict[str, float]
    line_utilisation: dict[str, float]
    max_line_utilisation: float
    overload_mw: dict[str, float]
    max_overload_mw: float

    # ---- security -------------------------------------------------------
    reserve_requirement_mw: float
    reserve_available_mw: float
    reserve_margin: float
    reserve_shortfall_mw: float
    frequency_proxy_hz: float
    frequency_proxy_in_range: bool

    # ---- economics (this timestep only) --------------------------------
    operating_cost_usd: float
    violation_cost_usd: float
    curtailment_cost_usd: float

    # ---- classification -------------------------------------------------
    operational_state: OperationalState
    safety_level: str
    violations: list[str]
    active_perturbations: dict[str, float]

    # ---- numerical hygiene ---------------------------------------------
    balance_residual_mw: float
    solver_status: int
    solver_iterations: int

    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable mapping, with deterministic key order."""
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, OperationalState):
                out[key] = value.value
            elif isinstance(value, np.generic):
                out[key] = value.item()
            else:
                out[key] = value
        return out

    #: Fields excluded from :meth:`state_hash`. ``run_id`` is a caller-supplied
    #: label: two executions of the same experiment under different labels are
    #: the same physical state, and letting the label change the digest would
    #: make determinism unverifiable across runs. It is still carried on the
    #: state and written to the CSV/Parquet output for traceability.
    HASH_EXCLUDED_FIELDS = ("run_id",)

    def hashable_dict(self) -> dict[str, Any]:
        """The content-addressed view of this state: physics, not labels."""
        return {
            key: value
            for key, value in self.to_dict().items()
            if key not in self.HASH_EXCLUDED_FIELDS
        }

    def state_hash(self) -> str:
        """SHA-256 over the full-precision physical state.

        ``repr`` of a Python float round-trips exactly and ``json.dumps`` uses
        it, so this digest is sensitive to the last bit of every value. Two
        runs agreeing on this hash agree bit-for-bit on the physics.
        """
        blob = json.dumps(self.hashable_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    #: Dict fields whose key set varies between timesteps. Expanding these
    #: into columns would give different timesteps different schemas, so they
    #: are serialised as a single canonical-JSON column instead. Keeping a
    #: uniform schema is what makes the CSV and Parquet exports loadable as
    #: one table.
    VARIABLE_KEY_DICT_FIELDS = ("active_perturbations",)

    def flat_record(self) -> dict[str, Any]:
        """Flattened row for tabular (CSV/Parquet) export.

        Fixed-key dicts (per-bus, per-line, per-unit quantities) become
        ``field.key`` columns. Variable-key dicts become one JSON column plus
        scalar summaries, so every timestep yields the same columns.
        """
        row: dict[str, Any] = {}
        for key, value in self.to_dict().items():
            if key == "extra":
                continue
            if key in self.VARIABLE_KEY_DICT_FIELDS:
                mapping = value or {}
                row[key] = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
                row[f"{key}.count"] = len(mapping)
                row[f"{key}.max_intensity"] = (
                    max(mapping.values()) if mapping else 0.0
                )
            elif isinstance(value, dict):
                for sub, sub_value in value.items():
                    row[f"{key}.{sub}"] = sub_value
            elif isinstance(value, list):
                row[key] = ";".join(str(v) for v in value)
            else:
                row[key] = value
        return row


def classify_operational_state(
    *,
    unserved_mw: float,
    max_overload_mw: float,
    reserve_shortfall_mw: float,
    max_line_utilisation: float,
    in_restoration: bool,
) -> tuple[OperationalState, list[str]]:
    """Classify the operating state and enumerate the violations observed.

    Returns:
        ``(state, violations)`` where ``violations`` holds one string per
        distinct violated or approached constraint, in a stable order.
    """
    violations: list[str] = []
    if unserved_mw > POWER_TOLERANCE_MW:
        violations.append(f"UNSERVED_LOAD:{unserved_mw:.6f}MW")
    if max_overload_mw > POWER_TOLERANCE_MW:
        violations.append(f"LINE_OVERLOAD:{max_overload_mw:.6f}MW")
    if reserve_shortfall_mw > POWER_TOLERANCE_MW:
        violations.append(f"RESERVE_SHORTFALL:{reserve_shortfall_mw:.6f}MW")
    if max_line_utilisation >= CONSTRAINT_WARNING_UTILISATION:
        violations.append(f"LINE_NEAR_LIMIT:{max_line_utilisation:.4f}")

    hard = unserved_mw > POWER_TOLERANCE_MW or max_overload_mw > POWER_TOLERANCE_MW
    if hard:
        return OperationalState.EMERGENCY, violations
    soft = (
        reserve_shortfall_mw > POWER_TOLERANCE_MW
        or max_line_utilisation >= CONSTRAINT_WARNING_UTILISATION
    )
    if soft:
        return OperationalState.ALERT, violations
    if in_restoration:
        return OperationalState.RESTORATIVE, violations
    return OperationalState.NORMAL, violations


def classify_safety_level(
    *,
    operational_state: OperationalState,
    violation_count: int,
    unserved_mw: float,
    outage_active: bool,
    major_dispatch_change: bool,
    storage_dispatch_mw: float,
) -> str:
    """Map a timestep to a QRADLE safety level.

    The mapping is deliberately conservative about the top of the scale:

    * ``ROUTINE``    -- ordinary state advancement.
    * ``ELEVATED``   -- a major dispatch change, material storage action, or
                        an eroded security margin (ALERT).
    * ``SENSITIVE``  -- a forced generator outage is in effect; this is the
                        operation that is authorization-gated.
    * ``CRITICAL``   -- a cascading condition: load shedding, or two or more
                        simultaneous constraint violations.
    * ``EXISTENTIAL`` -- never emitted by FLUXA. It is reserved in QRADLE for
                        architecture-level conditions; an ordinary grid
                        contingency is not one. See the report.
    """
    if unserved_mw > POWER_TOLERANCE_MW or (
        operational_state is OperationalState.EMERGENCY and violation_count >= 2
    ):
        return "CRITICAL"
    if outage_active:
        return "SENSITIVE"
    if (
        operational_state in (OperationalState.ALERT, OperationalState.EMERGENCY)
        or major_dispatch_change
        or storage_dispatch_mw > POWER_TOLERANCE_MW
    ):
        return "ELEVATED"
    return "ROUTINE"


def frequency_proxy_hz(system: EnergySystem, unarrested_mw: float) -> float:
    """Steady-state frequency implied by power primary response did not arrest.

    See :class:`fluxa.model.FrequencyModel` for the formulation and its
    limitations. ``unarrested_mw`` is unserved load plus reserve shortfall.
    """
    fm = system.frequency
    return fm.nominal_hz - unarrested_mw / fm.response_characteristic_mw_per_hz


def frequency_proxy_in_range(system: EnergySystem, frequency_hz: float) -> bool:
    """Whether the frequency proxy is inside its stated validity range."""
    return (
        abs(frequency_hz - system.frequency.nominal_hz)
        <= FREQUENCY_PROXY_VALID_DEVIATION_HZ
    )


def build_state(
    *,
    system: EnergySystem,
    solution: DispatchSolution,
    step: int,
    timestamp: str,
    scenario_id: str,
    run_id: str,
    bus_load_mw: np.ndarray,
    line_capacity_mw: np.ndarray,
    variable_availability_mw: np.ndarray,
    active_perturbations: dict[str, float],
    outage_active: bool,
    major_dispatch_change: bool,
    in_restoration: bool,
    timestep_h: float,
) -> FluxaState:
    """Assemble a :class:`FluxaState` from a committed dispatch solution."""
    bus_ids = [b.bus_id for b in system.buses]
    line_ids = [ln.line_id for ln in system.lines]

    unserved_mw = float(solution.unserved_mw.sum())
    max_overload = float(solution.overload_mw.max()) if solution.overload_mw.size else 0.0
    utilisation = np.abs(solution.line_flow_mw) / line_capacity_mw
    max_util = float(utilisation.max()) if utilisation.size else 0.0

    op_state, violations = classify_operational_state(
        unserved_mw=unserved_mw,
        max_overload_mw=max_overload,
        reserve_shortfall_mw=solution.reserve_short_mw,
        max_line_utilisation=max_util,
        in_restoration=in_restoration,
    )
    storage_dispatch = float(solution.p_charge_mw.sum() + solution.p_discharge_mw.sum())
    safety = classify_safety_level(
        operational_state=op_state,
        violation_count=len([v for v in violations if not v.startswith("LINE_NEAR_LIMIT")]),
        unserved_mw=unserved_mw,
        outage_active=outage_active,
        major_dispatch_change=major_dispatch_change,
        storage_dispatch_mw=storage_dispatch,
    )

    gen_by_unit: dict[str, float] = {}
    gen_by_source: dict[str, float] = {}
    co2 = 0.0
    for i, gen in enumerate(system.dispatchable_generators):
        p = float(solution.p_disp_mw[i])
        gen_by_unit[gen.gen_id] = p
        gen_by_source[gen.kind.value] = gen_by_source.get(gen.kind.value, 0.0) + p
        co2 += p * timestep_h * gen.co2_tonnes_per_mwh
    renewable = 0.0
    for i, gen in enumerate(system.variable_generators):
        p = float(solution.p_var_mw[i])
        gen_by_unit[gen.gen_id] = p
        gen_by_source[gen.kind.value] = gen_by_source.get(gen.kind.value, 0.0) + p
        renewable += p

    available_renewable = float(variable_availability_mw.sum())
    curtailment = max(available_renewable - renewable, 0.0)

    total_gen = float(solution.p_disp_mw.sum() + solution.p_var_mw.sum())
    total_load = float(bus_load_mw.sum())
    net_import = float(solution.p_import_mw.sum())
    net_export = float(solution.p_export_mw.sum())

    soc = {
        bat.battery_id: float(solution.soc_end[i]) for i, bat in enumerate(system.batteries)
    }
    stored = float(
        sum(
            solution.soc_end[i] * bat.energy_capacity_mwh
            for i, bat in enumerate(system.batteries)
        )
    )

    unarrested = unserved_mw + solution.reserve_short_mw
    freq_hz = frequency_proxy_hz(system, unarrested)
    requirement = solution.reserve_requirement_mw
    margin = (
        (solution.reserve_available_mw - requirement) / requirement
        if requirement > POWER_TOLERANCE_MW
        else float("inf")
    )

    return FluxaState(
        step=step,
        timestamp=timestamp,
        scenario_id=scenario_id,
        run_id=run_id,
        total_load_mw=total_load,
        served_load_mw=total_load - unserved_mw,
        unserved_mw=unserved_mw,
        unserved_energy_mwh=unserved_mw * timestep_h,
        bus_load_mw={bid: float(bus_load_mw[i]) for i, bid in enumerate(bus_ids)},
        generation_by_source_mw=gen_by_source,
        generation_by_unit_mw=gen_by_unit,
        total_generation_mw=total_gen,
        renewable_generation_mw=renewable,
        renewable_available_mw=available_renewable,
        curtailment_mw=curtailment,
        co2_tonnes=co2,
        battery_soc=soc,
        battery_charge_mw=float(solution.p_charge_mw.sum()),
        battery_discharge_mw=float(solution.p_discharge_mw.sum()),
        battery_stored_mwh=stored,
        net_import_mw=net_import,
        net_export_mw=net_export,
        bus_angle_deg={
            bid: float(np.degrees(solution.bus_angle_rad[i])) for i, bid in enumerate(bus_ids)
        },
        bus_injection_mw={
            bid: float(solution.bus_injection_mw[i]) for i, bid in enumerate(bus_ids)
        },
        line_flow_mw={lid: float(solution.line_flow_mw[i]) for i, lid in enumerate(line_ids)},
        line_utilisation={lid: float(utilisation[i]) for i, lid in enumerate(line_ids)},
        max_line_utilisation=max_util,
        overload_mw={lid: float(solution.overload_mw[i]) for i, lid in enumerate(line_ids)},
        max_overload_mw=max_overload,
        reserve_requirement_mw=requirement,
        reserve_available_mw=solution.reserve_available_mw,
        reserve_margin=margin,
        reserve_shortfall_mw=solution.reserve_short_mw,
        frequency_proxy_hz=freq_hz,
        frequency_proxy_in_range=frequency_proxy_in_range(system, freq_hz),
        operating_cost_usd=solution.step_operating_cost_usd,
        violation_cost_usd=solution.step_violation_cost_usd,
        curtailment_cost_usd=solution.step_curtailment_cost_usd,
        operational_state=op_state,
        safety_level=safety,
        violations=violations,
        active_perturbations=active_perturbations,
        balance_residual_mw=solution.balance_residual_mw,
        solver_status=solution.solver_status,
        solver_iterations=solution.solver_iterations,
    )
