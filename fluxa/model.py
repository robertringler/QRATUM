"""FLUXA energy-system model: typed, frozen, hashable description of a grid.

The model is pure data. It contains no simulation logic and no mutable state,
which is what makes it safe to hash into the QRADLE provenance chain: the
``model_hash`` of an :class:`EnergySystem` is a function of the physical
description alone.

All units follow :mod:`fluxa.units`.

Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from fluxa.units import F_NOMINAL_HZ


class GeneratorKind(str, Enum):
    """Physical class of a generating unit.

    DISPATCHABLE units are commanded by the dispatch engine within ramp and
    capacity limits. VARIABLE units are weather-driven: their upper bound at
    each timestep is an exogenous availability series and the dispatch engine
    may only curtail them downwards.
    """

    GAS_CCGT = "gas_ccgt"
    GAS_PEAKER = "gas_peaker"
    SOLAR_PV = "solar_pv"
    WIND = "wind"

    @property
    def is_dispatchable(self) -> bool:
        return self in (GeneratorKind.GAS_CCGT, GeneratorKind.GAS_PEAKER)

    @property
    def is_renewable(self) -> bool:
        return self in (GeneratorKind.SOLAR_PV, GeneratorKind.WIND)


@dataclass(frozen=True)
class Bus:
    """A network node.

    Attributes:
        bus_id: Stable identifier used in all cross-references.
        name: Human-readable label.
        is_slack: Angle reference for the DC power-flow solution. Exactly one
            bus in a connected system must be the slack.
        load_share: Fraction of the system-wide load allocated to this bus.
            Shares across all buses must sum to 1.0.
    """

    bus_id: str
    name: str
    is_slack: bool = False
    load_share: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.load_share <= 1.0:
            raise ValueError(f"bus {self.bus_id}: load_share must be in [0,1], got {self.load_share}")


@dataclass(frozen=True)
class Line:
    """A transmission corridor modelled as a lossless series reactance.

    Attributes:
        line_id: Stable identifier.
        from_bus: Origin bus id (positive flow direction).
        to_bus: Destination bus id.
        reactance_pu: Series reactance in per-unit on ``S_BASE_MVA``.
        capacity_mw: Thermal/stability limit on |flow|, in MW.
    """

    line_id: str
    from_bus: str
    to_bus: str
    reactance_pu: float
    capacity_mw: float

    def __post_init__(self) -> None:
        if self.reactance_pu <= 0.0:
            raise ValueError(f"line {self.line_id}: reactance_pu must be > 0")
        if self.capacity_mw <= 0.0:
            raise ValueError(f"line {self.line_id}: capacity_mw must be > 0")
        if self.from_bus == self.to_bus:
            raise ValueError(f"line {self.line_id}: self-loop is not a valid branch")


@dataclass(frozen=True)
class Generator:
    """A generating unit.

    Attributes:
        gen_id: Stable identifier.
        bus_id: Bus the unit injects into.
        kind: Physical class (see :class:`GeneratorKind`).
        p_max_mw: Nameplate capacity (MW).
        p_min_mw: Minimum stable generation when online (MW). For units
            modelled without a commitment binary this acts as a must-run
            floor whenever availability is 1.0.
        marginal_cost_usd_per_mwh: Fuel + variable O&M (USD/MWh).
        ramp_mw_per_min: Symmetric ramp capability (MW/min).
        co2_tonnes_per_mwh: Direct combustion emission factor (t CO2/MWh).
    """

    gen_id: str
    bus_id: str
    kind: GeneratorKind
    p_max_mw: float
    p_min_mw: float
    marginal_cost_usd_per_mwh: float
    ramp_mw_per_min: float
    co2_tonnes_per_mwh: float = 0.0

    def __post_init__(self) -> None:
        if self.p_max_mw <= 0.0:
            raise ValueError(f"generator {self.gen_id}: p_max_mw must be > 0")
        if not 0.0 <= self.p_min_mw <= self.p_max_mw:
            raise ValueError(f"generator {self.gen_id}: require 0 <= p_min_mw <= p_max_mw")
        if self.ramp_mw_per_min <= 0.0:
            raise ValueError(f"generator {self.gen_id}: ramp_mw_per_min must be > 0")
        if self.kind.is_renewable and self.p_min_mw != 0.0:
            raise ValueError(f"generator {self.gen_id}: variable units must have p_min_mw == 0")


@dataclass(frozen=True)
class Battery:
    """A battery energy-storage system (BESS).

    Charge and discharge efficiencies are applied one-way; the round-trip
    efficiency is their product and is exposed as
    :attr:`round_trip_efficiency`.

    Attributes:
        battery_id: Stable identifier.
        bus_id: Bus the BESS is connected to.
        energy_capacity_mwh: Usable nameplate energy (MWh).
        p_charge_max_mw: Maximum charge power drawn from the grid (MW).
        p_discharge_max_mw: Maximum discharge power delivered to grid (MW).
        charge_efficiency: One-way charge efficiency in (0, 1].
        discharge_efficiency: One-way discharge efficiency in (0, 1].
        soc_min: Lower state-of-charge bound (fraction).
        soc_max: Upper state-of-charge bound (fraction).
        soc_initial: State of charge at simulation start (fraction).
        cycle_cost_usd_per_mwh: Degradation/O&M charge per MWh of throughput.
    """

    battery_id: str
    bus_id: str
    energy_capacity_mwh: float
    p_charge_max_mw: float
    p_discharge_max_mw: float
    charge_efficiency: float
    discharge_efficiency: float
    soc_min: float
    soc_max: float
    soc_initial: float
    cycle_cost_usd_per_mwh: float = 0.0

    def __post_init__(self) -> None:
        if self.energy_capacity_mwh <= 0.0:
            raise ValueError(f"battery {self.battery_id}: energy_capacity_mwh must be > 0")
        for name in ("p_charge_max_mw", "p_discharge_max_mw"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"battery {self.battery_id}: {name} must be > 0")
        for name in ("charge_efficiency", "discharge_efficiency"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"battery {self.battery_id}: {name} must be in (0,1], got {value}")
        if not 0.0 <= self.soc_min < self.soc_max <= 1.0:
            raise ValueError(
                f"battery {self.battery_id}: require 0 <= soc_min < soc_max <= 1, "
                f"got [{self.soc_min}, {self.soc_max}]"
            )
        if not self.soc_min <= self.soc_initial <= self.soc_max:
            raise ValueError(
                f"battery {self.battery_id}: soc_initial {self.soc_initial} outside "
                f"[{self.soc_min}, {self.soc_max}]"
            )

    @property
    def round_trip_efficiency(self) -> float:
        """Round-trip efficiency (dimensionless)."""
        return self.charge_efficiency * self.discharge_efficiency


@dataclass(frozen=True)
class Interconnection:
    """A tie to an external balancing area, modelled as a price-taking link.

    Attributes:
        link_id: Stable identifier.
        bus_id: Bus the tie lands on.
        import_max_mw: Maximum power the system may import (MW).
        export_max_mw: Maximum power the system may export (MW).
        import_price_usd_per_mwh: Cost paid for imported energy.
        export_price_usd_per_mwh: Revenue received for exported energy.
    """

    link_id: str
    bus_id: str
    import_max_mw: float
    export_max_mw: float
    import_price_usd_per_mwh: float
    export_price_usd_per_mwh: float

    def __post_init__(self) -> None:
        if self.import_max_mw < 0.0 or self.export_max_mw < 0.0:
            raise ValueError(f"interconnection {self.link_id}: limits must be >= 0")


@dataclass(frozen=True)
class EconomicAssumptions:
    """Prices and penalties that define the dispatch objective.

    ``value_of_lost_load_usd_per_mwh`` is the economic penalty applied to
    unserved energy. ``overload_penalty_usd_per_mwh`` and
    ``reserve_shortfall_penalty_usd_per_mwh`` are *soft-constraint* prices:
    they make the LP always feasible so that violations are measured rather
    than rendering the problem infeasible. They are penalties, not market
    prices, and are excluded from the reported operating cost (they are
    reported separately as violation cost).
    """

    value_of_lost_load_usd_per_mwh: float = 10_000.0
    curtailment_cost_usd_per_mwh: float = 15.0
    overload_penalty_usd_per_mwh: float = 2_000.0
    reserve_shortfall_penalty_usd_per_mwh: float = 500.0
    stored_energy_value_usd_per_mwh: float = 48.0
    co2_price_usd_per_tonne: float = 0.0


@dataclass(frozen=True)
class ReservePolicy:
    """Operating-reserve requirement.

    The requirement is a linear function of load and of instantaneous
    variable-renewable output, a standard deterministic proxy for the
    combined contingency + variability reserve need.

    ``duration_h`` is the time the reserve must be sustainable for; it is what
    turns a battery's *power* headroom into an *energy*-tested contribution.
    """

    load_fraction: float = 0.08
    renewable_fraction: float = 0.15
    duration_h: float = 0.25


@dataclass(frozen=True)
class FrequencyModel:
    """Quasi-static frequency proxy.

    FLUXA does not integrate swing equations. It reports a steady-state
    frequency deviation implied by the power that primary response failed to
    arrest, using a single system frequency-response characteristic:

        delta_f = -P_unarrested / beta

    where ``P_unarrested`` is unserved load plus reserve shortfall (MW) and
    ``beta`` is the frequency-response characteristic (MW/Hz). This is a
    proxy, not a dynamic simulation; see the report's Limitations section.
    """

    nominal_hz: float = F_NOMINAL_HZ
    response_characteristic_mw_per_hz: float = 30.0
    alarm_deviation_hz: float = 0.20


@dataclass(frozen=True)
class EnergySystem:
    """Complete physical + economic description of a FLUXA energy system."""

    system_id: str
    description: str
    buses: tuple[Bus, ...]
    lines: tuple[Line, ...]
    generators: tuple[Generator, ...]
    batteries: tuple[Battery, ...]
    interconnections: tuple[Interconnection, ...]
    peak_load_mw: float
    economics: EconomicAssumptions = field(default_factory=EconomicAssumptions)
    reserve: ReservePolicy = field(default_factory=ReservePolicy)
    frequency: FrequencyModel = field(default_factory=FrequencyModel)

    # ---------------------------------------------------------------- checks
    def __post_init__(self) -> None:
        bus_ids = [b.bus_id for b in self.buses]
        if len(set(bus_ids)) != len(bus_ids):
            raise ValueError("duplicate bus_id in system")
        slacks = [b for b in self.buses if b.is_slack]
        if len(slacks) != 1:
            raise ValueError(f"exactly one slack bus required, found {len(slacks)}")
        share_total = sum(b.load_share for b in self.buses)
        if abs(share_total - 1.0) > 1e-9:
            raise ValueError(f"bus load_share must sum to 1.0, got {share_total}")
        known = set(bus_ids)
        for line in self.lines:
            for ref in (line.from_bus, line.to_bus):
                if ref not in known:
                    raise ValueError(f"line {line.line_id} references unknown bus {ref}")
        for gen in self.generators:
            if gen.bus_id not in known:
                raise ValueError(f"generator {gen.gen_id} references unknown bus {gen.bus_id}")
        for bat in self.batteries:
            if bat.bus_id not in known:
                raise ValueError(f"battery {bat.battery_id} references unknown bus {bat.bus_id}")
        for link in self.interconnections:
            if link.bus_id not in known:
                raise ValueError(f"interconnection {link.link_id} references unknown bus {link.bus_id}")
        if self.peak_load_mw <= 0.0:
            raise ValueError("peak_load_mw must be > 0")
        if not self.lines:
            raise ValueError("system must contain at least one line")

    # ---------------------------------------------------------------- views
    @property
    def bus_index(self) -> dict[str, int]:
        """Map bus_id -> column index used by the network matrices."""
        return {b.bus_id: i for i, b in enumerate(self.buses)}

    @property
    def slack_bus_id(self) -> str:
        return next(b.bus_id for b in self.buses if b.is_slack)

    @property
    def dispatchable_generators(self) -> tuple[Generator, ...]:
        return tuple(g for g in self.generators if g.kind.is_dispatchable)

    @property
    def variable_generators(self) -> tuple[Generator, ...]:
        return tuple(g for g in self.generators if g.kind.is_renewable)

    @property
    def load_buses(self) -> tuple[Bus, ...]:
        return tuple(b for b in self.buses if b.load_share > 0.0)

    # ------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        """Canonical, JSON-serialisable representation.

        Enum members are reduced to their string values so that the dict
        round-trips through ``json.dumps`` without a custom encoder, which is
        what makes :meth:`model_hash` stable across processes.
        """
        payload = asdict(self)
        for gen in payload["generators"]:
            gen["kind"] = GeneratorKind(gen["kind"]).value
        return payload

    def model_hash(self) -> str:
        """SHA-256 over the canonical model description.

        Deterministic across processes and machines: no wall-clock, no object
        identity, keys sorted.
        """
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
