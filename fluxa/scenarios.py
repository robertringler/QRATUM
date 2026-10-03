"""FLUXA scenario engine: controlled, reproducible perturbations.

A :class:`Scenario` is a declarative list of :class:`Perturbation` objects.
Applying a scenario is a pure function of (baseline series, scenario), so the
same scenario applied to the same baseline always yields identical drivers and
an identical ``scenario_hash``.

Each perturbation carries the QRADLE safety level required to apply it. The
engine refuses to apply a SENSITIVE-or-higher perturbation without
authorization, which is how the authorization gate is exercised (see
:mod:`fluxa.engine`).

Perturbation time shape
-----------------------
Every perturbation uses a trapezoid in time::

    intensity
      1.0            ____________
                    /            \\
      0.0  ________/              \\________
            start  +ramp   +hold   +recovery

``intensity`` scales the perturbation's magnitude, so a 70% solar derate over
15 minutes means intensity rises linearly from 0 to 1 across 15 minutes while
availability falls from 100% to 30%.

Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any

import numpy as np

from fluxa.config import canonical_hash
from fluxa.model import EnergySystem
from fluxa.profiles import ExogenousSeries


class PerturbationKind(str, Enum):
    """What a perturbation acts on."""

    #: Scale down the availability of named variable-renewable generators.
    RENEWABLE_DERATE = "renewable_derate"
    #: Force named dispatchable generators offline (availability -> 0).
    GENERATOR_OUTAGE = "generator_outage"
    #: Scale system demand up (or down, with negative magnitude).
    LOAD_SCALE = "load_scale"
    #: Reduce the thermal rating of named lines without changing topology.
    LINE_DERATE = "line_derate"


#: QRADLE safety level required to apply each perturbation kind.
#: Removing generation capacity from a modelled system is the operation whose
#: misuse could mislead a real operator, so it is the one gated at SENSITIVE.
PERTURBATION_SAFETY_LEVEL: dict[PerturbationKind, str] = {
    PerturbationKind.RENEWABLE_DERATE: "ELEVATED",
    PerturbationKind.LOAD_SCALE: "ELEVATED",
    PerturbationKind.LINE_DERATE: "ELEVATED",
    PerturbationKind.GENERATOR_OUTAGE: "SENSITIVE",
}

#: Ordering used to take the maximum over a scenario's perturbations.
SAFETY_LEVEL_ORDER: tuple[str, ...] = (
    "ROUTINE",
    "ELEVATED",
    "SENSITIVE",
    "CRITICAL",
    "EXISTENTIAL",
)


def max_safety_level(levels: list[str]) -> str:
    """Return the highest level in ``levels`` by :data:`SAFETY_LEVEL_ORDER`."""
    if not levels:
        return "ROUTINE"
    unknown = [lv for lv in levels if lv not in SAFETY_LEVEL_ORDER]
    if unknown:
        raise ValueError(f"unknown safety level(s): {unknown}")
    return max(levels, key=SAFETY_LEVEL_ORDER.index)


@dataclass(frozen=True)
class Perturbation:
    """A single controlled disturbance.

    Attributes:
        perturbation_id: Stable identifier, used in event provenance.
        kind: What the perturbation acts on.
        targets: Generator or line ids. Ignored (and must be empty) for
            LOAD_SCALE, which is system-wide.
        start_h: Onset, in hours from simulation start.
        ramp_min: Minutes over which intensity rises 0 -> 1.
        hold_min: Minutes at full intensity.
        recovery_min: Minutes over which intensity falls 1 -> 0. Zero means
            the perturbation persists to the end of the simulation.
        magnitude: For RENEWABLE_DERATE / LINE_DERATE, the fractional
            reduction in [0, 1]. For LOAD_SCALE, the fractional increase
            (0.25 == +25%). Ignored for GENERATOR_OUTAGE (always total).
    """

    perturbation_id: str
    kind: PerturbationKind
    start_h: float
    ramp_min: float = 0.0
    hold_min: float = 0.0
    recovery_min: float = 0.0
    magnitude: float = 1.0
    targets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.start_h < 0.0:
            raise ValueError(f"{self.perturbation_id}: start_h must be >= 0")
        for name in ("ramp_min", "hold_min", "recovery_min"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{self.perturbation_id}: {name} must be >= 0")
        if self.kind in (PerturbationKind.RENEWABLE_DERATE, PerturbationKind.LINE_DERATE):
            if not 0.0 <= self.magnitude <= 1.0:
                raise ValueError(
                    f"{self.perturbation_id}: derate magnitude must be in [0,1], "
                    f"got {self.magnitude}"
                )
            if not self.targets:
                raise ValueError(f"{self.perturbation_id}: derate requires explicit targets")
        if self.kind is PerturbationKind.GENERATOR_OUTAGE and not self.targets:
            raise ValueError(f"{self.perturbation_id}: outage requires explicit targets")
        if self.kind is PerturbationKind.LOAD_SCALE and self.targets:
            raise ValueError(f"{self.perturbation_id}: LOAD_SCALE is system-wide; targets must be empty")

    @property
    def safety_level(self) -> str:
        return PERTURBATION_SAFETY_LEVEL[self.kind]

    def intensity(self, hours: np.ndarray) -> np.ndarray:
        """Trapezoidal intensity in [0, 1] evaluated at ``hours``."""
        ramp_h = self.ramp_min / 60.0
        hold_h = self.hold_min / 60.0
        rec_h = self.recovery_min / 60.0
        t = hours - self.start_h
        out = np.zeros_like(hours, dtype=np.float64)

        if ramp_h > 0.0:
            rising = (t >= 0.0) & (t < ramp_h)
            out[rising] = t[rising] / ramp_h
        full_start = ramp_h
        full_end = ramp_h + hold_h
        out[(t >= full_start) & (t <= full_end)] = 1.0
        if rec_h > 0.0:
            falling = (t > full_end) & (t < full_end + rec_h)
            out[falling] = 1.0 - (t[falling] - full_end) / rec_h
        else:
            out[t > full_end] = 1.0
        return np.clip(out, 0.0, 1.0)

    def onset_h(self) -> float:
        """Hour at which this perturbation first has non-zero intensity."""
        return self.start_h

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["targets"] = list(self.targets)
        d["safety_level"] = self.safety_level
        return d


@dataclass(frozen=True)
class Scenario:
    """A named experiment: a baseline plus zero or more perturbations."""

    scenario_id: str
    name: str
    description: str
    perturbations: tuple[Perturbation, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def required_safety_level(self) -> str:
        """Highest safety level among this scenario's perturbations."""
        return max_safety_level([p.safety_level for p in self.perturbations])

    @property
    def first_onset_h(self) -> float | None:
        """Earliest perturbation onset, or None for an unperturbed scenario."""
        if not self.perturbations:
            return None
        return min(p.onset_h() for p in self.perturbations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "required_safety_level": self.required_safety_level,
            "perturbations": [p.to_dict() for p in self.perturbations],
        }

    def scenario_hash(self) -> str:
        """SHA-256 over the canonical scenario definition."""
        return canonical_hash(self.to_dict())


@dataclass(frozen=True)
class PerturbedSeries:
    """Exogenous drivers after a scenario has been applied.

    Attributes:
        base: The unperturbed series the scenario was applied to.
        scenario: The scenario applied.
        total_load_mw: (n_steps,) perturbed demand.
        bus_load_mw: (n_steps, n_buses) perturbed demand by bus.
        availability_mw: {gen_id: (n_steps,)} perturbed variable-renewable
            availability.
        dispatchable_availability: {gen_id: (n_steps,)} multiplier in [0, 1]
            on a dispatchable unit's capacity. 0.0 means forced offline.
        line_capacity_mw: (n_steps, n_lines) perturbed thermal ratings.
        perturbation_intensity: {perturbation_id: (n_steps,)} applied intensity,
            recorded so that each timestep's event can state exactly how much
            of each disturbance was active.
    """

    base: ExogenousSeries
    scenario: Scenario
    total_load_mw: np.ndarray
    bus_load_mw: np.ndarray
    availability_mw: dict[str, np.ndarray]
    dispatchable_availability: dict[str, np.ndarray]
    line_capacity_mw: np.ndarray
    perturbation_intensity: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def time(self):  # noqa: ANN201 - delegates to the base series' TimeGrid
        return self.base.time


class ScenarioApplicationError(RuntimeError):
    """Raised when a scenario cannot be applied as declared."""


def apply_scenario(
    system: EnergySystem,
    base: ExogenousSeries,
    scenario: Scenario,
) -> PerturbedSeries:
    """Apply ``scenario`` to ``base``, returning perturbed drivers.

    Pure function: ``base`` is never mutated.

    Raises:
        ScenarioApplicationError: if a perturbation names an unknown
            generator or line, or targets a generator of the wrong class.
    """
    hours = base.time.hours_elapsed()
    n_steps = base.time.n_steps

    total_load = base.total_load_mw.copy()
    availability = {k: v.copy() for k, v in base.availability_mw.items()}
    dispatchable = {g.gen_id: np.ones(n_steps, dtype=np.float64) for g in system.dispatchable_generators}
    line_index = {ln.line_id: i for i, ln in enumerate(system.lines)}
    line_cap = np.tile(
        np.array([ln.capacity_mw for ln in system.lines], dtype=np.float64), (n_steps, 1)
    )
    intensities: dict[str, np.ndarray] = {}

    variable_ids = {g.gen_id for g in system.variable_generators}
    dispatchable_ids = set(dispatchable)

    for pert in scenario.perturbations:
        inten = pert.intensity(hours)
        intensities[pert.perturbation_id] = inten

        if pert.kind is PerturbationKind.RENEWABLE_DERATE:
            for target in pert.targets:
                if target not in variable_ids:
                    raise ScenarioApplicationError(
                        f"{pert.perturbation_id}: '{target}' is not a variable-renewable "
                        f"generator in system {system.system_id}"
                    )
                availability[target] = availability[target] * (1.0 - pert.magnitude * inten)

        elif pert.kind is PerturbationKind.GENERATOR_OUTAGE:
            for target in pert.targets:
                if target in dispatchable_ids:
                    dispatchable[target] = dispatchable[target] * (1.0 - inten)
                elif target in variable_ids:
                    availability[target] = availability[target] * (1.0 - inten)
                else:
                    raise ScenarioApplicationError(
                        f"{pert.perturbation_id}: unknown generator '{target}' in "
                        f"system {system.system_id}"
                    )

        elif pert.kind is PerturbationKind.LOAD_SCALE:
            total_load = total_load * (1.0 + pert.magnitude * inten)

        elif pert.kind is PerturbationKind.LINE_DERATE:
            for target in pert.targets:
                if target not in line_index:
                    raise ScenarioApplicationError(
                        f"{pert.perturbation_id}: unknown line '{target}' in "
                        f"system {system.system_id}"
                    )
                col = line_index[target]
                line_cap[:, col] = line_cap[:, col] * (1.0 - pert.magnitude * inten)

        else:  # pragma: no cover - PerturbationKind is exhaustive above
            raise ScenarioApplicationError(f"unhandled perturbation kind {pert.kind}")

    shares = np.array([b.load_share for b in system.buses], dtype=np.float64)
    bus_load = total_load[:, None] * shares[None, :]

    # A fully derated line would make the LP's soft overload term the only
    # feasible path and destroy the utilisation metric, so hold a floor.
    line_cap = np.maximum(line_cap, 1e-6)

    return PerturbedSeries(
        base=base,
        scenario=scenario,
        total_load_mw=total_load,
        bus_load_mw=bus_load,
        availability_mw=availability,
        dispatchable_availability=dispatchable,
        line_capacity_mw=line_cap,
        perturbation_intensity=intensities,
    )


def scenario_from_dict(doc: dict[str, Any]) -> Scenario:
    """Deserialise a scenario from its JSON document form."""
    try:
        perts = tuple(
            Perturbation(
                perturbation_id=p["perturbation_id"],
                kind=PerturbationKind(p["kind"]),
                start_h=float(p["start_h"]),
                ramp_min=float(p.get("ramp_min", 0.0)),
                hold_min=float(p.get("hold_min", 0.0)),
                recovery_min=float(p.get("recovery_min", 0.0)),
                magnitude=float(p.get("magnitude", 1.0)),
                targets=tuple(p.get("targets", ())),
            )
            for p in doc.get("perturbations", [])
        )
        return Scenario(
            scenario_id=doc["scenario_id"],
            name=doc["name"],
            description=doc.get("description", ""),
            perturbations=perts,
            tags=tuple(doc.get("tags", ())),
        )
    except KeyError as exc:
        raise ScenarioApplicationError(f"scenario document missing key {exc}") from exc


def load_scenarios(path) -> dict[str, Scenario]:
    """Load a scenario library keyed by ``scenario_id``."""
    from fluxa.config import load_json

    doc = load_json(path)
    scenarios: dict[str, Scenario] = {}
    for entry in doc.get("scenarios", []):
        sc = scenario_from_dict(entry)
        if sc.scenario_id in scenarios:
            raise ScenarioApplicationError(f"duplicate scenario_id '{sc.scenario_id}'")
        scenarios[sc.scenario_id] = sc
    if not scenarios:
        raise ScenarioApplicationError(f"{path}: no scenarios defined")
    return scenarios


def rescale_onsets(scenario: Scenario, factor: float) -> Scenario:
    """Return ``scenario`` with every onset multiplied by ``factor``.

    Used by the extended multi-day runs to place a disturbance on a later day
    without duplicating the scenario definition.
    """
    return replace(
        scenario,
        perturbations=tuple(replace(p, start_h=p.start_h * factor) for p in scenario.perturbations),
    )
