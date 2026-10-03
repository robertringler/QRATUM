"""FLUXA simulation engine, executed as a QRADLE domain contract.

Integration points with QRADLE
------------------------------
1. The whole trajectory is computed inside
   :meth:`qradle.core.engine.DeterministicEngine.execute_contract`, so the run
   inherits QRADLE's invariant enforcement, its ``output_hash``, its
   checkpoint and its own event chain.
2. Before any physics runs, the scenario's required safety level is checked
   against :meth:`qradle.core.invariants.FatalInvariants.enforce_human_oversight`.
   A SENSITIVE-or-higher scenario without authorization raises
   ``InvariantViolation`` and the simulation never starts.
3. Checkpoints are created through
   :class:`qradle.core.rollback.RollbackManager`, so FLUXA rollback is QRADLE
   rollback, not a private mechanism.
4. Every significant state transition is appended to a
   :class:`fluxa.events.FluxaEventLedger` built from QRADLE's
   ``MerkleNode`` primitive.

Physical-invariant enforcement
------------------------------
A dispatch solution is only accepted after
:func:`validate_physical_state` confirms power balance, storage
conservation, capacity limits and non-negativity. A violation raises
:class:`PhysicalInvariantError` and halts the run -- the engine never
silently reports an unphysical trajectory. The rollback experiment uses
:meth:`SimulationEngine.inject_invalid_state` to deliberately create such a
state and then demonstrates detection and recovery.

Version: 1.0.0
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Sequence

import numpy as np

from fluxa.config import LoadedSystem, canonical_hash
from fluxa.dispatch import DispatchProblem, DispatchSolution, DispatchStep
from fluxa.events import FluxaEventLedger, FluxaEventType
from fluxa.metrics import RunMetrics, compute_all
from fluxa.model import EnergySystem
from fluxa.network import DCNetwork
from fluxa.profiles import ExogenousSeries, ProfileParameters, TimeGrid, build_series
from fluxa.provenance import (
    ProvenanceBundle,
    build_merkle_tree,
    environment_fingerprint,
    sha256_of,
)
from fluxa.scenarios import PerturbationKind, PerturbedSeries, Scenario, apply_scenario
from fluxa.state import FluxaState, OperationalState, build_state
from fluxa.units import ENERGY_TOLERANCE_MWH, POWER_TOLERANCE_MW, SOC_TOLERANCE
from qradle.core.engine import DeterministicEngine, ExecutionContext, ExecutionResult
from qradle.core.invariants import FatalInvariants, InvariantViolation

LOGGER = logging.getLogger("fluxa.engine")

#: Safety levels that QRADLE requires human authorization for.
AUTHORIZATION_REQUIRED_LEVELS = frozenset({"SENSITIVE", "CRITICAL", "EXISTENTIAL"})


class PhysicalInvariantError(RuntimeError):
    """Raised when a simulation state violates a physical conservation law."""


class AuthorizationDenied(RuntimeError):
    """Raised when an operation is attempted without the required authorization."""


@dataclass(frozen=True)
class RunConfig:
    """Everything that is not the system or the scenario.

    Attributes:
        run_id: Label for this execution. Excluded from the provenance
            identity hash so two runs of the same experiment can be compared.
        start_iso: Simulation start timestamp (UTC, ISO-8601).
        timestep_s: Timestep length.
        n_steps: Number of timesteps.
        horizon_steps: Receding-horizon look-ahead, in timesteps.
        load_seed, solar_seed, wind_seed: PCG64 seeds for the drivers.
        stochastic_profiles: When False the drivers are purely analytic.
        checkpoint_every: Create a QRADLE checkpoint every N steps. 0 disables.
        stabilization_steps: Consecutive NORMAL steps required to declare the
            system stabilised.
        detection_threshold_fraction: Fractional departure of a perturbed
            driver from its baseline that counts as "observable".
        major_dispatch_change_fraction: Fraction of a unit's per-step ramp
            capability above which a dispatch change is reported as major.
        authorized: Whether human authorization has been supplied.
        authorizer_id: Identity recorded in the authorization event.
        emit_per_step_events: When False only lifecycle and exception events
            are emitted. Used by the Monte Carlo campaign, where 100x2016
            STATE_ADVANCED events would dominate runtime without adding
            information.
        tolerate_invalid_states: When True, a physical-invariant failure is
            recorded as an event instead of raising. Used exclusively by the
            rollback and adversarial experiments.
    """

    run_id: str
    start_iso: str = "2026-01-05T00:00:00Z"
    timestep_s: float = 300.0
    n_steps: int = 288
    horizon_steps: int = 12
    load_seed: int = 10_001
    solar_seed: int = 20_002
    wind_seed: int = 30_003
    stochastic_profiles: bool = True
    checkpoint_every: int = 24
    stabilization_steps: int = 6
    detection_threshold_fraction: float = 0.02
    major_dispatch_change_fraction: float = 0.5
    authorized: bool = False
    authorizer_id: str = ""
    emit_per_step_events: bool = True
    tolerate_invalid_states: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def identity_payload(self) -> dict[str, Any]:
        """Run parameters that affect the result, excluding labels."""
        payload = self.to_dict()
        for label in ("run_id", "authorizer_id"):
            payload.pop(label, None)
        return payload

    @property
    def timestep_h(self) -> float:
        return self.timestep_s / 3600.0


@dataclass
class SimulationResult:
    """Everything one simulation run produced."""

    run_config: RunConfig
    scenario: Scenario
    system: EnergySystem
    states: list[FluxaState]
    ledger: FluxaEventLedger
    metrics: RunMetrics
    provenance: ProvenanceBundle
    qradle_result: ExecutionResult
    checkpoint_ids: list[str]
    wall_time_s: float
    solve_time_s: float
    detection_step: int | None
    invalid_state_reports: list[dict[str, Any]] = field(default_factory=list)

    @property
    def state_hashes(self) -> list[str]:
        return [s.state_hash() for s in self.states]

    def summary(self) -> dict[str, Any]:
        """Compact, JSON-serialisable summary for reports and comparisons."""
        rel, econ = self.metrics.reliability, self.metrics.economics
        ren, res = self.metrics.renewables, self.metrics.resilience
        return {
            "run_id": self.run_config.run_id,
            "scenario_id": self.scenario.scenario_id,
            "scenario_name": self.scenario.name,
            "required_safety_level": self.scenario.required_safety_level,
            "n_steps": len(self.states),
            "n_events": len(self.ledger),
            "wall_time_s": self.wall_time_s,
            "solve_time_s": self.solve_time_s,
            "energy_served_mwh": rel.energy_served_mwh,
            "unserved_energy_mwh": rel.unserved_energy_mwh,
            "unserved_energy_fraction": rel.unserved_energy_fraction,
            "n_steps_with_hard_violation": rel.n_steps_with_hard_violation,
            "max_line_utilisation": rel.max_line_utilisation,
            "max_line_overload_mw": rel.max_line_overload_mw,
            "min_reserve_margin": rel.min_reserve_margin,
            "reserve_shortfall_mwh": rel.reserve_shortfall_mwh,
            "min_frequency_proxy_hz": rel.min_frequency_proxy_hz,
            "total_operating_cost_usd": econ.total_operating_cost_usd,
            "unserved_energy_cost_usd": econ.unserved_energy_cost_usd,
            "co2_tonnes": econ.co2_tonnes,
            "renewable_penetration": ren.renewable_penetration,
            "curtailment_mwh": ren.curtailment_mwh,
            "storage_contribution": ren.storage_contribution,
            "detection_step": res.detection_step,
            "first_violation_step": res.first_violation_step,
            "stabilization_step": res.stabilization_step,
            "recovery_time_min": res.recovery_time_min,
            "energy_deficit_mwh": res.energy_deficit_mwh,
            "identity_hash": self.provenance.identity_hash(),
            "event_chain_root": self.provenance.event_chain_root,
            "event_tree_root": self.provenance.event_tree_root,
            "state_tree_root": self.provenance.state_tree_root,
            "final_state_hash": self.provenance.final_state_hash,
            "qradle_output_hash": self.provenance.qradle_output_hash,
            "qradle_chain_root": self.provenance.qradle_chain_root,
        }


# --------------------------------------------------------------- validation
def validate_physical_state(
    system: EnergySystem,
    state: FluxaState,
    previous: FluxaState | None,
    timestep_h: float,
    line_capacity_mw: np.ndarray,
    variable_availability_mw: np.ndarray,
) -> list[str]:
    """Check conservation laws and capacity limits. Returns failure strings.

    An empty list means the state is physically admissible. The checks are
    deliberately independent of the LP: they re-derive each law from the
    recorded state, so a solver that returned a wrong answer would be caught.
    """
    failures: list[str] = []

    # 1. Instantaneous power balance.
    supply = (
        state.total_generation_mw
        + state.battery_discharge_mw
        + state.net_import_mw
        + state.unserved_mw
    )
    demand = state.total_load_mw + state.battery_charge_mw + state.net_export_mw
    if abs(supply - demand) > 1e-6 * max(1.0, state.total_load_mw):
        failures.append(
            f"POWER_BALANCE: supply {supply:.9f} MW != demand {demand:.9f} MW "
            f"(residual {supply - demand:.3e})"
        )

    # 2. Storage: SOC bounds and energy conservation across the step.
    for bat in system.batteries:
        soc = state.battery_soc.get(bat.battery_id)
        if soc is None:
            failures.append(f"STORAGE_MISSING: no SOC recorded for {bat.battery_id}")
            continue
        if soc < bat.soc_min - SOC_TOLERANCE or soc > bat.soc_max + SOC_TOLERANCE:
            failures.append(
                f"SOC_BOUND: {bat.battery_id} soc {soc:.9f} outside "
                f"[{bat.soc_min}, {bat.soc_max}]"
            )
        if previous is not None:
            prev_soc = previous.battery_soc.get(bat.battery_id)
            if prev_soc is not None:
                expected = prev_soc * bat.energy_capacity_mwh + (
                    bat.charge_efficiency * state.battery_charge_mw * timestep_h
                    - state.battery_discharge_mw * timestep_h / bat.discharge_efficiency
                )
                actual = soc * bat.energy_capacity_mwh
                if abs(expected - actual) > ENERGY_TOLERANCE_MWH * max(
                    1.0, bat.energy_capacity_mwh
                ):
                    failures.append(
                        f"STORAGE_CONSERVATION: {bat.battery_id} stored {actual:.9f} MWh "
                        f"!= expected {expected:.9f} MWh"
                    )

    # 3. Generator capacity.
    for gen in system.generators:
        p = state.generation_by_unit_mw.get(gen.gen_id)
        if p is None:
            failures.append(f"GENERATOR_MISSING: no output recorded for {gen.gen_id}")
            continue
        if p < -POWER_TOLERANCE_MW:
            failures.append(f"NEGATIVE_GENERATION: {gen.gen_id} at {p:.9f} MW")
        if p > gen.p_max_mw + POWER_TOLERANCE_MW:
            failures.append(
                f"CAPACITY_EXCEEDED: {gen.gen_id} at {p:.9f} MW > p_max {gen.p_max_mw} MW"
            )

    # 4. Variable units may not exceed the resource that was available.
    total_var = sum(
        state.generation_by_unit_mw.get(g.gen_id, 0.0) for g in system.variable_generators
    )
    available = float(variable_availability_mw.sum())
    if total_var > available + POWER_TOLERANCE_MW:
        failures.append(
            f"AVAILABILITY_EXCEEDED: variable output {total_var:.9f} MW > "
            f"available {available:.9f} MW"
        )

    # 5. Non-negativity of the remaining physical quantities.
    for name in (
        "total_load_mw",
        "battery_charge_mw",
        "battery_discharge_mw",
        "net_import_mw",
        "net_export_mw",
        "unserved_mw",
        "curtailment_mw",
    ):
        value = getattr(state, name)
        if value < -POWER_TOLERANCE_MW:
            failures.append(f"NEGATIVE_QUANTITY: {name} = {value:.9f}")

    # 6. Reported overload must agree with flow and the step's rating.
    for i, line in enumerate(system.lines):
        flow = abs(state.line_flow_mw[line.line_id])
        expected_overload = max(flow - float(line_capacity_mw[i]), 0.0)
        reported = state.overload_mw[line.line_id]
        if abs(expected_overload - reported) > 1e-4 * max(1.0, float(line_capacity_mw[i])):
            failures.append(
                f"OVERLOAD_MISMATCH: {line.line_id} reported {reported:.6f} MW, "
                f"flow implies {expected_overload:.6f} MW"
            )

    return failures


# ------------------------------------------------------------------- engine
class SimulationEngine:
    """Runs FLUXA scenarios as QRADLE contracts."""

    def __init__(
        self,
        loaded: LoadedSystem,
        config: RunConfig,
        profile_params: ProfileParameters | None = None,
    ) -> None:
        self.loaded = loaded
        self.system = loaded.system
        self.config = config
        self.profile_params = profile_params or ProfileParameters()

        self.time = TimeGrid(
            start_iso=config.start_iso,
            timestep_s=config.timestep_s,
            n_steps=config.n_steps,
        )
        self.network = DCNetwork.from_system(self.system)
        self.problem = DispatchProblem(
            self.system, self.network, config.timestep_s, config.horizon_steps
        )
        self.baseline = build_series(
            self.system,
            self.time,
            self.profile_params,
            load_seed=config.load_seed,
            solar_seed=config.solar_seed,
            wind_seed=config.wind_seed,
            stochastic=config.stochastic_profiles,
        )
        self.qradle = DeterministicEngine()
        self._invalid_state_hook: Callable[[int, FluxaState], FluxaState] | None = None

    # ---------------------------------------------------------- public API
    def inject_invalid_state(
        self, hook: Callable[[int, FluxaState], FluxaState] | None
    ) -> None:
        """Install a hook that may corrupt a state before it is validated.

        Used by the rollback and adversarial experiments to create a state
        that violates a conservation law, so that detection can be observed
        rather than asserted. The hook receives ``(step, state)`` and returns
        the (possibly modified) state.
        """
        self._invalid_state_hook = hook

    def contract_id(self, scenario: Scenario) -> str:
        """Experiment-addressed QRADLE contract id.

        A function of the system, the scenario and the run parameters that
        affect the result -- deliberately not of the run label.
        """
        digest = canonical_hash(
            {
                "config_hash": self.loaded.config_hash,
                "model_hash": self.loaded.model_hash,
                "scenario_hash": scenario.scenario_hash(),
                "run_parameters": self.config.identity_payload(),
            }
        )
        return f"FLUXA:{self.system.system_id}:{scenario.scenario_id}:{digest[:16]}"

    def profile_hash(self, perturbed: PerturbedSeries) -> str:
        """Hash of the exogenous drivers actually used, at full precision."""
        payload = {
            "total_load_mw": perturbed.total_load_mw.tolist(),
            "bus_load_mw": perturbed.bus_load_mw.tolist(),
            "availability_mw": {
                k: v.tolist() for k, v in sorted(perturbed.availability_mw.items())
            },
            "dispatchable_availability": {
                k: v.tolist() for k, v in sorted(perturbed.dispatchable_availability.items())
            },
            "line_capacity_mw": perturbed.line_capacity_mw.tolist(),
            "seeds": perturbed.base.seeds,
        }
        return canonical_hash(payload)

    def authorize(self, scenario: Scenario, ledger: FluxaEventLedger) -> None:
        """Enforce QRADLE human oversight for this scenario.

        Raises:
            InvariantViolation: propagated from QRADLE when a
                SENSITIVE-or-higher scenario lacks authorization.
        """
        level = scenario.required_safety_level
        if level in AUTHORIZATION_REQUIRED_LEVELS and not self.config.authorized:
            ledger.append(
                FluxaEventType.AUTHORIZATION_DENIED,
                sim_timestamp=self.time.timestamp(0),
                step=-1,
                safety_level=level,
                payload={
                    "scenario_id": scenario.scenario_id,
                    "required_safety_level": level,
                    "authorized": False,
                    "reason": "QRADLE invariant 1 (human oversight) requires authorization",
                    "gated_perturbations": [
                        p.perturbation_id
                        for p in scenario.perturbations
                        if p.safety_level in AUTHORIZATION_REQUIRED_LEVELS
                    ],
                },
            )
        FatalInvariants.enforce_human_oversight(
            operation=f"fluxa_simulation:{scenario.scenario_id}",
            safety_level=level,
            authorized=self.config.authorized,
        )

    def run(self, scenario: Scenario) -> SimulationResult:
        """Execute ``scenario`` as a QRADLE contract.

        Raises:
            InvariantViolation: if authorization is insufficient.
            PhysicalInvariantError: if a state violates a conservation law and
                ``tolerate_invalid_states`` is False.
        """
        ledger = FluxaEventLedger(f"fluxa:{self.config.run_id}:{scenario.scenario_id}")
        self.authorize(scenario, ledger)

        perturbed = apply_scenario(self.system, self.baseline, scenario)
        wall_start = time.perf_counter()

        ledger.append(
            FluxaEventType.SIMULATION_CREATED,
            sim_timestamp=self.time.timestamp(0),
            step=-1,
            safety_level="ROUTINE",
            payload={
                "contract_id": self.contract_id(scenario),
                "system_id": self.system.system_id,
                "config_hash": self.loaded.config_hash,
                "model_hash": self.loaded.model_hash,
                "run_parameters": self.config.identity_payload(),
                "time_grid": self.time.to_dict(),
                "solver": self.problem.solver_metadata(),
            },
        )
        ledger.append(
            FluxaEventType.SCENARIO_INITIALIZED,
            sim_timestamp=self.time.timestamp(0),
            step=-1,
            safety_level=scenario.required_safety_level,
            payload={
                "scenario": scenario.to_dict(),
                "scenario_hash": scenario.scenario_hash(),
                "profile_hash": self.profile_hash(perturbed),
                "authorized": self.config.authorized,
            },
        )

        # The QRADLE contract id addresses the *experiment*, not the run
        # label. QRADLE folds the contract id into ``output_hash``, so a
        # label-dependent id would make two executions of the same experiment
        # produce different output hashes and determinism would be
        # unverifiable. The run label travels in ``metadata`` instead.
        contract_id = self.contract_id(scenario)
        context = ExecutionContext(
            contract_id=contract_id,
            parameters={
                "config_hash": self.loaded.config_hash,
                "model_hash": self.loaded.model_hash,
                "scenario_hash": scenario.scenario_hash(),
                "run_parameters": self.config.identity_payload(),
            },
            timestamp=self.time.timestamp(0),
            safety_level=scenario.required_safety_level,
            authorized=self.config.authorized,
            metadata={
                "vertical": "FLUXA",
                "substrate": "cpu",
                "run_id": self.config.run_id,
                "authorizer_id": self.config.authorizer_id,
            },
        )

        trajectory: dict[str, Any] = {}

        def executor(_params: dict[str, Any]) -> dict[str, Any]:
            trajectory.update(self._simulate(scenario, perturbed, ledger))
            return {
                "n_steps": len(trajectory["states"]),
                "final_state_hash": trajectory["states"][-1].state_hash(),
                "state_tree_root": trajectory["state_tree_root"],
                "event_chain_root": ledger.root_hash(),
                "metrics": trajectory["metrics"].output_payload(),
            }

        qradle_result = self.qradle.execute_contract(context, executor, create_checkpoint=True)
        if not qradle_result.success:
            raise PhysicalInvariantError(
                f"QRADLE contract execution failed: {qradle_result.error}"
            )

        wall_time = time.perf_counter() - wall_start
        states: list[FluxaState] = trajectory["states"]
        metrics: RunMetrics = trajectory["metrics"]

        ledger.append(
            FluxaEventType.SIMULATION_COMPLETED,
            sim_timestamp=self.time.timestamp(self.config.n_steps - 1),
            step=self.config.n_steps - 1,
            safety_level="ROUTINE",
            payload={
                "n_steps": len(states),
                "final_state_hash": states[-1].state_hash(),
                "output_hash": sha256_of(metrics.output_payload()),
                "qradle_output_hash": qradle_result.output_hash,
                "state_tree_root": trajectory["state_tree_root"],
            },
        )

        event_tree = build_merkle_tree(ledger.event_hashes())
        provenance = ProvenanceBundle(
            run_id=self.config.run_id,
            config_hash=self.loaded.config_hash,
            model_hash=self.loaded.model_hash,
            scenario_hash=scenario.scenario_hash(),
            profile_hash=self.profile_hash(perturbed),
            run_parameters=self.config.identity_payload(),
            environment=environment_fingerprint(),
            solver=self.problem.solver_metadata(),
            event_chain_root=ledger.root_hash(),
            event_tree_root=event_tree.root,
            state_tree_root=trajectory["state_tree_root"],
            final_state_hash=states[-1].state_hash(),
            output_hash=sha256_of(metrics.output_payload()),
            n_events=len(ledger),
            n_states=len(states),
            qradle_output_hash=qradle_result.output_hash,
            qradle_chain_root=self.qradle.merkle_chain.get_root_hash(),
        )

        return SimulationResult(
            run_config=self.config,
            scenario=scenario,
            system=self.system,
            states=states,
            ledger=ledger,
            metrics=metrics,
            provenance=provenance,
            qradle_result=qradle_result,
            checkpoint_ids=trajectory["checkpoint_ids"],
            wall_time_s=wall_time,
            solve_time_s=trajectory["solve_time_s"],
            detection_step=trajectory["detection_step"],
            invalid_state_reports=trajectory["invalid_state_reports"],
        )

    # ------------------------------------------------------------ internals
    def _window(self, perturbed: PerturbedSeries, step: int) -> list[DispatchStep]:
        """Build a look-ahead window, padding the tail by holding the last step.

        Padding keeps the LP structure fixed for every solve, which is what
        lets the constraint matrices be built once.
        """
        var_ids = [g.gen_id for g in self.system.variable_generators]
        disp_ids = [g.gen_id for g in self.system.dispatchable_generators]
        last = self.config.n_steps - 1
        window: list[DispatchStep] = []
        for offset in range(self.config.horizon_steps):
            k = min(step + offset, last)
            window.append(
                DispatchStep(
                    bus_load_mw=perturbed.bus_load_mw[k],
                    variable_availability_mw=np.array(
                        [perturbed.availability_mw[g][k] for g in var_ids], dtype=np.float64
                    ),
                    dispatchable_availability=np.array(
                        [perturbed.dispatchable_availability[g][k] for g in disp_ids],
                        dtype=np.float64,
                    ),
                    line_capacity_mw=perturbed.line_capacity_mw[k],
                )
            )
        return window

    def step_once(
        self,
        scenario: Scenario,
        perturbed: PerturbedSeries,
        step: int,
        soc: np.ndarray,
        p_disp_prev: np.ndarray,
        avail_prev: np.ndarray,
        *,
        previous_state: FluxaState | None = None,
        in_restoration: bool = False,
    ) -> tuple[FluxaState, DispatchSolution, list[str]]:
        """Advance one timestep: solve, assemble the state, validate it.

        This is the single physics entry point. Both the main run loop and the
        rollback experiment call it, so a recovered-and-resumed trajectory is
        produced by exactly the same code as the original one -- which is what
        makes "resume deterministic execution" a verifiable claim rather than
        a structural coincidence.

        Returns:
            ``(state, solution, validation_failures)``. An empty failure list
            means the state satisfies every physical invariant. The caller
            decides whether a failure halts the run or is recorded.
        """
        rc = self.config
        cfg = self.system
        var_ids = [g.gen_id for g in cfg.variable_generators]
        ramp_per_step = np.array(
            [g.ramp_mw_per_min * rc.timestep_s / 60.0 for g in cfg.dispatchable_generators],
            dtype=np.float64,
        )

        window = self._window(perturbed, step)
        solution = self.problem.solve(window, soc, p_disp_prev, avail_prev)

        active = {
            pid: float(series[step])
            for pid, series in perturbed.perturbation_intensity.items()
            if series[step] > 0.0
        }
        outage_active = any(
            p.perturbation_id in active
            for p in scenario.perturbations
            if p.kind is PerturbationKind.GENERATOR_OUTAGE
        )
        major_change = bool(
            np.any(
                np.abs(solution.p_disp_mw - p_disp_prev)
                > rc.major_dispatch_change_fraction * ramp_per_step
            )
        )
        availability = np.array(
            [perturbed.availability_mw[g][step] for g in var_ids], dtype=np.float64
        )

        state = build_state(
            system=cfg,
            solution=solution,
            step=step,
            timestamp=self.time.timestamp(step),
            scenario_id=scenario.scenario_id,
            run_id=rc.run_id,
            bus_load_mw=perturbed.bus_load_mw[step],
            line_capacity_mw=perturbed.line_capacity_mw[step],
            variable_availability_mw=availability,
            active_perturbations=active,
            outage_active=outage_active,
            major_dispatch_change=major_change,
            in_restoration=in_restoration,
            timestep_h=rc.timestep_h,
        )

        if self._invalid_state_hook is not None:
            state = self._invalid_state_hook(step, state)

        failures = validate_physical_state(
            cfg,
            state,
            previous_state,
            rc.timestep_h,
            perturbed.line_capacity_mw[step],
            availability,
        )
        return state, solution, failures

    def _simulate(
        self,
        scenario: Scenario,
        perturbed: PerturbedSeries,
        ledger: FluxaEventLedger,
    ) -> dict[str, Any]:
        """The timestep loop. Pure with respect to engine state except the ledger."""
        cfg = self.system
        rc = self.config
        states: list[FluxaState] = []
        checkpoint_ids: list[str] = []
        invalid_reports: list[dict[str, Any]] = []
        solve_time = 0.0

        soc = np.array([b.soc_initial for b in cfg.batteries], dtype=np.float64)
        p_prev = np.array([g.p_min_mw for g in cfg.dispatchable_generators], dtype=np.float64)
        ramp_per_step = np.array(
            [g.ramp_mw_per_min * rc.timestep_s / 60.0 for g in cfg.dispatchable_generators],
            dtype=np.float64,
        )
        disp_ids = [g.gen_id for g in cfg.dispatchable_generators]
        avail_prev = np.ones(len(disp_ids), dtype=np.float64)
        detection_step: int | None = None
        detection_emitted = False
        stabilized_emitted = False
        clean_run = 0
        had_episode = False
        previous_state: FluxaState | None = None
        load_threshold = rc.detection_threshold_fraction * cfg.peak_load_mw

        for step in range(rc.n_steps):
            t0 = time.perf_counter()
            state, solution, failures = self.step_once(
                scenario,
                perturbed,
                step,
                soc,
                p_prev,
                avail_prev,
                previous_state=previous_state,
                in_restoration=(had_episode and clean_run < rc.stabilization_steps),
            )
            solve_time += time.perf_counter() - t0
            active = state.active_perturbations
            if failures:
                report = {
                    "step": step,
                    "timestamp": state.timestamp,
                    "failures": failures,
                    "state_hash": state.state_hash(),
                }
                invalid_reports.append(report)
                ledger.append(
                    FluxaEventType.INVALID_STATE_DETECTED,
                    sim_timestamp=state.timestamp,
                    step=step,
                    safety_level="CRITICAL",
                    payload=report,
                )
                if not rc.tolerate_invalid_states:
                    raise PhysicalInvariantError(
                        f"step {step}: physical invariant violated: {failures}"
                    )

            # ---- events --------------------------------------------------
            if rc.emit_per_step_events:
                self._emit_step_events(
                    ledger=ledger,
                    state=state,
                    previous=previous_state,
                    solution=solution,
                    p_prev=p_prev,
                    ramp_per_step=ramp_per_step,
                    load_threshold=load_threshold,
                )

            if active and not detection_emitted:
                observable = self._observable_deviation(perturbed, step)
                if observable is not None:
                    detection_step = step
                    detection_emitted = True
                    ledger.append(
                        FluxaEventType.DISTURBANCE_DETECTED,
                        sim_timestamp=state.timestamp,
                        step=step,
                        safety_level=scenario.required_safety_level,
                        payload={
                            "active_perturbations": active,
                            "observable_deviation": observable,
                            "detection_threshold_fraction": rc.detection_threshold_fraction,
                            "operational_state": state.operational_state.value,
                        },
                    )

            # RESTORATIVE counts as clean: it is the label carried *while*
            # recovering, so excluding it would make stabilisation unreachable.
            if state.operational_state in (OperationalState.ALERT, OperationalState.EMERGENCY):
                had_episode = True
                clean_run = 0
                stabilized_emitted = False
            else:
                clean_run += 1
                if (
                    had_episode
                    and not stabilized_emitted
                    and clean_run >= rc.stabilization_steps
                ):
                    stabilized_emitted = True
                    ledger.append(
                        FluxaEventType.SYSTEM_STABILIZED,
                        sim_timestamp=state.timestamp,
                        step=step,
                        safety_level="ROUTINE",
                        payload={
                            "consecutive_normal_steps": clean_run,
                            "stabilization_window_steps": rc.stabilization_steps,
                            "first_normal_step": step - clean_run + 1,
                            "reserve_margin": state.reserve_margin,
                        },
                    )

            if rc.checkpoint_every > 0 and step % rc.checkpoint_every == 0:
                # The checkpoint id is supplied explicitly. QRADLE's default
                # id embeds int(datetime.now().timestamp()), which would make
                # the id -- and therefore the CHECKPOINT_CREATED event and the
                # event-chain root -- wall-clock dependent. The state payload
                # uses the label-free state view for the same reason.
                checkpoint = self.qradle.rollback_manager.create_checkpoint(
                    state_data={
                        "step": step,
                        "state": state.hashable_dict(),
                        "soc": soc.tolist(),
                        "p_disp_prev": p_prev.tolist(),
                        "avail_prev": avail_prev.tolist(),
                        "event_chain_root": ledger.root_hash(),
                        "n_events": len(ledger),
                    },
                    checkpoint_id=f"{self.contract_id(scenario)}:ckpt{step:06d}",
                    metadata={
                        "run_id": rc.run_id,
                        "scenario_id": scenario.scenario_id,
                        "step": step,
                    },
                )
                checkpoint_ids.append(checkpoint.checkpoint_id)
                ledger.append(
                    FluxaEventType.CHECKPOINT_CREATED,
                    sim_timestamp=state.timestamp,
                    step=step,
                    safety_level="ROUTINE",
                    payload={
                        "checkpoint_id": checkpoint.checkpoint_id,
                        "state_hash": checkpoint.state_hash,
                        "step": step,
                    },
                )

            states.append(state)
            previous_state = state
            soc = solution.soc_end.copy()
            p_prev = solution.p_disp_mw.copy()
            avail_prev = np.array(
                [perturbed.dispatchable_availability[g][step] for g in disp_ids],
                dtype=np.float64,
            )

        metrics = compute_all(
            cfg,
            states,
            rc.timestep_h,
            scenario_id=scenario.scenario_id,
            run_id=rc.run_id,
            onset_h=scenario.first_onset_h,
            detection_step=detection_step,
            stabilization_steps=rc.stabilization_steps,
        )
        state_tree = build_merkle_tree([s.state_hash() for s in states])

        return {
            "states": states,
            "metrics": metrics,
            "checkpoint_ids": checkpoint_ids,
            "solve_time_s": solve_time,
            "detection_step": detection_step,
            "state_tree_root": state_tree.root,
            "invalid_state_reports": invalid_reports,
        }

    def _observable_deviation(
        self, perturbed: PerturbedSeries, step: int
    ) -> dict[str, float] | None:
        """Largest relative departure of a driver from its baseline at ``step``.

        Returns None if no driver has yet moved by more than the configured
        detection threshold -- this is what makes "detection time" a measured
        observability latency rather than an assumed delay.
        """
        threshold = self.config.detection_threshold_fraction
        base = perturbed.base
        found: dict[str, float] = {}

        base_load = float(base.total_load_mw[step])
        if base_load > 0.0:
            rel = abs(float(perturbed.total_load_mw[step]) - base_load) / base_load
            if rel > threshold:
                found["total_load_mw"] = rel

        for gen_id, series in perturbed.availability_mw.items():
            reference = float(base.availability_mw[gen_id][step])
            scale = max(reference, 0.01 * self.system.peak_load_mw)
            rel = abs(float(series[step]) - reference) / scale
            if rel > threshold:
                found[f"availability:{gen_id}"] = rel

        for gen_id, series in perturbed.dispatchable_availability.items():
            if float(series[step]) < 1.0 - threshold:
                found[f"dispatchable_availability:{gen_id}"] = 1.0 - float(series[step])

        for i, line in enumerate(self.system.lines):
            rated = line.capacity_mw
            rel = abs(float(perturbed.line_capacity_mw[step, i]) - rated) / rated
            if rel > threshold:
                found[f"line_capacity:{line.line_id}"] = rel

        return found or None

    def _emit_step_events(
        self,
        *,
        ledger: FluxaEventLedger,
        state: FluxaState,
        previous: FluxaState | None,
        solution: DispatchSolution,
        p_prev: np.ndarray,
        ramp_per_step: np.ndarray,
        load_threshold: float,
    ) -> None:
        """Emit the per-timestep event set for one committed step."""
        ledger.append(
            FluxaEventType.STATE_ADVANCED,
            sim_timestamp=state.timestamp,
            step=state.step,
            safety_level=state.safety_level,
            payload={
                "state_hash": state.state_hash(),
                "operational_state": state.operational_state.value,
                "total_load_mw": state.total_load_mw,
                "total_generation_mw": state.total_generation_mw,
                "renewable_generation_mw": state.renewable_generation_mw,
                "battery_soc": state.battery_soc,
                "net_import_mw": state.net_import_mw,
                "max_line_utilisation": state.max_line_utilisation,
                "reserve_margin": state.reserve_margin,
                "unserved_mw": state.unserved_mw,
                "frequency_proxy_hz": state.frequency_proxy_hz,
                "balance_residual_mw": state.balance_residual_mw,
            },
        )

        if previous is not None:
            delta_load = state.total_load_mw - previous.total_load_mw
            if abs(delta_load) > load_threshold:
                ledger.append(
                    FluxaEventType.LOAD_CHANGED,
                    sim_timestamp=state.timestamp,
                    step=state.step,
                    safety_level=state.safety_level,
                    payload={
                        "previous_load_mw": previous.total_load_mw,
                        "load_mw": state.total_load_mw,
                        "delta_mw": delta_load,
                        "threshold_mw": load_threshold,
                    },
                )

        changed = {
            gen.gen_id: (float(p_prev[i]), float(solution.p_disp_mw[i]))
            for i, gen in enumerate(self.system.dispatchable_generators)
            if abs(float(solution.p_disp_mw[i]) - float(p_prev[i]))
            > self.config.major_dispatch_change_fraction * float(ramp_per_step[i])
        }
        if changed:
            ledger.append(
                FluxaEventType.GENERATION_CHANGED,
                sim_timestamp=state.timestamp,
                step=state.step,
                safety_level=state.safety_level,
                payload={
                    "units": {k: {"from_mw": v[0], "to_mw": v[1]} for k, v in changed.items()},
                    "ramp_capability_mw_per_step": {
                        gen.gen_id: float(ramp_per_step[i])
                        for i, gen in enumerate(self.system.dispatchable_generators)
                    },
                },
            )

        if (
            state.battery_charge_mw > POWER_TOLERANCE_MW
            or state.battery_discharge_mw > POWER_TOLERANCE_MW
        ):
            ledger.append(
                FluxaEventType.STORAGE_DISPATCHED,
                sim_timestamp=state.timestamp,
                step=state.step,
                safety_level=state.safety_level,
                payload={
                    "charge_mw": state.battery_charge_mw,
                    "discharge_mw": state.battery_discharge_mw,
                    "soc": state.battery_soc,
                    "stored_mwh": state.battery_stored_mwh,
                    "simultaneous_charge_discharge_mw": (
                        solution.simultaneous_charge_discharge_mw
                    ),
                },
            )

        hard = [v for v in state.violations if not v.startswith("LINE_NEAR_LIMIT")]
        near = [v for v in state.violations if v.startswith("LINE_NEAR_LIMIT")]
        if hard:
            ledger.append(
                FluxaEventType.CONSTRAINT_VIOLATED,
                sim_timestamp=state.timestamp,
                step=state.step,
                safety_level=state.safety_level,
                payload={
                    "violations": hard,
                    "unserved_mw": state.unserved_mw,
                    "overload_mw": state.overload_mw,
                    "reserve_shortfall_mw": state.reserve_shortfall_mw,
                    "line_utilisation": state.line_utilisation,
                    "operational_state": state.operational_state.value,
                },
            )
        if near:
            ledger.append(
                FluxaEventType.CONSTRAINT_APPROACHED,
                sim_timestamp=state.timestamp,
                step=state.step,
                safety_level=state.safety_level,
                payload={
                    "approached": near,
                    "max_line_utilisation": state.max_line_utilisation,
                    "line_utilisation": state.line_utilisation,
                },
            )

        # A mitigation is an observable corrective action taken while a
        # disturbance is active: starting the peaker, increasing import, or
        # increasing battery discharge beyond a material step change.
        if state.active_perturbations and previous is not None:
            actions: dict[str, Any] = {}
            peaker_ids = [
                g.gen_id
                for g in self.system.dispatchable_generators
                if g.kind.value == "gas_peaker"
            ]
            for gen_id in peaker_ids:
                before = previous.generation_by_unit_mw.get(gen_id, 0.0)
                after = state.generation_by_unit_mw.get(gen_id, 0.0)
                if after - before > 1.0:
                    actions[f"peaker_increase:{gen_id}"] = after - before
            if state.net_import_mw - previous.net_import_mw > 1.0:
                actions["import_increase_mw"] = state.net_import_mw - previous.net_import_mw
            if state.battery_discharge_mw - previous.battery_discharge_mw > 1.0:
                actions["battery_discharge_increase_mw"] = (
                    state.battery_discharge_mw - previous.battery_discharge_mw
                )
            if state.unserved_mw > POWER_TOLERANCE_MW:
                actions["load_shed_mw"] = state.unserved_mw
            if actions:
                ledger.append(
                    FluxaEventType.MITIGATION_EXECUTED,
                    sim_timestamp=state.timestamp,
                    step=state.step,
                    safety_level=state.safety_level,
                    payload={
                        "actions": actions,
                        "active_perturbations": state.active_perturbations,
                        "operational_state": state.operational_state.value,
                    },
                )
