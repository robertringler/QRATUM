"""Checkpoint / rollback / resume for FLUXA, built on QRADLE's RollbackManager.

The experiment this module implements
-------------------------------------
Advance t0 -> t1 -> t2 -> t3 -> t4 with a checkpoint at every step, corrupt the
state at t4, then:

    detect invalid state
      -> authorize rollback (QRADLE invariant 1, SENSITIVE)
      -> restore the t2 checkpoint
      -> verify the restored state against the recorded t2 state hash
      -> resume deterministic execution from t2

Every stage emits an event, and each stage's success is established by a
recomputed hash comparison rather than by the absence of an exception.

Rollback is a QRADLE operation, not a FLUXA one: checkpoints are created and
restored through :class:`qradle.core.rollback.RollbackManager`, whose
``Checkpoint.verify()`` recomputes the state hash from the stored state data.

Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from fluxa.engine import RunConfig, SimulationEngine
from fluxa.events import FluxaEventLedger, FluxaEventType
from fluxa.scenarios import Scenario, apply_scenario
from fluxa.state import FluxaState
from qradle.core.invariants import FatalInvariants, InvariantViolation
from qradle.core.rollback import Checkpoint

LOGGER = logging.getLogger("fluxa.recovery")

#: Restoring a simulation to an earlier state discards recorded history, so it
#: is classified SENSITIVE and gated on human authorization.
ROLLBACK_SAFETY_LEVEL = "SENSITIVE"


class RollbackError(RuntimeError):
    """Raised when a rollback cannot be completed or verified."""


@dataclass
class StepRecord:
    """One advanced step, with the checkpoint taken immediately after it."""

    step: int
    state: FluxaState
    state_hash: str
    checkpoint_id: str
    checkpoint_state_hash: str
    soc_after: list[float]
    p_disp_after: list[float]
    avail_after: list[float]
    validation_failures: list[str] = field(default_factory=list)


@dataclass
class RollbackReport:
    """Outcome of the rollback experiment. Every field is measured."""

    scenario_id: str
    forward_steps: int
    checkpoint_step: int
    corrupted_step: int
    corruption_description: str

    invalid_state_detected: bool
    detected_at_step: int | None
    detection_failures: list[str]

    rollback_authorized: bool
    rollback_denied_without_authorization: bool
    authorization_error: str

    checkpoint_verified_before_restore: bool
    restored_state_hash: str
    expected_state_hash: str
    restore_matched_checkpoint: bool

    resumed_steps: int
    resumed_state_hashes: list[str]
    reference_state_hashes: list[str]
    resume_is_deterministic: bool
    first_resume_divergence_step: int | None

    event_sequence: list[str]
    ledger_verified: bool
    ledger_root: str

    def success(self) -> bool:
        """Rollback succeeded only if every stage was independently verified."""
        return (
            self.invalid_state_detected
            and self.rollback_denied_without_authorization
            and self.rollback_authorized
            and self.checkpoint_verified_before_restore
            and self.restore_matched_checkpoint
            and self.resume_is_deterministic
            and self.ledger_verified
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {k: v for k, v in self.__dict__.items()}
        payload["success"] = self.success()
        return payload


def corrupt_power_balance(magnitude_mw: float = 25.0) -> Callable[[int, FluxaState], FluxaState]:
    """Build a corruption hook that breaks instantaneous power balance.

    The injected fault adds phantom generation without matching demand, which
    is the clearest possible conservation-law violation: it cannot be an
    artefact of solver tolerance at this magnitude.
    """

    def hook(step: int, state: FluxaState) -> FluxaState:
        import dataclasses

        unit = next(iter(state.generation_by_unit_mw))
        return dataclasses.replace(
            state,
            total_generation_mw=state.total_generation_mw + magnitude_mw,
            generation_by_unit_mw={
                **state.generation_by_unit_mw,
                unit: state.generation_by_unit_mw[unit] + magnitude_mw,
            },
            extra={
                **state.extra,
                "injected_fault": {
                    "kind": "phantom_generation",
                    "magnitude_mw": magnitude_mw,
                    "unit": unit,
                },
            },
        )

    return hook


def corrupt_soc_bound(excess: float = 0.4) -> Callable[[int, FluxaState], FluxaState]:
    """Build a corruption hook that pushes state of charge above its limit."""

    def hook(step: int, state: FluxaState) -> FluxaState:
        import dataclasses

        battery = next(iter(state.battery_soc))
        return dataclasses.replace(
            state,
            battery_soc={**state.battery_soc, battery: state.battery_soc[battery] + excess},
            extra={
                **state.extra,
                "injected_fault": {"kind": "soc_overflow", "excess": excess, "battery": battery},
            },
        )

    return hook


class RollbackExperiment:
    """Drives the detect -> authorize -> restore -> verify -> resume sequence."""

    def __init__(
        self,
        engine: SimulationEngine,
        scenario: Scenario,
        *,
        forward_steps: int = 5,
        checkpoint_step: int = 2,
        corrupted_step: int = 4,
    ) -> None:
        if not 0 <= checkpoint_step < corrupted_step < forward_steps:
            raise ValueError(
                "require 0 <= checkpoint_step < corrupted_step < forward_steps, got "
                f"{checkpoint_step}, {corrupted_step}, {forward_steps}"
            )
        if not engine.config.tolerate_invalid_states:
            raise ValueError(
                "the engine must be configured with tolerate_invalid_states=True so the "
                "corrupted state can be observed and recovered from instead of halting"
            )
        self.engine = engine
        self.scenario = scenario
        self.forward_steps = forward_steps
        self.checkpoint_step = checkpoint_step
        self.corrupted_step = corrupted_step

    # ------------------------------------------------------------------ run
    def run(
        self,
        corruption: Callable[[int, FluxaState], FluxaState] | None = None,
        corruption_description: str = "phantom generation injected at the corrupted step",
    ) -> RollbackReport:
        """Execute the experiment and return a fully measured report."""
        engine = self.engine
        system = engine.system
        perturbed = apply_scenario(system, engine.baseline, self.scenario)
        ledger = FluxaEventLedger(f"fluxa-rollback:{self.scenario.scenario_id}")
        corruption = corruption or corrupt_power_balance()

        ledger.append(
            FluxaEventType.SIMULATION_CREATED,
            sim_timestamp=engine.time.timestamp(0),
            step=-1,
            safety_level="ROUTINE",
            payload={
                "experiment": "rollback",
                "contract_id": engine.contract_id(self.scenario),
                "forward_steps": self.forward_steps,
                "checkpoint_step": self.checkpoint_step,
                "corrupted_step": self.corrupted_step,
            },
        )

        # ---- reference trajectory: clean, no corruption ------------------
        engine.inject_invalid_state(None)
        reference = self._advance(perturbed, 0, self.forward_steps, ledger=None)
        reference_hashes = [r.state_hash for r in reference]

        # ---- forward pass with the fault injected at corrupted_step ------
        engine.inject_invalid_state(
            lambda step, state: corruption(step, state) if step == self.corrupted_step else state
        )
        records = self._advance(perturbed, 0, self.forward_steps, ledger=ledger)
        engine.inject_invalid_state(None)

        faulted = records[self.corrupted_step]
        detected = bool(faulted.validation_failures)
        if detected:
            ledger.append(
                FluxaEventType.INVALID_STATE_DETECTED,
                sim_timestamp=faulted.state.timestamp,
                step=faulted.step,
                safety_level="CRITICAL",
                payload={
                    "step": faulted.step,
                    "failures": faulted.validation_failures,
                    "state_hash": faulted.state_hash,
                    "injected_fault": faulted.state.extra.get("injected_fault"),
                },
            )

        # ---- authorization gate on the rollback itself -------------------
        denied_without_authorization = False
        try:
            FatalInvariants.enforce_human_oversight(
                operation=f"fluxa_rollback:{self.scenario.scenario_id}",
                safety_level=ROLLBACK_SAFETY_LEVEL,
                authorized=False,
            )
        except InvariantViolation as exc:
            denied_without_authorization = True
            authorization_error = str(exc)
            ledger.append(
                FluxaEventType.AUTHORIZATION_DENIED,
                sim_timestamp=faulted.state.timestamp,
                step=faulted.step,
                safety_level=ROLLBACK_SAFETY_LEVEL,
                payload={
                    "operation": "rollback",
                    "authorized": False,
                    "reason": authorization_error,
                },
            )
        else:
            authorization_error = ""

        authorized = engine.config.authorized
        FatalInvariants.enforce_human_oversight(
            operation=f"fluxa_rollback:{self.scenario.scenario_id}",
            safety_level=ROLLBACK_SAFETY_LEVEL,
            authorized=authorized,
        )
        target = records[self.checkpoint_step]
        ledger.append(
            FluxaEventType.ROLLBACK_AUTHORIZED,
            sim_timestamp=faulted.state.timestamp,
            step=faulted.step,
            safety_level=ROLLBACK_SAFETY_LEVEL,
            payload={
                "authorizer_id": engine.config.authorizer_id,
                "target_checkpoint_id": target.checkpoint_id,
                "target_step": target.step,
                "discarded_steps": list(range(target.step + 1, self.forward_steps)),
            },
        )

        # ---- restore ------------------------------------------------------
        manager = engine.qradle.rollback_manager
        checkpoint: Checkpoint | None = manager.get_checkpoint(target.checkpoint_id)
        if checkpoint is None:
            raise RollbackError(f"checkpoint {target.checkpoint_id} is not retrievable")
        checkpoint_verified = checkpoint.verify()

        restored = manager.rollback_to(target.checkpoint_id)
        restored_state = restored["state"]
        restored_hash = _hash_state_dict(restored_state)
        restore_matched = restored_hash == target.state_hash

        ledger.append(
            FluxaEventType.ROLLBACK_EXECUTED,
            sim_timestamp=target.state.timestamp,
            step=target.step,
            safety_level=ROLLBACK_SAFETY_LEVEL,
            payload={
                "checkpoint_id": target.checkpoint_id,
                "checkpoint_verified": checkpoint_verified,
                "restored_state_hash": restored_hash,
                "expected_state_hash": target.state_hash,
                "restore_matched": restore_matched,
            },
        )

        # ---- resume from the restored state ------------------------------
        resumed = self._advance(
            perturbed,
            target.step + 1,
            self.forward_steps,
            ledger=ledger,
            soc=np.array(restored["soc"], dtype=np.float64),
            p_disp_prev=np.array(restored["p_disp_prev"], dtype=np.float64),
            avail_prev=np.array(restored["avail_prev"], dtype=np.float64),
            previous_state=target.state,
            checkpointing=False,
        )
        resumed_hashes = [r.state_hash for r in resumed]
        expected_tail = reference_hashes[target.step + 1 :]
        deterministic = resumed_hashes == expected_tail
        divergence = None
        if not deterministic:
            for offset, (a, b) in enumerate(zip(resumed_hashes, expected_tail, strict=False)):
                if a != b:
                    divergence = target.step + 1 + offset
                    break

        ledger.append(
            FluxaEventType.SYSTEM_STABILIZED if deterministic else FluxaEventType.INVALID_STATE_DETECTED,
            sim_timestamp=engine.time.timestamp(self.forward_steps - 1),
            step=self.forward_steps - 1,
            safety_level="ROUTINE" if deterministic else "CRITICAL",
            payload={
                "resumed_from_step": target.step,
                "resumed_steps": len(resumed_hashes),
                "resume_is_deterministic": deterministic,
                "first_divergence_step": divergence,
            },
        )
        ledger.append(
            FluxaEventType.SIMULATION_COMPLETED,
            sim_timestamp=engine.time.timestamp(self.forward_steps - 1),
            step=self.forward_steps - 1,
            safety_level="ROUTINE",
            payload={"experiment": "rollback", "n_events": len(ledger) + 1},
        )

        ledger_ok, _ = ledger.verify()
        return RollbackReport(
            scenario_id=self.scenario.scenario_id,
            forward_steps=self.forward_steps,
            checkpoint_step=self.checkpoint_step,
            corrupted_step=self.corrupted_step,
            corruption_description=corruption_description,
            invalid_state_detected=detected,
            detected_at_step=faulted.step if detected else None,
            detection_failures=faulted.validation_failures,
            rollback_authorized=authorized,
            rollback_denied_without_authorization=denied_without_authorization,
            authorization_error=authorization_error,
            checkpoint_verified_before_restore=checkpoint_verified,
            restored_state_hash=restored_hash,
            expected_state_hash=target.state_hash,
            restore_matched_checkpoint=restore_matched,
            resumed_steps=len(resumed_hashes),
            resumed_state_hashes=resumed_hashes,
            reference_state_hashes=reference_hashes,
            resume_is_deterministic=deterministic,
            first_resume_divergence_step=divergence,
            event_sequence=[e.event_type.value for e in ledger],
            ledger_verified=ledger_ok,
            ledger_root=ledger.root_hash(),
        )

    # ------------------------------------------------------------ internals
    def _advance(
        self,
        perturbed,
        start: int,
        stop: int,
        *,
        ledger: FluxaEventLedger | None,
        soc: np.ndarray | None = None,
        p_disp_prev: np.ndarray | None = None,
        avail_prev: np.ndarray | None = None,
        previous_state: FluxaState | None = None,
        checkpointing: bool = True,
    ) -> list[StepRecord]:
        """Advance ``[start, stop)`` through :meth:`SimulationEngine.step_once`."""
        engine = self.engine
        system = engine.system
        disp_ids = [g.gen_id for g in system.dispatchable_generators]

        soc = (
            soc
            if soc is not None
            else np.array([b.soc_initial for b in system.batteries], dtype=np.float64)
        )
        p_prev = (
            p_disp_prev
            if p_disp_prev is not None
            else np.array([g.p_min_mw for g in system.dispatchable_generators], dtype=np.float64)
        )
        avail = (
            avail_prev
            if avail_prev is not None
            else np.ones(len(disp_ids), dtype=np.float64)
        )

        records: list[StepRecord] = []
        for step in range(start, stop):
            state, solution, failures = engine.step_once(
                self.scenario, perturbed, step, soc, p_prev, avail,
                previous_state=previous_state,
            )
            next_soc = solution.soc_end.copy()
            next_p = solution.p_disp_mw.copy()
            next_avail = np.array(
                [perturbed.dispatchable_availability[g][step] for g in disp_ids],
                dtype=np.float64,
            )

            checkpoint_id = ""
            checkpoint_hash = ""
            if checkpointing:
                checkpoint = engine.qradle.rollback_manager.create_checkpoint(
                    state_data={
                        "step": step,
                        "state": state.hashable_dict(),
                        "soc": next_soc.tolist(),
                        "p_disp_prev": next_p.tolist(),
                        "avail_prev": next_avail.tolist(),
                    },
                    checkpoint_id=f"{engine.contract_id(self.scenario)}:rb{step:06d}",
                    metadata={"experiment": "rollback", "step": step},
                )
                checkpoint_id = checkpoint.checkpoint_id
                checkpoint_hash = checkpoint.state_hash
                if ledger is not None:
                    ledger.append(
                        FluxaEventType.CHECKPOINT_CREATED,
                        sim_timestamp=state.timestamp,
                        step=step,
                        safety_level="ROUTINE",
                        payload={
                            "checkpoint_id": checkpoint_id,
                            "state_hash": checkpoint_hash,
                            "step": step,
                        },
                    )

            if ledger is not None:
                ledger.append(
                    FluxaEventType.STATE_ADVANCED,
                    sim_timestamp=state.timestamp,
                    step=step,
                    safety_level=state.safety_level,
                    payload={
                        "state_hash": state.state_hash(),
                        "operational_state": state.operational_state.value,
                        "validation_failures": failures,
                    },
                )

            records.append(
                StepRecord(
                    step=step,
                    state=state,
                    state_hash=state.state_hash(),
                    checkpoint_id=checkpoint_id,
                    checkpoint_state_hash=checkpoint_hash,
                    soc_after=next_soc.tolist(),
                    p_disp_after=next_p.tolist(),
                    avail_after=next_avail.tolist(),
                    validation_failures=failures,
                )
            )
            soc, p_prev, avail, previous_state = next_soc, next_p, next_avail, state
        return records


def _hash_state_dict(state_dict: dict[str, Any]) -> str:
    """Hash a restored state dict the same way :meth:`FluxaState.state_hash` does."""
    import hashlib
    import json

    blob = json.dumps(state_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_rollback_engine(loaded, run_id: str = "rollback", **overrides) -> SimulationEngine:
    """Construct an engine configured for the rollback experiment."""
    config = RunConfig(
        run_id=run_id,
        n_steps=overrides.pop("n_steps", 12),
        checkpoint_every=0,
        authorized=overrides.pop("authorized", True),
        authorizer_id=overrides.pop("authorizer_id", "operator_rollback"),
        tolerate_invalid_states=True,
        emit_per_step_events=False,
        **overrides,
    )
    return SimulationEngine(loaded, config)
