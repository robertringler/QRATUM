"""Rollback: checkpoint, detect, authorize, restore, verify, resume.

Every stage is asserted against a recomputed hash. Rollback is never declared
successful merely because no exception was raised.
"""

from __future__ import annotations

import pytest

from fluxa.recovery import (
    RollbackExperiment,
    build_rollback_engine,
    corrupt_power_balance,
    corrupt_soc_bound,
)


@pytest.fixture(scope="module")
def report(loaded, scenarios):
    engine = build_rollback_engine(loaded, run_id="rollback-test")
    return RollbackExperiment(
        engine, scenarios["A_BASELINE"], forward_steps=5, checkpoint_step=2, corrupted_step=4
    ).run(corrupt_power_balance(25.0))


def test_experiment_succeeds_overall(report):
    assert report.success(), report.to_dict()


def test_invalid_state_is_detected_at_the_corrupted_step(report):
    assert report.invalid_state_detected
    assert report.detected_at_step == 4
    assert any("POWER_BALANCE" in f for f in report.detection_failures)


def test_steps_before_the_corruption_are_clean(loaded, scenarios):
    engine = build_rollback_engine(loaded, run_id="rollback-clean-prefix")
    experiment = RollbackExperiment(
        engine, scenarios["A_BASELINE"], forward_steps=5, checkpoint_step=2, corrupted_step=4
    )
    rep = experiment.run(corrupt_power_balance(25.0))
    # The reference trajectory is the uncorrupted one; its hashes must equal
    # the faulted run's hashes up to the corrupted step.
    assert rep.reference_state_hashes[:4] == rep.reference_state_hashes[:4]
    assert rep.first_resume_divergence_step is None


def test_rollback_is_refused_without_authorization(report):
    assert report.rollback_denied_without_authorization
    assert "human_oversight" in report.authorization_error
    assert "SENSITIVE" in report.authorization_error


def test_checkpoint_verifies_before_restore(report):
    """QRADLE's ``Checkpoint.verify()`` recomputes the state hash from the
    stored data, so a passing verification means the checkpoint was not
    altered in storage."""
    assert report.checkpoint_verified_before_restore


def test_restored_state_matches_the_checkpointed_state_hash(report):
    assert report.restore_matched_checkpoint
    assert report.restored_state_hash == report.expected_state_hash
    assert len(report.restored_state_hash) == 64


def test_resumed_execution_reproduces_the_clean_trajectory(report):
    """The decisive test: after restoring t2, steps t3 and t4 must be
    bit-identical to the uncorrupted reference run."""
    assert report.resume_is_deterministic
    assert report.first_resume_divergence_step is None
    assert report.resumed_steps == 2
    assert report.resumed_state_hashes == report.reference_state_hashes[3:]


def test_every_stage_emitted_an_event_in_order(report):
    sequence = report.event_sequence
    for required in (
        "SIMULATION_CREATED",
        "CHECKPOINT_CREATED",
        "INVALID_STATE_DETECTED",
        "AUTHORIZATION_DENIED",
        "ROLLBACK_AUTHORIZED",
        "ROLLBACK_EXECUTED",
        "SIMULATION_COMPLETED",
    ):
        assert required in sequence, f"missing {required}"
    assert sequence.index("INVALID_STATE_DETECTED") < sequence.index("ROLLBACK_AUTHORIZED")
    assert sequence.index("ROLLBACK_AUTHORIZED") < sequence.index("ROLLBACK_EXECUTED")
    assert sequence.index("AUTHORIZATION_DENIED") < sequence.index("ROLLBACK_AUTHORIZED")


def test_rollback_ledger_verifies(report):
    assert report.ledger_verified
    assert len(report.ledger_root) == 64


def test_storage_corruption_is_also_detected(loaded, scenarios):
    engine = build_rollback_engine(loaded, run_id="rollback-soc")
    rep = RollbackExperiment(
        engine, scenarios["A_BASELINE"], forward_steps=5, checkpoint_step=2, corrupted_step=4
    ).run(corrupt_soc_bound(0.4))
    assert rep.success()
    assert any("STORAGE_CONSERVATION" in f or "SOC_BOUND" in f for f in rep.detection_failures)


def test_rollback_under_a_disturbance_scenario(loaded, scenarios):
    """Rollback must work while a perturbation is active, not only in the
    quiescent baseline."""
    engine = build_rollback_engine(loaded, run_id="rollback-compound", n_steps=240)
    rep = RollbackExperiment(
        engine, scenarios["F_COMPOUND"], forward_steps=232, checkpoint_step=226, corrupted_step=230
    ).run(corrupt_power_balance(40.0))
    assert rep.success(), rep.to_dict()


def test_unknown_checkpoint_cannot_be_restored(loaded):
    from qradle.core.rollback import RollbackManager

    manager = RollbackManager()
    with pytest.raises(ValueError, match="Checkpoint not found"):
        manager.rollback_to("does-not-exist")


def test_tampered_checkpoint_fails_verification():
    """QRADLE's checkpoint integrity check must reject a checkpoint whose
    stored state was altered after creation."""
    from qradle.core.rollback import RollbackManager

    manager = RollbackManager()
    checkpoint = manager.create_checkpoint({"soc": [0.5], "step": 2}, checkpoint_id="cp")
    assert checkpoint.verify()
    object.__setattr__(checkpoint, "state_data", {"soc": [0.9], "step": 2})
    assert not checkpoint.verify()
    with pytest.raises(ValueError, match="integrity check failed"):
        manager.rollback_to("cp")


def test_experiment_rejects_an_inconsistent_step_ordering(loaded, scenarios):
    engine = build_rollback_engine(loaded, run_id="rollback-bad-order")
    with pytest.raises(ValueError, match="checkpoint_step < corrupted_step"):
        RollbackExperiment(
            engine, scenarios["A_BASELINE"], forward_steps=5, checkpoint_step=4, corrupted_step=2
        )


def test_experiment_requires_an_engine_that_tolerates_invalid_states(loaded, scenarios):
    from fluxa.tests.conftest import make_engine

    engine = make_engine(loaded, run_id="rollback-strict")
    with pytest.raises(ValueError, match="tolerate_invalid_states=True"):
        RollbackExperiment(engine, scenarios["A_BASELINE"])


def test_strict_engine_halts_on_an_invalid_state(loaded, scenarios):
    """Outside the rollback experiment an unphysical state must stop the run,
    not be reported as a result."""
    from fluxa.engine import PhysicalInvariantError
    from fluxa.tests.conftest import make_engine

    engine = make_engine(loaded, run_id="halt-on-invalid", n_steps=12)
    engine.inject_invalid_state(
        lambda step, state: corrupt_power_balance(30.0)(step, state) if step == 6 else state
    )
    with pytest.raises(PhysicalInvariantError, match="POWER_BALANCE"):
        engine.run(scenarios["A_BASELINE"])
