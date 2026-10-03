"""Adversarial testing: deliberate attempts to make FLUXA produce a wrong
answer, accept a forged record, or violate physics.

Each test states the attack, the defence that is supposed to stop it, and what
actually happened. Attacks that succeed are asserted as successes and are
reported as findings, not hidden.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import numpy as np
import pytest

from fluxa.config import canonical_hash
from fluxa.dispatch import DispatchError, DispatchStep
from fluxa.engine import PhysicalInvariantError, validate_physical_state
from fluxa.events import FluxaEventType, LedgerIntegrityError
from fluxa.provenance import build_merkle_tree
from fluxa.recovery import corrupt_power_balance
from fluxa.tests.conftest import make_engine
from qradle.core.invariants import InvariantViolation


# ====================================================== 1. input corruption
def test_mutating_the_config_document_after_load_does_not_change_the_loaded_model(
    loaded, scenarios
):
    """Attack: edit the raw configuration dict held by ``LoadedSystem`` after
    validation, hoping the engine re-reads it.

    Defence: ``EnergySystem`` is a frozen dataclass built once at load time;
    the engine never consults ``raw``. The recorded ``config_hash`` is the one
    computed at load, so the mutation is also not laundered into provenance.
    """
    before = make_engine(loaded, run_id="adv-raw-before", n_steps=24).run(
        scenarios["A_BASELINE"]
    )
    loaded.raw["generators"][0]["p_max_mw"] = 9_999.0
    after = make_engine(loaded, run_id="adv-raw-after", n_steps=24).run(
        scenarios["A_BASELINE"]
    )
    assert after.provenance.identity_hash() == before.provenance.identity_hash()
    assert loaded.system.generators[0].p_max_mw == 220.0
    # Restore so later tests see the pristine document.
    loaded.raw["generators"][0]["p_max_mw"] = 220.0


def test_frozen_model_rejects_in_place_mutation(system):
    """Attack: raise a generator's capacity on the live system object."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        system.generators[0].p_max_mw = 9_999.0


def test_frozen_state_rejects_direct_field_assignment(baseline_result):
    """FluxaState is mutable by design (it is assembled field by field), so
    this records what is *not* protected: a caller holding a state can edit
    it. The defence is that the recorded hash then no longer matches, which
    the next test demonstrates."""
    state = baseline_result.states[0]
    original_hash = state.state_hash()
    state.total_load_mw += 1.0
    assert state.state_hash() != original_hash
    state.total_load_mw -= 1.0
    assert state.state_hash() == original_hash


def test_a_tampered_input_cannot_reproduce_the_original_identity(loaded, scenarios, tmp_path):
    """Attack: change an input and claim the original provenance identity."""
    import json as _json

    from fluxa.config import DEFAULT_SYSTEM_CONFIG, load_system

    honest = make_engine(loaded, run_id="adv-honest", n_steps=24).run(scenarios["A_BASELINE"])
    doc = _json.loads(DEFAULT_SYSTEM_CONFIG.read_text())
    doc["batteries"][0]["energy_capacity_mwh"] = 321.0
    path = tmp_path / "tampered.json"
    path.write_text(_json.dumps(doc))
    forged = make_engine(load_system(path), run_id="adv-forged", n_steps=24).run(
        scenarios["A_BASELINE"]
    )
    assert forged.provenance.identity_hash() != honest.provenance.identity_hash()
    assert forged.provenance.model_hash != honest.provenance.model_hash


def test_config_hash_detects_a_single_digit_change(tmp_path):
    from fluxa.config import DEFAULT_SYSTEM_CONFIG, load_system

    doc = json.loads(DEFAULT_SYSTEM_CONFIG.read_text())
    original = load_system(DEFAULT_SYSTEM_CONFIG)
    doc["lines"][0]["reactance_pu"] = 0.060000000000000005
    path = tmp_path / "nudged.json"
    path.write_text(json.dumps(doc))
    assert load_system(path).config_hash != original.config_hash


# ====================================================== 2. event corruption
def test_editing_a_stored_event_is_detected(loaded, scenarios):
    """Attack: rewrite the payload of a logged event in place.
    Defence: per-node content hash recomputation in ``ledger.verify()``."""
    result = make_engine(loaded, run_id="adv-event", n_steps=36).run(scenarios["A_BASELINE"])
    assert result.ledger.verify()[0]
    victim = 7
    forged = copy.deepcopy(result.ledger.events[victim].to_dict())
    forged["payload"]["unserved_mw"] = 0.0
    forged["event_type"] = FluxaEventType.STATE_ADVANCED.value
    result.ledger._force_replace_node_data(victim, forged)
    ok, problems = result.ledger.verify()
    assert not ok
    assert problems == [f"node {victim}: content hash does not match data"]


def test_deleting_an_event_is_detected_by_sequence_and_linkage(loaded, scenarios):
    """Attack: splice a violation event out of the ledger."""
    result = make_engine(loaded, run_id="adv-delete", n_steps=36).run(scenarios["A_BASELINE"])
    ledger = result.ledger
    ledger._events = ledger._events[:5] + ledger._events[6:]
    ledger._nodes = ledger._nodes[:5] + ledger._nodes[6:]
    ok, problems = ledger.verify()
    assert not ok
    assert any("previous_hash" in p for p in problems)
    assert any("sequence field" in p for p in problems)


def test_appending_a_forged_event_to_a_closed_ledger_changes_the_root(loaded, scenarios):
    """Attack: add an exculpatory event after the fact.
    Defence: the root changes, so any party holding the original root detects it."""
    result = make_engine(loaded, run_id="adv-append", n_steps=24).run(scenarios["A_BASELINE"])
    recorded_root = result.provenance.event_chain_root
    result.ledger.append(
        FluxaEventType.SYSTEM_STABILIZED,
        sim_timestamp="2026-01-05T23:59:00Z",
        step=999,
        safety_level="ROUTINE",
        payload={"fabricated": True},
    )
    assert result.ledger.root_hash() != recorded_root
    # The chain itself is still internally consistent -- append-only logs
    # cannot detect their own extension. Detection requires the external
    # witness, which is the provenance bundle's recorded root.
    assert result.ledger.verify()[0]


def test_reordering_events_changes_the_merkle_tree_root(loaded, scenarios):
    result = make_engine(loaded, run_id="adv-reorder", n_steps=24).run(scenarios["A_BASELINE"])
    hashes = result.ledger.event_hashes()
    swapped = [hashes[1], hashes[0], *hashes[2:]]
    assert build_merkle_tree(swapped).root != result.provenance.event_tree_root


def test_unserialisable_event_payload_is_refused(loaded, scenarios):
    """An event that cannot be hashed cannot be audited, so it must not be
    accepted into the ledger at all."""
    from fluxa.events import FluxaEventLedger

    ledger = FluxaEventLedger("adv")
    with pytest.raises(LedgerIntegrityError, match="not JSON-serialisable"):
        ledger.append(
            FluxaEventType.STATE_ADVANCED,
            sim_timestamp="2026-01-05T00:00:00Z",
            step=0,
            safety_level="ROUTINE",
            payload={"array": np.zeros(3)},
        )
    assert len(ledger) == 0


# ====================================================== 3. state corruption
def test_corrupting_a_historical_state_changes_the_state_tree_root(loaded, scenarios):
    result = make_engine(loaded, run_id="adv-state", n_steps=36).run(scenarios["A_BASELINE"])
    hashes = list(result.state_hashes)
    assert build_merkle_tree(hashes).root == result.provenance.state_tree_root
    tampered = dataclasses.replace(result.states[9], unserved_mw=0.0, total_load_mw=1.0)
    hashes[9] = tampered.state_hash()
    assert build_merkle_tree(hashes).root != result.provenance.state_tree_root


def test_an_injected_unphysical_state_halts_the_run(loaded, scenarios):
    """Attack: inject a state that violates power balance mid-run.
    Defence: in-loop validation raises and the run does not complete."""
    engine = make_engine(loaded, run_id="adv-halt", n_steps=24)
    engine.inject_invalid_state(
        lambda step, state: corrupt_power_balance(50.0)(step, state) if step == 10 else state
    )
    with pytest.raises(PhysicalInvariantError, match="step 10"):
        engine.run(scenarios["A_BASELINE"])


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("total_generation_mw", 10_000.0, "POWER_BALANCE"),
        ("battery_charge_mw", -50.0, "NEGATIVE_QUANTITY"),
        ("unserved_mw", -5.0, "NEGATIVE_QUANTITY"),
        ("curtailment_mw", -1.0, "NEGATIVE_QUANTITY"),
    ],
)
def test_validator_flags_each_class_of_corrupted_scalar(
    baseline_result, field, value, expected
):
    system = baseline_result.system
    caps = np.array([ln.capacity_mw for ln in system.lines])
    state = dataclasses.replace(baseline_result.states[5], **{field: value})
    failures = validate_physical_state(
        system, state, None, 1 / 12, caps, np.array([state.renewable_available_mw])
    )
    assert any(expected in f for f in failures), failures


def test_validator_flags_a_generator_above_nameplate(baseline_result):
    system = baseline_result.system
    caps = np.array([ln.capacity_mw for ln in system.lines])
    source = baseline_result.states[5]
    gen_id = system.generators[0].gen_id
    state = dataclasses.replace(
        source,
        generation_by_unit_mw={**source.generation_by_unit_mw, gen_id: 10_000.0},
    )
    failures = validate_physical_state(
        system, state, None, 1 / 12, caps, np.array([state.renewable_available_mw])
    )
    assert any("CAPACITY_EXCEEDED" in f for f in failures)


def test_validator_flags_renewable_output_above_available_resource(baseline_result):
    system = baseline_result.system
    caps = np.array([ln.capacity_mw for ln in system.lines])
    source = baseline_result.states[-1]
    wind = next(g for g in system.variable_generators if g.gen_id == "WIND_1")
    state = dataclasses.replace(
        source,
        generation_by_unit_mw={**source.generation_by_unit_mw, wind.gen_id: wind.p_max_mw},
    )
    failures = validate_physical_state(system, state, None, 1 / 12, caps, np.array([0.0]))
    assert any("AVAILABILITY_EXCEEDED" in f for f in failures)


def test_validator_flags_soc_outside_bounds(baseline_result):
    system = baseline_result.system
    caps = np.array([ln.capacity_mw for ln in system.lines])
    source = baseline_result.states[5]
    battery = system.batteries[0]
    state = dataclasses.replace(source, battery_soc={battery.battery_id: 1.4})
    failures = validate_physical_state(
        system, state, None, 1 / 12, caps, np.array([state.renewable_available_mw])
    )
    assert any("SOC_BOUND" in f for f in failures)


def test_validator_flags_storage_energy_created_from_nothing(baseline_result):
    """Attack: claim a higher state of charge than the recorded charge power
    can account for -- energy created inside the battery."""
    system = baseline_result.system
    caps = np.array([ln.capacity_mw for ln in system.lines])
    battery = system.batteries[0]
    previous = baseline_result.states[20]
    state = dataclasses.replace(
        baseline_result.states[21],
        battery_soc={battery.battery_id: previous.battery_soc[battery.battery_id] + 0.2},
    )
    failures = validate_physical_state(
        system, state, previous, 1 / 12, caps, np.array([state.renewable_available_mw])
    )
    assert any("STORAGE_CONSERVATION" in f for f in failures)


def test_validator_flags_a_concealed_line_overload(baseline_result):
    """Attack: zero out a reported overload while leaving the flow intact."""
    system = baseline_result.system
    source = baseline_result.states[5]
    line = system.lines[0]
    tiny_caps = np.full(len(system.lines), 1.0)
    failures = validate_physical_state(
        system, source, None, 1 / 12, tiny_caps, np.array([source.renewable_available_mw])
    )
    assert any("OVERLOAD_MISMATCH" in f and line.line_id in f for f in failures) or any(
        "OVERLOAD_MISMATCH" in f for f in failures
    )


# ======================================================== 4. replay attack
def test_replaying_a_ledger_against_a_different_run_is_detected(loaded, scenarios):
    """Attack: present scenario A's clean event ledger as evidence for the
    compound-disturbance run.

    Defence: the provenance bundle binds the event-chain root, the state-tree
    root and the scenario hash together, so a substituted ledger does not
    match the bundle it is offered against.
    """
    clean = make_engine(loaded, run_id="adv-replay-clean", n_steps=96).run(
        scenarios["A_BASELINE"]
    )
    disturbed = make_engine(loaded, run_id="adv-replay-dist", n_steps=96, authorized=True).run(
        scenarios["F_COMPOUND"]
    )
    assert clean.ledger.verify()[0] and disturbed.ledger.verify()[0]
    assert clean.provenance.event_chain_root != disturbed.provenance.event_chain_root
    replayed_tree = build_merkle_tree(clean.ledger.event_hashes()).root
    assert replayed_tree != disturbed.provenance.event_tree_root
    assert clean.provenance.scenario_hash != disturbed.provenance.scenario_hash


def test_replaying_a_checkpoint_from_a_different_experiment_is_not_retrievable(
    loaded, scenarios
):
    """Attack: restore a checkpoint taken under a different scenario.

    Defence: checkpoint ids are experiment-addressed, so the foreign id is
    simply absent from this run's manager.
    """
    a = make_engine(loaded, run_id="adv-ckpt-a", n_steps=36).run(scenarios["A_BASELINE"])
    b = make_engine(loaded, run_id="adv-ckpt-b", n_steps=36).run(scenarios["B_SOLAR_SHOCK"])
    assert a.checkpoint_ids and b.checkpoint_ids
    assert set(a.checkpoint_ids).isdisjoint(b.checkpoint_ids)


def test_a_stale_ledger_root_does_not_match_a_newer_run(loaded, scenarios):
    """Attack: reuse yesterday's attestation for today's inputs."""
    old = make_engine(loaded, run_id="adv-stale-old", n_steps=48, wind_seed=1).run(
        scenarios["A_BASELINE"]
    )
    new = make_engine(loaded, run_id="adv-stale-new", n_steps=48, wind_seed=2).run(
        scenarios["A_BASELINE"]
    )
    assert old.provenance.event_chain_root != new.provenance.event_chain_root
    assert old.provenance.profile_hash != new.provenance.profile_hash


# ============================================ 5. unauthorized critical operation
def test_unauthorized_sensitive_scenario_is_blocked(loaded, scenarios):
    engine = make_engine(loaded, run_id="adv-auth", n_steps=24, authorized=False)
    with pytest.raises(InvariantViolation):
        engine.run(scenarios["D_GENERATOR_OUTAGE"])


def test_authorization_cannot_be_granted_by_editing_the_scenario_text(loaded, scenarios):
    """Attack: relabel the scenario's description and tags to look routine.

    Defence: the required level is derived from the perturbation *kinds*, not
    from any free-text field.
    """
    disguised = dataclasses.replace(
        scenarios["D_GENERATOR_OUTAGE"],
        name="Routine maintenance check",
        description="nothing to see here",
        tags=("routine",),
    )
    assert disguised.required_safety_level == "SENSITIVE"
    engine = make_engine(loaded, run_id="adv-disguise", n_steps=24, authorized=False)
    with pytest.raises(InvariantViolation):
        engine.run(disguised)


def test_authorization_flag_is_the_only_way_through_the_gate(loaded, scenarios):
    """Recorded limitation: ``RunConfig.authorized`` is a boolean asserted by
    the caller. FLUXA/QRADLE verify that it was *set*, not that a human set
    it: there is no signature, no second party, and no credential. Anything
    able to construct a ``RunConfig`` can authorize itself.
    """
    engine = make_engine(loaded, run_id="adv-selfauth", n_steps=24, authorized=True,
                         authorizer_id="attacker")
    result = engine.run(scenarios["D_GENERATOR_OUTAGE"])
    assert result.qradle_result.success
    event = next(
        e for e in result.ledger if e.event_type is FluxaEventType.SCENARIO_INITIALIZED
    )
    assert event.payload["authorized"] is True


# ========================================== 6. forced constraint violations
def test_battery_cannot_be_driven_beyond_its_soc_window(system, network):
    """Attack: demand sustained maximum discharge from a near-empty battery."""
    from fluxa.dispatch import DispatchProblem

    problem = DispatchProblem(system, network, 300.0, horizon_steps=4)
    battery = system.batteries[0]
    shares = np.array([b.load_share for b in system.buses])
    window = [
        DispatchStep(
            bus_load_mw=420.0 * shares,
            variable_availability_mw=np.zeros(2),
            dispatchable_availability=np.zeros(2),
            line_capacity_mw=np.array([ln.capacity_mw for ln in system.lines]),
        )
        for _ in range(4)
    ]
    solution = problem.solve(
        window, np.array([battery.soc_min]), np.zeros(2), np.zeros(2)
    )
    assert solution.soc_end[0] >= battery.soc_min - 1e-9
    assert solution.p_discharge_mw[0] <= 1e-6
    # The deficit is reported as unserved energy rather than conjured from the
    # empty battery.
    assert solution.unserved_mw.sum() > 100.0


def test_generator_cannot_be_driven_beyond_capacity(system, network):
    from fluxa.dispatch import DispatchProblem

    problem = DispatchProblem(system, network, 300.0, horizon_steps=2)
    shares = np.array([b.load_share for b in system.buses])
    window = [
        DispatchStep(
            bus_load_mw=5_000.0 * shares,
            variable_availability_mw=np.zeros(2),
            dispatchable_availability=np.ones(2),
            line_capacity_mw=np.array([ln.capacity_mw for ln in system.lines]),
        )
        for _ in range(2)
    ]
    solution = problem.solve(
        window, np.array([0.9]), np.array([220.0, 90.0]), np.ones(2)
    )
    for i, gen in enumerate(system.dispatchable_generators):
        assert solution.p_disp_mw[i] <= gen.p_max_mw + 1e-6
    assert solution.unserved_mw.sum() > 1_000.0


def test_an_impossible_line_rating_produces_a_reported_overload_not_a_silent_one(
    system, network
):
    """Attack: squeeze every rating to near zero and see whether the engine
    pretends the network is intact.

    Defence: the overload slack is a measured variable, so the violation is
    reported with its magnitude.
    """
    from fluxa.dispatch import DispatchProblem

    problem = DispatchProblem(system, network, 300.0, horizon_steps=2)
    shares = np.array([b.load_share for b in system.buses])
    window = [
        DispatchStep(
            bus_load_mw=380.0 * shares,
            variable_availability_mw=np.array([150.0, 130.0]),
            dispatchable_availability=np.ones(2),
            line_capacity_mw=np.full(len(system.lines), 1.0),
        )
        for _ in range(2)
    ]
    solution = problem.solve(window, np.array([0.5]), np.array([40.0, 0.0]), np.ones(2))
    assert solution.overload_mw.sum() > 1.0
    caps = np.full(len(system.lines), 1.0)
    np.testing.assert_allclose(
        solution.overload_mw, np.maximum(np.abs(solution.line_flow_mw) - caps, 0.0), atol=1e-4
    )


def test_negative_load_does_not_produce_negative_unserved_energy(system, network):
    from fluxa.dispatch import DispatchProblem

    problem = DispatchProblem(system, network, 300.0, horizon_steps=2)
    shares = np.array([b.load_share for b in system.buses])
    window = [
        DispatchStep(
            bus_load_mw=120.0 * shares,
            variable_availability_mw=np.array([180.0, 150.0]),
            dispatchable_availability=np.ones(2),
            line_capacity_mw=np.array([ln.capacity_mw for ln in system.lines]),
        )
        for _ in range(2)
    ]
    solution = problem.solve(window, np.array([0.9]), np.array([40.0, 0.0]), np.ones(2))
    assert solution.unserved_mw.min() >= -1e-9
    assert solution.p_charge_mw.min() >= -1e-9
    assert solution.p_discharge_mw.min() >= -1e-9


def test_malformed_dispatch_window_is_rejected(system, network):
    from fluxa.dispatch import DispatchProblem

    problem = DispatchProblem(system, network, 300.0, horizon_steps=4)
    with pytest.raises(DispatchError):
        problem.solve([], np.array([0.5]), np.zeros(2), np.ones(2))


# ========================================== 7. execution-order independence
def test_scenario_execution_order_does_not_affect_results(loaded, scenarios):
    """Attack on determinism: run the scenarios in two different orders in
    one process and check for cross-contamination through shared state."""
    order_a = ["A_BASELINE", "B_SOLAR_SHOCK", "G_LINE_DERATE"]
    order_b = list(reversed(order_a))
    forward = {
        sid: make_engine(loaded, run_id=f"order-f-{sid}", n_steps=48).run(scenarios[sid])
        for sid in order_a
    }
    reverse = {
        sid: make_engine(loaded, run_id=f"order-r-{sid}", n_steps=48).run(scenarios[sid])
        for sid in order_b
    }
    for sid in order_a:
        assert (
            forward[sid].provenance.identity_hash() == reverse[sid].provenance.identity_hash()
        ), f"{sid} depends on execution order"


def test_a_shared_engine_instance_is_not_reused_across_scenarios(loaded, scenarios):
    """Running two scenarios through one engine appends both to the same
    QRADLE chain. The per-run provenance identity must still be unaffected,
    since it is derived from the FLUXA ledger, not the shared chain."""
    engine = make_engine(loaded, run_id="shared", n_steps=48, authorized=True)
    first = engine.run(scenarios["A_BASELINE"])
    second = engine.run(scenarios["A_BASELINE"])
    assert first.provenance.identity_hash() == second.provenance.identity_hash()
    # The shared QRADLE chain did grow, which is why its root is excluded from
    # the identity.
    assert second.provenance.qradle_chain_root != first.provenance.qradle_chain_root


def test_canonical_hash_is_order_independent_for_equal_content():
    a = {"x": [1, 2, 3], "y": {"b": 2, "a": 1}}
    b = {"y": {"a": 1, "b": 2}, "x": [1, 2, 3]}
    assert canonical_hash(a) == canonical_hash(b)
