"""Determinism: identical inputs must produce identical outputs.

This is QRADLE fatal invariant 8 applied to the FLUXA domain. The tests are
written to *falsify* determinism, not to confirm it: they compare at the
tightest level available (bit-exact state hashes) and locate the first
divergence when there is one.

One known non-determinism is asserted explicitly rather than hidden: QRADLE's
own ``MerkleChain`` timestamps nodes with ``datetime.now()``, so its chain
root differs between runs by construction. ``test_qradle_native_chain_root_is_wall_clock_dependent``
records that as a measured property of the substrate.
"""

from __future__ import annotations

import pytest

from fluxa.provenance import diff_bundles
from fluxa.tests.conftest import make_engine

#: Scenarios replayed for determinism. The compound case is included because
#: it exercises load shedding, outage handling and the ramp relaxation -- the
#: paths most likely to depend on solver state.
REPLAY_SCENARIOS = ("A_BASELINE", "F_COMPOUND", "G_LINE_DERATE")


def _run(loaded, scenarios, scenario_id, run_id):
    return make_engine(loaded, run_id=run_id, n_steps=96).run(scenarios[scenario_id])


@pytest.fixture(scope="module")
def triplicates(loaded, scenarios):
    """Three runs of each replay scenario, under distinct run ids."""
    return {
        sid: [_run(loaded, scenarios, sid, f"replay-{sid}-{k}") for k in (1, 2, 3)]
        for sid in REPLAY_SCENARIOS
    }


@pytest.mark.parametrize("scenario_id", REPLAY_SCENARIOS)
def test_every_timestep_state_is_bit_identical(triplicates, scenario_id):
    runs = triplicates[scenario_id]
    hashes = [r.state_hashes for r in runs]
    for k in (1, 2):
        if hashes[0] != hashes[k]:
            first = next(
                i for i, (a, b) in enumerate(zip(hashes[0], hashes[k], strict=True)) if a != b
            )
            pytest.fail(f"run 1 and run {k + 1} diverge first at step {first}")
    assert hashes[0] == hashes[1] == hashes[2]


@pytest.mark.parametrize("scenario_id", REPLAY_SCENARIOS)
def test_final_state_hash_is_identical(triplicates, scenario_id):
    finals = {r.provenance.final_state_hash for r in triplicates[scenario_id]}
    assert len(finals) == 1


@pytest.mark.parametrize("scenario_id", REPLAY_SCENARIOS)
def test_event_sequence_is_identical(triplicates, scenario_id):
    runs = triplicates[scenario_id]
    sequences = [
        [(e.sequence, e.event_type.value, e.step, e.safety_level) for e in r.ledger]
        for r in runs
    ]
    assert sequences[0] == sequences[1] == sequences[2]


@pytest.mark.parametrize("scenario_id", REPLAY_SCENARIOS)
def test_fluxa_event_chain_root_is_identical(triplicates, scenario_id):
    """FLUXA stamps events with simulation time, so the chain root is a pure
    function of the run -- unlike QRADLE's own chain."""
    roots = {r.provenance.event_chain_root for r in triplicates[scenario_id]}
    assert len(roots) == 1, f"FLUXA chain roots diverged: {roots}"


@pytest.mark.parametrize("scenario_id", REPLAY_SCENARIOS)
def test_merkle_tree_roots_are_identical(triplicates, scenario_id):
    runs = triplicates[scenario_id]
    assert len({r.provenance.event_tree_root for r in runs}) == 1
    assert len({r.provenance.state_tree_root for r in runs}) == 1


@pytest.mark.parametrize("scenario_id", REPLAY_SCENARIOS)
def test_metrics_output_hash_is_identical(triplicates, scenario_id):
    assert len({r.provenance.output_hash for r in triplicates[scenario_id]}) == 1


@pytest.mark.parametrize("scenario_id", REPLAY_SCENARIOS)
def test_provenance_identity_hash_is_identical(triplicates, scenario_id):
    runs = triplicates[scenario_id]
    identities = {r.provenance.identity_hash() for r in runs}
    if len(identities) != 1:
        pytest.fail(f"identity differs; fields: {diff_bundles(runs[0].provenance, runs[1].provenance)}")


@pytest.mark.parametrize("scenario_id", REPLAY_SCENARIOS)
def test_qradle_output_hash_is_identical(triplicates, scenario_id):
    """QRADLE's own ``ExecutionResult.output_hash`` covers the contract output
    and the contract id, neither of which contains a timestamp, so it is
    reproducible."""
    assert len({r.provenance.qradle_output_hash for r in triplicates[scenario_id]}) == 1


def test_qradle_native_chain_root_is_wall_clock_dependent(triplicates):
    """A measured negative result about the substrate, not about FLUXA.

    ``qradle.core.merkle.MerkleChain.append`` stamps each node with
    ``datetime.now(timezone.utc)``, so the chain root is a function of when
    the run happened. Any claim that QRADLE chain roots are reproducible
    across executions is false, and FLUXA therefore excludes
    ``qradle_chain_root`` from its provenance identity.
    """
    runs = triplicates["A_BASELINE"]
    roots = {r.provenance.qradle_chain_root for r in runs}
    assert len(roots) > 1, (
        "QRADLE native chain roots were identical across runs; the wall-clock "
        "dependence documented here may have been fixed upstream, in which "
        "case this test and the report must be updated"
    )


def test_changing_a_seed_changes_the_result(loaded, scenarios):
    a = make_engine(loaded, run_id="seed-a", n_steps=96).run(scenarios["A_BASELINE"])
    b = make_engine(loaded, run_id="seed-b", n_steps=96, wind_seed=99_999).run(
        scenarios["A_BASELINE"]
    )
    assert a.provenance.profile_hash != b.provenance.profile_hash
    assert a.provenance.identity_hash() != b.provenance.identity_hash()


def test_changing_the_horizon_changes_the_result(loaded, scenarios):
    a = make_engine(loaded, run_id="h12", n_steps=96, horizon_steps=12).run(
        scenarios["A_BASELINE"]
    )
    b = make_engine(loaded, run_id="h1", n_steps=96, horizon_steps=1).run(
        scenarios["A_BASELINE"]
    )
    assert a.provenance.identity_hash() != b.provenance.identity_hash()


def test_run_id_alone_does_not_change_the_identity(loaded, scenarios):
    """A label must not be able to change a run's cryptographic identity, or
    comparing two executions of the same experiment would be impossible."""
    a = make_engine(loaded, run_id="label-one", n_steps=96).run(scenarios["A_BASELINE"])
    b = make_engine(loaded, run_id="label-two", n_steps=96).run(scenarios["A_BASELINE"])
    assert a.provenance.identity_hash() == b.provenance.identity_hash()
    assert a.provenance.bundle_hash() != b.provenance.bundle_hash()


def test_exogenous_series_are_reproducible_from_seeds(system, time_grid):
    from fluxa.profiles import build_series

    a = build_series(system, time_grid, load_seed=5, solar_seed=6, wind_seed=7)
    b = build_series(system, time_grid, load_seed=5, solar_seed=6, wind_seed=7)
    import numpy as np

    np.testing.assert_array_equal(a.total_load_mw, b.total_load_mw)
    for key in a.availability_mw:
        np.testing.assert_array_equal(a.availability_mw[key], b.availability_mw[key])


def test_driver_seeds_are_independent(system, time_grid):
    """Changing the wind seed must not perturb the load or solar series, or the
    sensitivity studies would be confounded."""
    import numpy as np

    from fluxa.profiles import build_series

    a = build_series(system, time_grid, load_seed=5, solar_seed=6, wind_seed=7)
    b = build_series(system, time_grid, load_seed=5, solar_seed=6, wind_seed=8)
    np.testing.assert_array_equal(a.total_load_mw, b.total_load_mw)
    np.testing.assert_array_equal(
        a.availability_mw["SOLAR_PV_1"], b.availability_mw["SOLAR_PV_1"]
    )
    assert not np.array_equal(a.availability_mw["WIND_1"], b.availability_mw["WIND_1"])


def test_timestamps_are_derived_arithmetically_not_from_a_clock(time_grid):
    assert time_grid.timestamp(0) == "2026-01-05T00:00:00Z"
    assert time_grid.timestamp(12) == "2026-01-05T01:00:00Z"
    assert time_grid.timestamp(0) == time_grid.timestamp(0)
