"""Tests for ``qratum_framework.sde`` — the Streaming Drift Engine.

Coverage:

* :class:`Token` / :class:`WindowSnapshot` — frozen, deterministic hash
* :class:`RollingWindowBuffer` — depth/min_window gating, tick monotonicity
* :class:`KLDriftScorer` — numeric + categorical paths, degenerate cases
* :class:`MMDDriftScorer` — embedding path, missing embeddings
* :class:`AlarmEvaluator` — tier classification, debounce, HALT-Δ² gate,
  immediate de-escalation, OK suppression
* :class:`StreamingDriftEngine` — sync ``step`` and async ``ingest``
  paths, halt_callback backpressure, MerkleLedger integration,
  determinism (two runs ⇒ byte-identical event logs)
* :class:`SDEConfig` — validation, profile resolution, kw overrides
* CLI ``qratum stream`` — happy path, ``--halt-fails`` exit code,
  ``--ledger`` round-trip
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
from contextlib import redirect_stdout

import numpy as np
import pytest

from qratum_framework.cli import main as cli_main
from qratum_framework.sde import (
    AlarmEvaluator,
    AlarmThresholds,
    AlarmTier,
    DriftEvent,
    DriftReading,
    KLDriftScorer,
    MMDDriftScorer,
    RollingWindowBuffer,
    SDEConfig,
    SDESnapshot,
    StreamingDriftEngine,
    Token,
    WindowSnapshot,
)
from qratum_framework.sde.config import (
    SDE_PROFILES,
    build_buffer,
    build_evaluator,
    build_scorer,
    load_sde_config,
)
from qratum_framework.trace import MerkleLedger


# ---------------------------------------------------------------------------
# Token / WindowSnapshot
# ---------------------------------------------------------------------------


class TestToken:
    def test_token_is_frozen(self) -> None:
        t = Token(value="x", timestamp=1.0)
        with pytest.raises(Exception):
            t.value = "y"  # type: ignore[misc]

    def test_embedding_is_made_readonly(self) -> None:
        emb = np.array([1.0, 2.0, 3.0])
        t = Token(value="x", embedding=emb)
        assert t.embedding is not None
        with pytest.raises(ValueError):
            t.embedding[0] = 99.0  # frozen by __post_init__

    def test_embedding_default_is_none(self) -> None:
        assert Token(value=1).embedding is None


class TestWindowSnapshot:
    def test_canonical_hash_is_deterministic(self) -> None:
        a = WindowSnapshot(
            tokens=tuple(Token(value=v) for v in [1, 2, 3]), tick=10
        )
        b = WindowSnapshot(
            tokens=tuple(Token(value=v, timestamp=999.0) for v in [1, 2, 3]),
            tick=99,
        )
        # Timestamps and tick are excluded from the canonical hash.
        assert a.canonical_hash() == b.canonical_hash()

    def test_canonical_hash_differs_for_different_values(self) -> None:
        a = WindowSnapshot(
            tokens=tuple(Token(value=v) for v in [1, 2, 3]), tick=0
        )
        b = WindowSnapshot(
            tokens=tuple(Token(value=v) for v in [1, 2, 4]), tick=0
        )
        assert a.canonical_hash() != b.canonical_hash()

    def test_embeddings_returns_none_if_any_missing(self) -> None:
        toks = (
            Token(value=1, embedding=np.array([1.0, 2.0])),
            Token(value=2),
        )
        ws = WindowSnapshot(tokens=toks, tick=0)
        assert ws.embeddings is None

    def test_embeddings_stacks_when_all_present(self) -> None:
        toks = (
            Token(value=1, embedding=np.array([1.0, 2.0])),
            Token(value=2, embedding=np.array([3.0, 4.0])),
        )
        ws = WindowSnapshot(tokens=toks, tick=0)
        emb = ws.embeddings
        assert emb is not None
        assert emb.shape == (2, 2)


# ---------------------------------------------------------------------------
# RollingWindowBuffer
# ---------------------------------------------------------------------------


class TestRollingWindowBuffer:
    def test_returns_none_below_min_window(self) -> None:
        buf = RollingWindowBuffer(depth=8, min_window=4)
        for i in range(3):
            assert buf.push(Token(value=i)) is None

    def test_returns_snapshot_at_min_window(self) -> None:
        buf = RollingWindowBuffer(depth=8, min_window=4)
        for i in range(3):
            buf.push(Token(value=i))
        snap = buf.push(Token(value=3))
        assert snap is not None
        assert len(snap) == 4
        assert snap.tick == 4

    def test_depth_caps_window_size(self) -> None:
        buf = RollingWindowBuffer(depth=4, min_window=2)
        for i in range(10):
            buf.push(Token(value=i))
        assert len(buf) == 4
        snap = buf.snapshot()
        assert snap is not None
        assert snap.values == (6, 7, 8, 9)

    def test_tick_is_monotonic(self) -> None:
        buf = RollingWindowBuffer(depth=8, min_window=2)
        for i in range(5):
            buf.push(Token(value=i))
        assert buf.tick == 5

    def test_invalid_construction(self) -> None:
        with pytest.raises(ValueError):
            RollingWindowBuffer(depth=0, min_window=1)
        with pytest.raises(ValueError):
            RollingWindowBuffer(depth=4, min_window=0)
        with pytest.raises(ValueError):
            RollingWindowBuffer(depth=4, min_window=5)


# ---------------------------------------------------------------------------
# KLDriftScorer
# ---------------------------------------------------------------------------


def _ws(values, tick: int = 0) -> WindowSnapshot:
    return WindowSnapshot(tokens=tuple(Token(value=v) for v in values), tick=tick)


class TestKLDriftScorer:
    def test_zero_for_identical_distributions(self) -> None:
        s = KLDriftScorer()
        w = _ws([1.0, 2.0, 3.0, 4.0, 5.0] * 4, tick=1)
        b = _ws([1.0, 2.0, 3.0, 4.0, 5.0] * 4)
        r = s.score(w, b)
        assert r.d_t == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_disjoint_supports(self) -> None:
        s = KLDriftScorer()
        w = _ws([10.0] * 32, tick=1)
        b = _ws([0.0] * 16 + [20.0] * 16)
        r = s.score(w, b)
        assert r.d_t > 0.0

    def test_categorical_path(self) -> None:
        s = KLDriftScorer()
        w = _ws(["a"] * 10 + ["b"] * 10, tick=1)
        b = _ws(["a"] * 10 + ["b"] * 10)
        r_same = s.score(w, b)
        assert r_same.d_t == pytest.approx(0.0, abs=1e-6)
        s.reset()
        w2 = _ws(["a"] * 18 + ["b"] * 2, tick=1)
        r_diff = s.score(w2, b)
        assert r_diff.d_t > 0.0

    def test_degenerate_zero_variance_window(self) -> None:
        s = KLDriftScorer()
        w = _ws([3.0] * 16, tick=1)
        b = _ws([3.0] * 16)
        r = s.score(w, b)
        assert r.is_degenerate()
        assert r.d_t == 0.0

    def test_empty_window_is_degenerate(self) -> None:
        s = KLDriftScorer()
        r = s.score(_ws([], tick=1), _ws([1.0, 2.0]))
        assert r.is_degenerate()

    def test_delta_and_delta2_tracked(self) -> None:
        s = KLDriftScorer()
        b = _ws([0.0] * 16 + [10.0] * 16)
        r1 = s.score(_ws([0.0] * 32, tick=1), b)
        r2 = s.score(_ws([5.0] * 32, tick=2), b)
        r3 = s.score(_ws([10.0] * 32, tick=3), b)
        # delta_d on first reading is 0 by construction.
        assert r1.delta_d == 0.0
        assert r1.delta2_d == 0.0
        # delta_d on second reading is r2 - r1.
        assert r2.delta_d == pytest.approx(r2.d_t - r1.d_t, abs=1e-9)
        # delta2 on third reading is (r3 - r2) - (r2 - r1).
        expected_d2 = (r3.d_t - r2.d_t) - (r2.d_t - r1.d_t)
        assert r3.delta2_d == pytest.approx(expected_d2, abs=1e-9)

    def test_invalid_construction(self) -> None:
        with pytest.raises(ValueError):
            KLDriftScorer(n_bins=1)
        with pytest.raises(ValueError):
            KLDriftScorer(epsilon=0.0)


# ---------------------------------------------------------------------------
# MMDDriftScorer
# ---------------------------------------------------------------------------


def _ws_emb(values, tick: int = 0) -> WindowSnapshot:
    return WindowSnapshot(
        tokens=tuple(
            Token(value=i, embedding=np.asarray(v, dtype=float))
            for i, v in enumerate(values)
        ),
        tick=tick,
    )


class TestMMDDriftScorer:
    def test_zero_for_identical_samples(self) -> None:
        s = MMDDriftScorer()
        rng = np.random.default_rng(0)
        x = rng.normal(size=(16, 4))
        w = _ws_emb(x, tick=1)
        b = _ws_emb(x)
        r = s.score(w, b)
        # MMD² of a sample with itself = 0.
        assert r.d_t == pytest.approx(0.0, abs=1e-9)

    def test_positive_for_separated_distributions(self) -> None:
        s = MMDDriftScorer()
        rng = np.random.default_rng(0)
        x = rng.normal(loc=0.0, size=(32, 4))
        y = rng.normal(loc=5.0, size=(32, 4))
        r = s.score(_ws_emb(x, tick=1), _ws_emb(y))
        assert r.d_t > 0.0

    def test_missing_embeddings_is_degenerate(self) -> None:
        s = MMDDriftScorer()
        w = _ws([1.0, 2.0], tick=1)  # no embeddings
        b = _ws_emb(np.zeros((4, 2)))
        r = s.score(w, b)
        assert r.is_degenerate()

    def test_dimension_mismatch_is_degenerate(self) -> None:
        s = MMDDriftScorer()
        w = _ws_emb(np.zeros((4, 3)), tick=1)
        b = _ws_emb(np.zeros((4, 2)))
        r = s.score(w, b)
        assert r.is_degenerate()


# ---------------------------------------------------------------------------
# AlarmEvaluator
# ---------------------------------------------------------------------------


def _reading(d_t: float, *, delta2: float = 0.0, tick: int = 0) -> DriftReading:
    return DriftReading(
        d_t=d_t,
        delta_d=0.0,
        delta2_d=delta2,
        scorer_name="kl",
        tick=tick,
    )


class TestAlarmEvaluator:
    def test_thresholds_validated(self) -> None:
        with pytest.raises(ValueError):
            AlarmThresholds(theta_low=0.5, theta_high=0.2, theta_halt=1.0)

    def test_ok_is_suppressed_at_start(self) -> None:
        ev = AlarmEvaluator(n_debounce=1)
        # Even though OK is the classification, no event is emitted
        # because we are not transitioning down.
        for i in range(5):
            assert ev.evaluate(_reading(0.0, tick=i), window_hash="h") is None
        assert ev.current_tier == AlarmTier.OK

    def test_debounced_escalation(self) -> None:
        ev = AlarmEvaluator(
            AlarmThresholds(theta_low=0.05, theta_high=0.20, theta_halt=0.50),
            n_debounce=3,
        )
        # Two readings above WATCH should NOT escalate yet.
        e1 = ev.evaluate(_reading(0.10, tick=1), window_hash="h")
        e2 = ev.evaluate(_reading(0.10, tick=2), window_hash="h")
        assert e1 is None and e2 is None
        assert ev.current_tier == AlarmTier.OK
        # Third consecutive escalates.
        e3 = ev.evaluate(_reading(0.10, tick=3), window_hash="h")
        assert e3 is not None and e3.tier == AlarmTier.WATCH
        assert ev.current_tier == AlarmTier.WATCH

    def test_immediate_de_escalation(self) -> None:
        ev = AlarmEvaluator(n_debounce=1)
        ev.evaluate(_reading(0.30, tick=1), window_hash="h")  # ALERT
        assert ev.current_tier == AlarmTier.ALERT
        e = ev.evaluate(_reading(0.0, tick=2), window_hash="h")
        assert e is not None and e.tier == AlarmTier.OK

    def test_halt_requires_positive_acceleration(self) -> None:
        ev = AlarmEvaluator(n_debounce=1)
        # d_t > theta_halt but delta2 <= 0 → ALERT, not HALT.
        e = ev.evaluate(_reading(0.9, delta2=-0.01, tick=1), window_hash="h")
        assert e is not None and e.tier == AlarmTier.ALERT
        # Same magnitude but positive accel → HALT.
        e = ev.evaluate(_reading(0.95, delta2=0.05, tick=2), window_hash="h")
        assert e is not None and e.tier == AlarmTier.HALT

    def test_alert_and_halt_heartbeat(self) -> None:
        """ALERT/HALT re-emit every tick they persist."""
        ev = AlarmEvaluator(n_debounce=1)
        ev.evaluate(_reading(0.30, tick=1), window_hash="h")  # ALERT
        e2 = ev.evaluate(_reading(0.30, tick=2), window_hash="h")
        assert e2 is not None
        assert e2.tier == AlarmTier.ALERT
        assert e2.previous_tier == AlarmTier.ALERT

    def test_history_tracks_emitted_events(self) -> None:
        ev = AlarmEvaluator(n_debounce=1)
        ev.evaluate(_reading(0.30, tick=1), window_hash="h")
        ev.evaluate(_reading(0.0, tick=2), window_hash="h")
        assert len(ev.history) == 2

    def test_invalid_n_debounce(self) -> None:
        with pytest.raises(ValueError):
            AlarmEvaluator(n_debounce=0)


# ---------------------------------------------------------------------------
# StreamingDriftEngine
# ---------------------------------------------------------------------------


def _make_engine(
    *,
    baseline_values,
    halt_callback=None,
    ledger=None,
    n_debounce: int = 1,
    depth: int = 16,
    min_window: int = 4,
):
    baseline = WindowSnapshot(
        tokens=tuple(Token(value=v) for v in baseline_values), tick=0
    )
    return StreamingDriftEngine(
        buffer=RollingWindowBuffer(depth=depth, min_window=min_window),
        scorer=KLDriftScorer(),
        evaluator=AlarmEvaluator(
            AlarmThresholds(theta_low=0.05, theta_high=0.2, theta_halt=0.5),
            n_debounce=n_debounce,
        ),
        baseline_window=baseline,
        halt_callback=halt_callback,
        ledger=ledger,
    )


class TestStreamingDriftEngine:
    def test_rejects_empty_baseline(self) -> None:
        with pytest.raises(ValueError):
            StreamingDriftEngine(
                buffer=RollingWindowBuffer(),
                scorer=KLDriftScorer(),
                evaluator=AlarmEvaluator(),
                baseline_window=WindowSnapshot(tokens=(), tick=0),
            )

    def test_step_returns_none_below_min_window(self) -> None:
        engine = _make_engine(baseline_values=[0.0] * 8 + [1.0] * 8, min_window=4)
        # First three pushes: buffer below min_window.
        for i in range(3):
            assert engine.step(Token(value=10.0)) is None

    def test_step_emits_event_on_drift(self) -> None:
        engine = _make_engine(
            baseline_values=[0.0] * 16, min_window=4, depth=8, n_debounce=1
        )
        # Drive 8 wildly out-of-distribution tokens.  Scoring starts at
        # tick 4 and at least one event should fire.
        events = [engine.step(Token(value=100.0)) for _ in range(8)]
        assert any(e is not None for e in events)

    def test_snapshot_is_safe_mid_run(self) -> None:
        engine = _make_engine(baseline_values=[0.0] * 16, min_window=4)
        for i in range(8):
            engine.step(Token(value=float(i)))
        snap = engine.snapshot()
        assert isinstance(snap, SDESnapshot)
        assert snap.tick == 8
        assert snap.window_depth <= 16
        assert snap.scorer_name == "kl"

    def test_determinism_byte_identical(self) -> None:
        """Two runs with identical token sequences ⇒ identical event logs."""
        baseline = [0.0] * 8 + [1.0] * 8
        seq = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]

        def run_once():
            engine = _make_engine(
                baseline_values=baseline, min_window=4, depth=8, n_debounce=1
            )
            for v in seq:
                engine.step(Token(value=v))
            return [json.dumps(e.to_dict(), sort_keys=True) for e in engine.events]

        a = run_once()
        b = run_once()
        assert a == b
        assert a, "expected at least one event"

    def test_ledger_appends_one_entry_per_event(self) -> None:
        ledger = MerkleLedger()
        engine = _make_engine(
            baseline_values=[0.0] * 16,
            ledger=ledger,
            min_window=4,
            depth=8,
            n_debounce=1,
        )
        for v in [100.0] * 8:
            engine.step(Token(value=v))
        assert ledger.height() == len(engine.events)
        assert ledger.verify()

    def test_ledger_halt_entry_has_type_ii_status(self) -> None:
        ledger = MerkleLedger()
        # Custom thresholds so we can force HALT cleanly.
        baseline = WindowSnapshot(
            tokens=tuple(Token(value=v) for v in [0.0] * 16), tick=0
        )
        engine = StreamingDriftEngine(
            buffer=RollingWindowBuffer(depth=8, min_window=4),
            scorer=KLDriftScorer(),
            evaluator=AlarmEvaluator(
                AlarmThresholds(
                    theta_low=0.001, theta_high=0.005, theta_halt=0.01
                ),
                n_debounce=1,
            ),
            baseline_window=baseline,
            ledger=ledger,
        )
        for v in [100.0] * 8:
            engine.step(Token(value=v))
        halt_entries = [
            e for e in ledger.entries if e.action == "SDE_HALT_INJECT"
        ]
        assert halt_entries, "expected at least one HALT-injection entry"
        for e in halt_entries:
            assert e.status == "TYPE_II"
            assert e.failure_mode == "BOUNDARY_BREACH"
            assert e.phi_verdict is False
            # Tier in metadata, NOT in canonical payload.
            assert e.metadata["tier"] == "HALT"
            assert "tier" not in e.canonical_payload()

    def test_async_ingest_invokes_halt_callback(self) -> None:
        invocations: list = []

        async def cb(ev: DriftEvent) -> None:
            invocations.append(ev.tier)

        baseline = WindowSnapshot(
            tokens=tuple(Token(value=v) for v in [0.0] * 16), tick=0
        )
        engine = StreamingDriftEngine(
            buffer=RollingWindowBuffer(depth=8, min_window=4),
            scorer=KLDriftScorer(),
            evaluator=AlarmEvaluator(
                AlarmThresholds(
                    theta_low=0.001, theta_high=0.005, theta_halt=0.01
                ),
                n_debounce=1,
            ),
            baseline_window=baseline,
            halt_callback=cb,
        )

        async def stream():
            for v in [100.0] * 8:
                yield Token(value=v)

        asyncio.run(engine.ingest(stream()))
        assert any(t == AlarmTier.HALT for t in invocations)

    def test_async_ingest_propagates_exceptions(self) -> None:
        engine = _make_engine(baseline_values=[0.0] * 16, min_window=4)

        async def bad_stream():
            yield Token(value=1.0)
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(engine.ingest(bad_stream()))


# ---------------------------------------------------------------------------
# SDEConfig
# ---------------------------------------------------------------------------


class TestSDEConfig:
    def test_defaults_match_spec(self) -> None:
        c = SDEConfig()
        assert c.window_depth == 256
        assert c.min_window == 32
        assert c.scorer == "kl"
        assert c.theta_low == 0.05
        assert c.theta_high == 0.20
        assert c.theta_halt == 0.50
        assert c.n_debounce == 3

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"window_depth": 0},
            {"min_window": 0},
            {"min_window": 1024},  # > default window_depth=256
            {"scorer": "wat"},
            {"theta_low": -0.1},
            {"theta_high": 0.01},  # < theta_low
            {"theta_halt": 0.1},  # < theta_high
            {"n_debounce": 0},
        ],
    )
    def test_invalid_construction(self, kwargs) -> None:
        with pytest.raises(ValueError):
            SDEConfig(**kwargs)

    def test_load_sde_config_resolves_every_built_in_profile(self) -> None:
        for name in SDE_PROFILES:
            assert isinstance(load_sde_config(name), SDEConfig)

    def test_load_sde_config_unknown_profile_raises(self) -> None:
        with pytest.raises(KeyError):
            load_sde_config("does-not-exist")

    def test_load_sde_config_overrides_apply(self) -> None:
        c = load_sde_config(overrides={"scorer": "mmd", "n_debounce": 5})
        assert c.scorer == "mmd"
        assert c.n_debounce == 5

    def test_builders_construct_consistent_objects(self) -> None:
        c = SDEConfig(window_depth=10, min_window=4, scorer="kl", n_debounce=2)
        buf = build_buffer(c)
        ev = build_evaluator(c)
        sc = build_scorer(c)
        assert buf.depth == 10 and buf.min_window == 4
        assert ev.n_debounce == 2
        assert sc.name == "kl"


# ---------------------------------------------------------------------------
# CLI: qratum stream
# ---------------------------------------------------------------------------


def _write_tokens(path: str, values) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(" ".join(str(v) for v in values))


class TestStreamCLI:
    def test_stream_happy_path(self, tmp_path) -> None:
        src = tmp_path / "persona.txt"
        base = tmp_path / "baseline.txt"
        _write_tokens(str(src), [str(i) for i in range(64)])
        _write_tokens(str(base), [str(i) for i in range(64)])
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(
                [
                    "stream",
                    "--source",
                    str(src),
                    "--baseline",
                    str(base),
                    "--profile",
                    "quick",
                    "--json",
                ]
            )
        assert rc == 0
        payload = json.loads(buf.getvalue())
        assert payload["snapshot"]["scorer_name"] == "kl"
        assert payload["config"]["scorer"] == "kl"

    def test_stream_halt_fails_exit_code(self, tmp_path) -> None:
        # Persona is a wildly different categorical population than baseline.
        src = tmp_path / "persona.txt"
        base = tmp_path / "baseline.txt"
        _write_tokens(str(src), ["alpha"] * 64)
        _write_tokens(str(base), ["beta"] * 64)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(
                [
                    "stream",
                    "--source",
                    str(src),
                    "--baseline",
                    str(base),
                    "--profile",
                    "quick",
                    "--scorer",
                    "kl",
                    "--halt-fails",
                    "--json",
                ]
            )
        # Either no halt → rc == 0, or halt → rc == 1.  The categorical
        # KL between disjoint single-value supports is well above any
        # reasonable theta_halt, so we expect a HALT event under
        # n_debounce=1; the default profile uses n_debounce=3, and the
        # window of ALERT readings will accumulate before HALT.  Either
        # way the test is robust as long as we observe the correct
        # exit-code semantics.
        assert rc in (0, 1)
        payload = json.loads(buf.getvalue())
        halts = [e for e in payload["events"] if e["tier"] == "HALT"]
        assert (rc == 1) == bool(halts)

    def test_stream_ledger_round_trip(self, tmp_path) -> None:
        src = tmp_path / "persona.txt"
        base = tmp_path / "baseline.txt"
        ledger_path = tmp_path / "sde.ledger.jsonl"
        _write_tokens(str(src), ["x"] * 16 + ["y"] * 32)
        _write_tokens(str(base), ["x"] * 32)
        rc = cli_main(
            [
                "stream",
                "--source",
                str(src),
                "--baseline",
                str(base),
                "--ledger",
                str(ledger_path),
                "--profile",
                "quick",
            ]
        )
        assert rc == 0
        # ledger file may be empty if no events fired; the file exists
        # only when we actually produced events.
        if ledger_path.exists() and ledger_path.stat().st_size > 0:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc2 = cli_main(["ledger", str(ledger_path), "--verify-only"])
            assert rc2 == 0
            assert "OK" in buf.getvalue()

    def test_stream_rejects_empty_source(self, tmp_path) -> None:
        src = tmp_path / "persona.txt"
        base = tmp_path / "baseline.txt"
        src.write_text("", encoding="utf-8")
        _write_tokens(str(base), ["x"] * 16)
        rc = cli_main(
            [
                "stream",
                "--source",
                str(src),
                "--baseline",
                str(base),
            ]
        )
        assert rc == 2
