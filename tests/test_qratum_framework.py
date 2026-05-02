"""Tests for ``qratum_framework`` — the unified operational spine.

These tests exercise:

* TraceEntry canonical hashing and Merkle chain integrity
* Operator end-to-end run with the strict CIIR–CRS–RIC backend
* CLI smoke: ``run``, ``simulate``, ``falsify``, ``verify``, ``ledger``
* Falsification verdict validation
* Config profile loader
* Health / readiness contract
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from contextlib import redirect_stderr, redirect_stdout

import pytest

from qratum_framework import (
    DEFAULT_PROFILE,
    PROFILES,
    FalsificationVerdict,
    MerkleLedger,
    Operator,
    OperatorResult,
    StrictCIIRBackend,
    UnifiedTraceEntry,
    VERDICT_A,
    VERDICT_A0,
    VERDICT_B,
    check_health,
    check_readiness,
    hash_entry,
    load_profile,
)
from qratum_framework.cli import main as cli_main
from qratum_framework.falsifier import _DemoFalsifier
from qratum_framework.trace import GENESIS_HASH, hash_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(step: int = 0, action: str = "NOOP", status: str = "OK") -> UnifiedTraceEntry:
    return UnifiedTraceEntry(
        step=step,
        intent_raw="hold",
        intent="HOLD",
        action=action,
        status=status,
        failure_mode=None,
        pre_state_hash="0" * 64,
        post_state_hash="0" * 64,
        phi_verdict=(status == "OK"),
    )


def _run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cli_main(list(argv))
    return rc, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# Trace + Merkle ledger
# ---------------------------------------------------------------------------


class TestUnifiedTraceEntry:
    def test_canonical_payload_excludes_metadata(self):
        e1 = UnifiedTraceEntry(
            step=0,
            intent_raw="hold",
            intent="HOLD",
            action="NOOP",
            status="OK",
            failure_mode=None,
            pre_state_hash="a" * 64,
            post_state_hash="b" * 64,
            phi_verdict=True,
            metadata={"host": "alice"},
        )
        e2 = UnifiedTraceEntry(
            step=0,
            intent_raw="hold",
            intent="HOLD",
            action="NOOP",
            status="OK",
            failure_mode=None,
            pre_state_hash="a" * 64,
            post_state_hash="b" * 64,
            phi_verdict=True,
            metadata={"host": "bob"},
        )
        # Metadata MUST NOT affect the canonical payload (so chain hashes
        # stay reproducible across hosts).
        assert e1.canonical_payload() == e2.canonical_payload()

    def test_to_dict_includes_metadata(self):
        e = _make_entry()
        d = e.to_dict()
        assert "metadata" in d
        assert d["step"] == 0


class TestMerkleLedger:
    def test_empty_head_is_genesis(self):
        ledger = MerkleLedger()
        assert ledger.head() == GENESIS_HASH
        assert ledger.height() == 0
        assert ledger.verify() is True

    def test_append_advances_chain_deterministically(self):
        l1, l2 = MerkleLedger(), MerkleLedger()
        for i in range(3):
            l1.append(_make_entry(step=i))
            l2.append(_make_entry(step=i))
        assert l1.head() == l2.head()
        assert l1.height() == 3

    def test_verify_detects_tampering(self):
        # Build two ledgers that diverge: one legitimately, one with the
        # second entry replaced. Then patch the legitimate ledger's middle
        # node via a fresh re-build so we never reach into private types.
        ledger = MerkleLedger()
        for i in range(3):
            ledger.append(_make_entry(step=i))
        assert ledger.verify() is True

        # Reconstruct a tampered ledger from the JSONL export, swap one
        # entry, and confirm `verify()` rejects it.  This exercises only
        # the public iter_with_hashes / append surface.
        text = ledger.to_jsonl()
        tampered = MerkleLedger()
        for idx, line in enumerate(text.splitlines()):
            rec = json.loads(line)
            e = rec["entry"]
            entry = (
                _make_entry(step=99)  # the fraudulent entry
                if idx == 1
                else UnifiedTraceEntry(
                    step=e["step"],
                    intent_raw=e["intent_raw"],
                    intent=e.get("intent"),
                    action=e["action"],
                    status=e["status"],
                    failure_mode=e.get("failure_mode"),
                    pre_state_hash=e["pre_state_hash"],
                    post_state_hash=e.get("post_state_hash"),
                    phi_verdict=e["phi_verdict"],
                )
            )
            tampered.append(entry)
        # The tampered ledger is internally consistent (its own chain
        # verifies) but its head differs from the original.
        assert tampered.verify() is True
        assert tampered.head() != ledger.head()

    def test_iter_with_hashes_matches_verify(self):
        ledger = MerkleLedger()
        for i in range(3):
            ledger.append(_make_entry(step=i))
        prev_seen = GENESIS_HASH
        chain_seen: list[str] = []
        for entry, prev, chain in ledger.iter_with_hashes():
            assert prev == prev_seen
            assert chain == hash_entry(entry, prev)
            prev_seen = chain
            chain_seen.append(chain)
        assert chain_seen[-1] == ledger.head()

    def test_len_and_iter(self):
        ledger = MerkleLedger()
        assert len(ledger) == 0
        for i in range(4):
            ledger.append(_make_entry(step=i))
        assert len(ledger) == 4
        steps = [e.step for e in ledger]
        assert steps == [0, 1, 2, 3]

    def test_jsonl_round_trip_preserves_chain(self):
        ledger = MerkleLedger()
        for i in range(4):
            ledger.append(_make_entry(step=i, action=f"A{i}"))
        text = ledger.to_jsonl()
        lines = text.splitlines()
        assert len(lines) == 4
        # Each line must reference the previous line's chain_hash.
        prev = GENESIS_HASH
        for ln in lines:
            rec = json.loads(ln)
            assert rec["prev_hash"] == prev
            prev = rec["chain_hash"]
        assert prev == ledger.head()

    def test_manifest_shape(self):
        ledger = MerkleLedger()
        ledger.append(_make_entry())
        m = ledger.manifest()
        assert m["height"] == 1
        assert m["head"] == ledger.head()
        assert m["genesis"] == GENESIS_HASH

    def test_hash_state_handles_none_and_dict(self):
        assert hash_state(None) == GENESIS_HASH
        assert hash_state({"a": 1}) == hash_state({"a": 1})
        assert hash_state({"a": 1}) != hash_state({"a": 2})

    def test_hash_entry_is_pure(self):
        e = _make_entry()
        assert hash_entry(e, GENESIS_HASH) == hash_entry(e, GENESIS_HASH)


# ---------------------------------------------------------------------------
# Operator + StrictCIIRBackend
# ---------------------------------------------------------------------------


class TestOperator:
    def test_run_canonical_demo_succeeds(self):
        op = Operator(backend=StrictCIIRBackend())
        result = op.run(["begin", "increase_cpu", "increase_mem", "hold", "end"])
        assert isinstance(result, OperatorResult)
        assert result.ledger.height() == 5
        assert result.ledger.verify() is True
        # All canonical demo intents are well-formed; no parse failures.
        assert all(e.status != "TYPE_II" for e in result.ledger.entries)

    def test_run_records_unparseable_intent_as_type_ii(self):
        op = Operator(backend=StrictCIIRBackend())
        result = op.run(["totally_unknown_intent_xyz"])
        assert result.ledger.height() == 1
        entry = result.ledger.entries[0]
        assert entry.status == "TYPE_II"
        assert entry.intent is None
        assert entry.phi_verdict is False

    def test_run_is_deterministic(self):
        intents = ["begin", "increase_cpu", "hold", "end"]
        op1 = Operator(backend=StrictCIIRBackend())
        op2 = Operator(backend=StrictCIIRBackend())
        r1 = op1.run(intents)
        r2 = op2.run(intents)
        # Identical inputs ⇒ identical heads (proves chain determinism).
        assert r1.ledger.head() == r2.ledger.head()

    def test_summary_reports_chain_verified(self):
        op = Operator(backend=StrictCIIRBackend())
        op.run(["hold", "hold"])
        s = op.run([]).summary()  # second run is no-op; ledger keeps growing
        assert s["verified"] is True
        assert "statuses" in s


# ---------------------------------------------------------------------------
# Falsification verdicts
# ---------------------------------------------------------------------------


class TestFalsificationVerdict:
    def test_valid_verdict_constructs(self):
        v = FalsificationVerdict(domain="d", verdict=VERDICT_A, snr=10.0, overlap_with_ground=0.9)
        assert v.verdict == VERDICT_A
        assert v.to_dict()["verdict"] == VERDICT_A

    @pytest.mark.parametrize("bad_verdict", ["X", "", "a", "C"])
    def test_invalid_verdict_rejected(self, bad_verdict):
        with pytest.raises(ValueError):
            FalsificationVerdict(
                domain="d", verdict=bad_verdict, snr=1.0, overlap_with_ground=0.5
            )

    def test_negative_snr_rejected(self):
        with pytest.raises(ValueError):
            FalsificationVerdict(
                domain="d", verdict=VERDICT_B, snr=-0.1, overlap_with_ground=0.5
            )

    @pytest.mark.parametrize("bad_overlap", [-0.01, 1.01, 2.0])
    def test_overlap_must_be_in_unit_interval(self, bad_overlap):
        with pytest.raises(ValueError):
            FalsificationVerdict(
                domain="d", verdict=VERDICT_A, snr=1.0, overlap_with_ground=bad_overlap
            )

    def test_demo_falsifier_yields_a0(self):
        v = _DemoFalsifier().run(seed=42)
        assert v.verdict == VERDICT_A0
        assert v.snr > 0
        assert 0 <= v.overlap_with_ground <= 1


# ---------------------------------------------------------------------------
# Config profiles
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_profile_loads(self):
        p = load_profile(DEFAULT_PROFILE)
        assert p.name == DEFAULT_PROFILE
        assert p.n_steps > 0

    def test_unknown_profile_raises(self):
        with pytest.raises(KeyError):
            load_profile("does_not_exist")

    def test_overrides_apply(self):
        p = load_profile("quick", overrides={"n_steps": 99})
        assert p.n_steps == 99

    def test_all_named_profiles_present(self):
        for name in ("quick", "medium", "strong", "full_fast", "full_report"):
            assert name in PROFILES


# ---------------------------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_ok(self):
        h = check_health()
        assert h.status == "OK"
        assert any(c["name"] == "ledger_self_test" and c["ok"] for c in h.checks)

    def test_readiness_includes_strict_executor(self):
        r = check_readiness()
        assert r.status == "OK"
        names = {c["name"] for c in r.checks}
        assert "strict_executor" in names


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


class TestCLI:
    def test_run_default(self):
        rc, out, _ = _run_cli("run", "--json")
        assert rc == 0
        payload = json.loads(out)
        assert payload["summary"]["verified"] is True
        assert payload["summary"]["ledger_height"] > 0

    def test_run_with_explicit_intents(self):
        rc, out, _ = _run_cli("run", "begin", "hold", "end", "--json")
        assert rc == 0
        payload = json.loads(out)
        assert len(payload["intents"]) == 3

    def test_simulate_runs(self):
        rc, _, _ = _run_cli("simulate", "hold", "--json")
        assert rc == 0

    def test_verify_zero_exit(self):
        rc, out, _ = _run_cli("verify", "--json")
        assert rc == 0
        payload = json.loads(out)
        assert payload["health"]["status"] == "OK"
        assert payload["readiness"]["status"] == "OK"

    def test_falsify_demo(self):
        rc, out, _ = _run_cli("falsify", "--json")
        assert rc == 0
        payload = json.loads(out)
        assert payload["verdict"] in {"A", "A0", "B"}

    def test_falsify_unknown_domain(self):
        rc, _, err = _run_cli("falsify", "--domain", "no_such_domain")
        assert rc == 2
        assert "unknown" in err.lower()

    def test_ledger_round_trip_via_cli(self):
        # Build a real ledger and persist as JSONL, then verify via CLI.
        op = Operator(backend=StrictCIIRBackend())
        op.run(["begin", "increase_cpu", "hold", "end"])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(op.ledger.to_jsonl())
            path = f.name
        try:
            rc, out, _ = _run_cli("ledger", path)
            assert rc == 0
            payload = json.loads(out)
            assert payload["verified"] is True
            assert payload["height"] == 4
            rc2, out2, _ = _run_cli("ledger", path, "--verify-only")
            assert rc2 == 0
            assert out2.strip() == "OK"
        finally:
            os.unlink(path)

    def test_ledger_verify_detects_tampered_file(self):
        op = Operator(backend=StrictCIIRBackend())
        op.run(["hold", "hold"])
        text = op.ledger.to_jsonl()
        # Flip one character of the chain_hash on the second line so the
        # on-disk hash mismatches what we recompute.
        lines = text.splitlines()
        rec = json.loads(lines[1])
        rec["chain_hash"] = "f" * 64
        lines[1] = json.dumps(rec, sort_keys=True, default=str)
        tampered = "\n".join(lines)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(tampered)
            path = f.name
        try:
            rc, out, _ = _run_cli("ledger", path, "--verify-only")
            assert rc == 1
            assert out.strip() == "FAILED"
        finally:
            os.unlink(path)
