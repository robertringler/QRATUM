"""Unit tests for QuNimbus v6 enhancements."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from quasim.audit.log import audit_event, verify_audit_chain
from quasim.qunimbus.bridge import QNimbusBridge, QNimbusConfig


class FakeHttp:
    """Mock HTTP client for testing."""

    def __init__(self):
        """Initialize fake HTTP client."""

        self.posts = []
        self.downloads = []

    def post_json(self, url, payload, timeout):
        """Mock POST request."""

        self.posts.append((url, payload, timeout))
        query_hash = str(abs(hash(payload.get("query", ""))))[:8]
        return {
            "query_id": f"qid-{query_hash}",
            "status": "success",
            "artifacts": {},
        }

    def download(self, url, dest):
        """Mock file download."""

        self.downloads.append((url, dest))
        return dest


def test_audit_event_with_query_id():
    """Test that audit events include query_id at top level."""

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = f"{tmpdir}/audit.jsonl"

        # Log event with query_id in data
        event_id = audit_event(
            "qnimbus.ascend",
            {"query": "test", "seed": 42, "query_id": "qid-123"},
            log_path=log_path,
        )

        assert event_id is not None

        # Verify query_id is at top level
        with open(log_path) as f:
            line = f.readline()
            event = json.loads(line)
            assert event["query_id"] == "qid-123"
            assert "query_id" in event["data"]


def test_audit_event_with_qid_alias():
    """Test that audit events handle 'qid' alias for query_id."""

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = f"{tmpdir}/audit.jsonl"

        # Log event with 'qid' instead of 'query_id'
        event_id = audit_event(
            "qnimbus.validate",
            {"snapshot": "test.hdf5", "qid": "qid-456"},
            log_path=log_path,
        )

        assert event_id is not None

        # Verify qid is promoted to query_id
        with open(log_path) as f:
            line = f.readline()
            event = json.loads(line)
            assert event["query_id"] == "qid-456"


def test_verify_audit_chain_with_query_ids():
    """Test that audit chain verification works with query_ids."""

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = f"{tmpdir}/audit.jsonl"

        # Create chain with multiple query IDs
        audit_event("event1", {"query_id": "qid-1"}, log_path=log_path)
        audit_event("event2", {"query_id": "qid-2"}, log_path=log_path)
        audit_event("event3", {"query_id": "qid-1"}, log_path=log_path)  # Same query

        # Verify chain
        assert verify_audit_chain(log_path)


def test_verify_audit_chain_detects_corruption():
    """Test that audit chain verification detects corrupted chains."""

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = f"{tmpdir}/audit.jsonl"

        # Create valid chain
        audit_event("event1", {"data": "test1"}, log_path=log_path)
        audit_event("event2", {"data": "test2"}, log_path=log_path)

        # Corrupt the chain
        with open(log_path) as f:
            lines = f.readlines()

        # Modify second event's data
        event2 = json.loads(lines[1])
        event2["data"]["data"] = "corrupted"
        lines[1] = json.dumps(event2) + "\n"

        with open(log_path, "w") as f:
            f.writelines(lines)

        # Verification should fail
        assert not verify_audit_chain(log_path)


def test_dry_run_mode_no_network_calls():
    """Test that dry-run mode doesn't make network calls."""

    from click.testing import CliRunner

    from quasim.qunimbus.cli import cli

    runner = CliRunner(mix_stderr=False)

    # Mock the bridge to verify it's not called
    with patch("quasim.qunimbus.cli.QNimbusBridge") as mock_bridge_class:
        result = runner.invoke(
            cli,
            [
                "ascend",
                "--query",
                "test query",
                "--seed",
                "42",
                "--dry-run",
            ],
        )

        # Dry run should succeed
        assert result.exit_code == 0

        # Bridge should not be instantiated
        mock_bridge_class.assert_not_called()

        # Output should indicate dry run (in JSON output)
        output_lower = result.output.lower()
        assert "dry_run" in output_lower or "valid" in output_lower


def test_dry_run_validates_policy():
    """Test that dry-run mode still validates policy."""

    from click.testing import CliRunner

    from quasim.qunimbus.cli import cli

    runner = CliRunner(mix_stderr=False)

    # Try to run with banned query
    result = runner.invoke(
        cli,
        [
            "ascend",
            "--query",
            "bio-weapons design",
            "--dry-run",
        ],
    )

    # Should fail policy check
    assert result.exit_code == 1


def test_strict_mode_fails_on_missing_observable():
    """Test that strict mode fails when observable is missing."""

    from click.testing import CliRunner

    from quasim.qunimbus.cli import cli

    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config with expected observables
        config_path = f"{tmpdir}/config.yml"
        Path(config_path).write_text(
            """version: 1
observables:
  test_observable:
    source: "/nonexistent"
    reduce: "mean"
    expected: 100
    tolerance_abs: 10
"""
        )

        # Create empty snapshot
        snapshot_path = f"{tmpdir}/snapshot.hdf5"
        Path(snapshot_path).touch()

        # Run validation in strict mode
        result = runner.invoke(
            cli,
            [
                "validate",
                "--snapshot",
                snapshot_path,
                "--metrics",
                config_path,
                "--strict",
            ],
        )

        # Should fail in strict mode
        assert result.exit_code in [2, 3]


def test_strict_mode_passes_with_all_observables():
    """Test that strict mode passes when all observables are present."""

    import numpy as np
    from click.testing import CliRunner

    from quasim.io.hdf5 import write_snapshot
    from quasim.qunimbus.cli import cli

    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config
        config_path = f"{tmpdir}/config.yml"
        Path(config_path).write_text(
            """version: 1
observables:
  test_observable:
    source: "/data"
    reduce: "mean"
    expected: 100
    tolerance_abs: 10
"""
        )

        # Create snapshot with matching data
        snapshot_path = f"{tmpdir}/snapshot.hdf5"
        meta = {"version": "1.0"}
        arrays = {"data": np.ones((100,)) * 100}
        write_snapshot(snapshot_path, meta, arrays)

        # Run validation in strict mode
        result = runner.invoke(
            cli,
            [
                "validate",
                "--snapshot",
                snapshot_path,
                "--metrics",
                config_path,
                "--strict",
            ],
        )

        # Should pass in strict mode
        assert result.exit_code == 0


def test_bridge_ascend_documentation():
    """Test that QNimbusBridge.ascend has comprehensive documentation."""

    # Check that docstring exists and contains key information
    docstring = QNimbusBridge.ascend.__doc__
    assert docstring is not None
    assert "seed" in docstring.lower()
    assert "deterministic" in docstring.lower()
    assert "example" in docstring.lower()
    assert "artifact" in docstring.lower()


def test_bridge_config_defaults():
    """Test QNimbusConfig default values."""

    cfg = QNimbusConfig()
    assert cfg.base_url == "https://omni.x.ai/qunimbus/v6"
    assert cfg.timeout_s == 120
    assert cfg.retries == 3


def test_query_id_in_response():
    """Test that bridge.ascend returns query_id."""

    fake_http = FakeHttp()
    bridge = QNimbusBridge(QNimbusConfig(), fake_http)

    resp = bridge.ascend("test query", seed=42)

    assert "query_id" in resp
    assert resp["query_id"].startswith("qid-")


def test_bridge_ascend_with_query_id():
    """Test that bridge.ascend accepts and includes query_id in payload."""

    fake_http = FakeHttp()
    bridge = QNimbusBridge(QNimbusConfig(), fake_http)

    bridge.ascend("test query", seed=42, query_id="custom-qid-123")

    # Verify query_id was included in request payload
    assert len(fake_http.posts) == 1
    url, payload, timeout = fake_http.posts[0]
    assert payload.get("query_id") == "custom-qid-123"


def test_hmac_sign():
    """Test HMAC-SHA256 signing function."""

    from quasim.qunimbus.auth import sign_hmac

    # Test with explicit key
    sig = sign_hmac("test message", "secret-key")
    assert len(sig) > 0
    assert isinstance(sig, str)

    # Signature should be deterministic
    sig2 = sign_hmac("test message", "secret-key")
    assert sig == sig2

    # Different messages should produce different signatures
    sig3 = sign_hmac("different message", "secret-key")
    assert sig != sig3


def test_verify_jwt_without_pyjwt():
    """Test JWT verification graceful degradation without PyJWT."""

    from quasim.qunimbus.auth import verify_jwt

    # Valid JWT structure (3 parts)
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"

    ok, data = verify_jwt(token)
    assert isinstance(data, dict)

    # Should handle malformed tokens
    ok, data = verify_jwt("invalid.token")
    assert not ok
    assert "error" in data


def test_refresh_token_not_implemented():
    """Test that refresh_token raises NotImplementedError (stub)."""

    from quasim.qunimbus.auth import refresh_token

    try:
        refresh_token("current-token")
        raise AssertionError("Should have raised NotImplementedError")
    except NotImplementedError as e:
        assert "Q1 2026" in str(e)


def test_cli_with_query_id_and_qid():
    """Test that CLI accepts both --query-id and --qid."""

    from click.testing import CliRunner

    from quasim.qunimbus.cli import cli

    runner = CliRunner()

    # Test with --query-id
    result = runner.invoke(
        cli,
        [
            "ascend",
            "--query",
            "test query",
            "--query-id",
            "qid-test-123",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    # Extract JSON from output (it's indented, so join all lines starting with '{' or containing json)
    # The JSON starts after the log messages
    json_start = result.output.find("{")
    assert json_start >= 0, f"No JSON in output: {result.output}"
    json_str = result.output[json_start:]
    output = json.loads(json_str)
    assert output.get("query_id") == "qid-test-123"

    # Test with --qid alias
    result = runner.invoke(
        cli,
        [
            "ascend",
            "--query",
            "test query",
            "--qid",
            "qid-test-456",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    json_start = result.output.find("{")
    assert json_start >= 0, f"No JSON in output: {result.output}"
    json_str = result.output[json_start:]
    output = json.loads(json_str)
    assert output.get("query_id") == "qid-test-456"
