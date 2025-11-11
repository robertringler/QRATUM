"""Unit tests for QuNimbus bridge module."""

import pytest

from quasim.net.http import HttpClient
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
            "artifacts": {
                "earth_snapshot": {
                    "id": "art-123",
                    "filename": "earth_snapshot.hdf5",
                }
            },
        }

    def download(self, url, dest):
        """Mock file download."""
        self.downloads.append((url, dest))
        return dest


def test_qnimbus_config_defaults():
    """Test QNimbusConfig default values."""
    cfg = QNimbusConfig()
    assert cfg.base_url == "https://omni.x.ai/qunimbus/v6"
    assert cfg.timeout_s == 120
    assert cfg.retries == 3


def test_qnimbus_config_custom():
    """Test QNimbusConfig with custom values."""
    cfg = QNimbusConfig(
        base_url="https://test.example.com",
        timeout_s=60,
        retries=5,
    )
    assert cfg.base_url == "https://test.example.com"
    assert cfg.timeout_s == 60
    assert cfg.retries == 5


def test_ascend_posts_and_returns():
    """Test ascend operation posts to correct endpoint."""
    fake_http = FakeHttp()
    bridge = QNimbusBridge(QNimbusConfig(base_url="https://test"), fake_http)

    resp = bridge.ascend("real world simulation", mode="test", seed=42)

    # Verify POST was made
    assert len(fake_http.posts) == 1
    url, payload, timeout = fake_http.posts[0]
    assert url == "https://test/ascend"
    assert payload["query"] == "real world simulation"
    assert payload["mode"] == "test"
    assert payload["seed"] == 42
    assert timeout == 120

    # Verify response
    assert "query_id" in resp
    assert resp["query_id"].startswith("qid-")
    assert resp["status"] == "success"


def test_fetch_artifact():
    """Test artifact fetching."""
    fake_http = FakeHttp()
    bridge = QNimbusBridge(QNimbusConfig(base_url="https://test"), fake_http)

    path = bridge.fetch_artifact("art-123", "output/artifact.hdf5")

    # Verify download was called
    assert len(fake_http.downloads) == 1
    url, dest = fake_http.downloads[0]
    assert url == "https://test/artifact/art-123"
    assert dest == "output/artifact.hdf5"
    assert path == "output/artifact.hdf5"


def test_ascend_with_defaults():
    """Test ascend with default parameters."""
    fake_http = FakeHttp()
    bridge = QNimbusBridge(QNimbusConfig(), fake_http)

    resp = bridge.ascend("test query")

    # Verify defaults were used
    _, payload, _ = fake_http.posts[0]
    assert payload["mode"] == "singularity"
    assert payload["seed"] == 42


def test_bridge_with_real_http():
    """Test bridge initialization with real HTTP client."""
    http_client = HttpClient()
    bridge = QNimbusBridge(QNimbusConfig(), http_client)

    assert bridge.cfg.base_url == "https://omni.x.ai/qunimbus/v6"
    assert bridge.http == http_client
