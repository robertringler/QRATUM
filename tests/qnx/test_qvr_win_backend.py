import json
import types

import pytest

from qnx.backends.qvr_win import QVRWinBackend
from qnx.types import SimulationConfig


def test_qvr_backend_raises_on_non_windows(monkeypatch):
    monkeypatch.setenv("QVR_EXECUTABLE", "qvr.exe")
    backend = QVRWinBackend()
    config = SimulationConfig(scenario_id="smoke", timesteps=1, seed=0, backend="qvr_win")

    with pytest.raises(RuntimeError):
        backend.run(config)


def test_qvr_backend_handles_process(monkeypatch):
    class DummyCompletedProcess(types.SimpleNamespace):
        pass

    def _fake_run(cmd, input, capture_output, check):
        return DummyCompletedProcess(stdout=json.dumps({"ok": True}).encode(), stderr=b"", returncode=0)

    monkeypatch.setattr("qnx.backends.qvr_win.subprocess.run", _fake_run)
    monkeypatch.setattr("qnx.backends.qvr_win._is_windows", lambda: True)
    monkeypatch.setattr("qnx.backends.qvr_win.Path.exists", lambda self: True)
    monkeypatch.setenv("QVR_EXECUTABLE", "qvr.exe")
    backend = QVRWinBackend()

    config = SimulationConfig(scenario_id="smoke", timesteps=1, seed=0, backend="qvr_win")
    result = backend.run(config)

    assert result["engine"] == "qvr_win"
    assert result["payload"] == {"ok": True}
